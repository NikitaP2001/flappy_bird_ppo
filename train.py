import os
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
import numpy as np
from multiprocessing import Pool
from flappy_env import *
from ppo_model import PPOModel

class PPOTrainer:
    def __init__(self, 
                 learning_rate=0.0003,
                 gamma=0.99,
                 epsilon=0.2,
                 value_coef=0.5,
                 entropy_coef=0.01):
        self.initial_lr = learning_rate
        self.min_lr = 1e-5
        self.lr_decay = 0.9999
        self.replay_buffer = []
        self.buffer_size = 10000
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Torch device:', self.device)
        self.model = PPOModel().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.writer = SummaryWriter('runs/flappy_bird_ppo')
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.steps = 0  # Added: initialize steps counter
        
    def update_model(self, trajectories):
        # Combine all trajectories
        states = []
        actions = []
        rewards = []
        values = []
        log_probs = []

        self.replay_buffer.extend(trajectories)
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer = self.replay_buffer[-self.buffer_size:]
            
        # Sample from buffer
        sample_size = min(len(self.replay_buffer), 32)
        trajectories = random.sample(self.replay_buffer, sample_size)
        
        for trajectory in trajectories:
            states.extend(trajectory[0])
            actions.extend(trajectory[1])
            rewards.extend(trajectory[2])
            values.extend(trajectory[3])
            log_probs.extend(trajectory[4])
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.stack(log_probs).to(self.device)
        old_values = torch.stack(values).to(self.device)  # Changed: stack instead of cat
        
        # Calculate returns and advantages
        returns = self._compute_returns(rewards)
        advantages = returns - old_values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for epoch in range(5):  # Multiple epochs
            action_probs, values = self.model(states)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)

            if torch.any(torch.isnan(action_probs)):
                print(f"Warning: NaN in action probabilities at epoch {epoch}")
                continue
            if not torch.allclose(action_probs.sum(dim=1), torch.ones_like(action_probs.sum(dim=1))):
                print(f"Warning: Action probabilities don't sum to 1 at epoch {epoch}")
                action_probs = F.softmax(action_probs, dim=1)
            
            # Calculate ratio and clipped ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon)
            if torch.any(ratio > 20) or torch.any(ratio < 0.05):
                print(f"Warning: Large policy update ratios detected: min={ratio.min()}, max={ratio.max()}")
            
            # Calculate losses
            policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
            value_loss = ((values - returns) ** 2).mean()
            entropy = dist.entropy().mean()
            
            # Total loss
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            # Update model
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Log metrics
            self.writer.add_scalar('Loss/policy', policy_loss.item(), self.steps)
            self.writer.add_scalar('Loss/value', value_loss.item(), self.steps)
            self.writer.add_scalar('Loss/entropy', entropy.item(), self.steps)
            print(f"Loss at epoch {epoch}: {loss.item()}")
            
            self.steps += 1

    def _compute_returns(self, rewards):
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Add normalization check
        if returns.std() < 1e-8:
            print("Warning: Returns have near-zero standard deviation")
            returns = (returns - returns.mean()) / 1e-8
        else:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        return returns

def run_episode(model_state_dict):
    pygame.init()
    headless = True    
    try:
        env = FlappyBirdEnv(headless=headless)
        device = torch.device("cpu")
        model = PPOModel().to(device)
        clock = pygame.time.Clock() 
        
        if model_state_dict is not None:
            model.load_state_dict(model_state_dict)
        
        states, actions, rewards, values, log_probs = [], [], [], [], []
        state = env.reset()
        total_reward = 0
        temperature = 1.0 
        
        max_steps = 1000
        for _ in range(max_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            with torch.no_grad():
                action_probs, value = model(state_tensor)
                
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            temperature = max(0.5, temperature * 0.995)
            
            next_state, reward, done, _ = env.step(action.item())
            if not headless:
                env.render()
            
            states.append(state)
            actions.append(action.item())
            rewards.append(reward)
            values.append(value.squeeze(0))
            log_probs.append(log_prob)
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        pygame.quit()
        return states, actions, rewards, values, log_probs, total_reward
        
    except Exception as e:
        print(f"Error in run_episode: {str(e)}")
        pygame.quit()
        return None

def validate_trajectory(trajectory):
    states, actions, rewards, values, log_probs, total_reward = trajectory
    
    # Check length consistency
    lengths = [len(states), len(actions), len(rewards), len(values), len(log_probs)]
    if len(set(lengths)) != 1:
        print(f"Warning: Inconsistent lengths in trajectory: {lengths}")
        return False
        
    # Check reward sum
    if not np.isclose(sum(rewards), total_reward):
        print(f"Warning: Total reward mismatch. Sum={sum(rewards)}, Reported={total_reward}")
        return False
        
    return True

def train_parallel(num_processes=20, episodes=1000, load_episode=0):
    trainer = PPOTrainer()

    if load_episode > 0:
        model_path = f'models/ppo_model_{load_episode}.pt'
        if os.path.exists(model_path):
            print(f"Loading model from episode {load_episode}")
            trainer.model.load_state_dict(torch.load(model_path))
        else:
            print(f"Warning: Model file {model_path} not found, starting from scratch")
    
    with Pool(num_processes) as pool:
        start_episode = load_episode + 1
        for episode in range(start_episode, start_episode + episodes):
            try:
                model_state = trainer.model.state_dict()
                trajectories = pool.map(run_episode, [model_state] * num_processes)
                
                pvalid_trajectories = [t for t in trajectories if t is not None and validate_trajectory(t)]
                
                if pvalid_trajectories:
                    trainer.update_model(pvalid_trajectories)
                    
                    if episode % 10 == 0:
                        print('Run trajectories', len(trajectories))
                        avg_reward = np.mean([t[5] for t in pvalid_trajectories])
                        max_reward = max([t[5] for t in pvalid_trajectories])
                        print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Max Reward: {max_reward:.2f}")

                        trainer.model.save(f'models/ppo_model_{episode}.pt')
                        
                        # Early stopping if good performance reached
                        if avg_reward > 50:
                            print("Good performance achieved, saving model...")
                            trainer.model.save('models/ppo_model_best.pt')
                            break
            except Exception as e:
                print(f"Error in episode {episode}: {str(e)}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    models_dir = 'models'
    num_procs = 1
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.startswith('ppo_model_') and f.endswith('.pt')]
        if model_files:
            # Extract episode numbers and find max
            episode_nums = [int(f.split('_')[2].split('.')[0]) for f in model_files]
            last_episode = max(episode_nums)
            print(f"Found latest model at episode {last_episode}")
            train_parallel(num_processes=num_procs, episodes=100000, load_episode=last_episode)
            exit(0)
    else:
        os.makedirs(models_dir)
    print("No existing models found, starting from scratch")
    train_parallel(num_processes=num_procs, episodes=100000, load_episode=0)