import numpy as np
import pygame
from game import *

class FlappyBirdEnv:
    def __init__(self, headless=True):
        self.headless = headless
        if headless:
            self.screen = pygame.display.set_mode((1, 1))
        else:
            self.screen = pygame.display.set_mode((SCREEN_WIDHT, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.reset()
    
    def reset(self):
        # Reset game state
        self.bird = Bird()
        self.ground = Ground(0)
        
        # Initialize pipes
        self.pipes = []
        # Initialize with first pair of pipes
        initial_pipes = get_random_pipes(SCREEN_WIDHT * 2)
        self.pipes.extend([initial_pipes[0], initial_pipes[1]])
        
        # Add second pair of pipes
        second_pipes = get_random_pipes(SCREEN_WIDHT * 3)
        self.pipes.extend([second_pipes[0], second_pipes[1]])
        self.score = 0
        self.steps = 0
        return self.get_state()
    
    def _check_collision(self):
        # Check ground collision
        if (self.bird.rect[1] >= SCREEN_HEIGHT - GROUND_HEIGHT or 
            self.bird.rect[1] <= 0):  # Added ceiling check
            return True   
            
        bird_mask = pygame.mask.from_surface(self.bird.image)
        for pipe in self.pipes:
            pipe_mask = pygame.mask.from_surface(pipe.image)
            offset = (pipe.rect[0] - self.bird.rect[0], 
                    pipe.rect[1] - self.bird.rect[1])
            if bird_mask.overlap(pipe_mask, offset):
                return True
        return False
    
    def _check_pipe_passed(self):
        for i in range(0, len(self.pipes), 2):
            pipe = self.pipes[i]
            # If bird passed pipe's right edge
            if self.bird.rect[0] > pipe.rect[0] + PIPE_WIDHT and not hasattr(pipe, 'passed'):
                pipe.passed = True
                return True
        return False
    
    def _update_pipes(self):
        # Remove off-screen pipes
        self.pipes = [pipe for pipe in self.pipes if pipe.rect[0] > -PIPE_WIDHT]
        
        # Add new pipes if needed
        if len(self.pipes) < 4:  # Maintain 2 pipe pairs
            new_pipes = get_random_pipes(SCREEN_WIDHT * 2)
            self.pipes.extend([new_pipes[0], new_pipes[1]])
    
    def render(self):
        if not self.headless:
            self.screen.fill((255, 255, 255))
            # Draw bird
            self.screen.blit(self.bird.image, self.bird.rect)
            # Draw pipes
            for pipe in self.pipes:
                self.screen.blit(pipe.image, pipe.rect)
            # Draw ground
            self.screen.blit(self.ground.image, self.ground.rect)
            pygame.display.update()
            self.clock.tick(30)

    def get_state(self):
        # Normalize all values to [0,1] range
        bird_y = self.bird.rect[1] / SCREEN_HEIGHT
        bird_vel = self.bird.speed / 50
        
        if len(self.pipes) > 0:
            next_pipe = self.pipes[0]
            pipe_dist = min(1.0, (next_pipe.rect[0] - self.bird.rect[0]) / SCREEN_WIDHT)
            pipe_top = next_pipe.rect[1] / SCREEN_HEIGHT
            pipe_bottom = (next_pipe.rect[1] - PIPE_GAP) / SCREEN_HEIGHT
        else:
            pipe_dist = 1.0
            pipe_top = 0.5
            pipe_bottom = 0.5
            # Log all state variables to console
        dist_to_ceiling = self.bird.rect[1] / SCREEN_HEIGHT  # Distance to top
        dist_to_floor = (SCREEN_HEIGHT - GROUND_HEIGHT - self.bird.rect[1]) / SCREEN_HEIGHT  # Distance to ground
        '''
        print(f"Bird Y original: {self.bird.rect[1]}")
        print(f"Bird Y: {bird_y:.3f}, Velocity: {bird_vel:.3f}")
        print(f"Pipe Distance: {pipe_dist:.3f}")
        print(f"Pipe Top: {pipe_top:.3f}, Bottom: {pipe_bottom:.3f}")
        print(f"Distance to Ceiling: {dist_to_ceiling:.3f}, Floor: {dist_to_floor:.3f}")
        '''
        return np.array([bird_y, bird_vel, pipe_dist, dist_to_ceiling, dist_to_floor, pipe_top, pipe_bottom])

    def step(self, action):
        self.steps += 1
        prev_y = self.bird.rect[1]  # Store previous position

        if action == 1:
            self.bird.bump()

        self.bird.update()
        for pipe in self.pipes:
            pipe.update()
        self._update_pipes()

        # Base reward for staying alive
        reward = 0.1

        done = False
          # Stronger rewards/penalties
        if self._check_collision():
            reward = -10.0
            done = True
        elif self._check_pipe_passed():
            reward = 10.0  # Significant reward for passing pipes
            self.score += 1
        else:
            # Add reward for good positioning relative to next pipe
            if len(self.pipes) > 0:
                next_pipe = self.pipes[0]
                pipe_center = next_pipe.rect[1] - PIPE_GAP/2
                bird_vertical_distance = abs(self.bird.rect[1] - pipe_center)
                gap_reward = 0.3 * (1 - bird_vertical_distance/SCREEN_HEIGHT)
                reward += gap_reward
        reward = reward / 10

        return self.get_state(), reward, done, {"score": self.score}