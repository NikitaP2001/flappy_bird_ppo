import torch
import torch.nn as nn
import torch.nn.functional as F

class PPOModel(nn.Module):
    def __init__(self, state_dim=7, hidden_dim=256):  # Increased state_dim and hidden_dim
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # Feature extractor with residual connections
        self.input_layer = nn.Linear(state_dim, hidden_dim)
        
        # Residual blocks for better feature extraction
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(3)
        ])
        
        # Policy head (actor) with attention
        self.policy_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True
        )
        self.policy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),
            nn.Softmax(dim=-1)
        )
        
        # Value head (critic) with separate feature processing
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, state):
        # Initial feature extraction
        x = F.relu(self.input_layer(state))
        x = self.layer_norm(x)
        
        # Process through residual blocks
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Add sequence dimension for attention
        if len(x.shape) == 2:
            x_seq = x.unsqueeze(1)
        else:
            x_seq = x
            
        # Apply attention for policy
        x_att, _ = self.policy_attention(x_seq, x_seq, x_seq)
        x_att = x_att.squeeze(1)
        
        # Generate policy and value
        action_probs = self.policy(x_att)
        value = self.value(x)
        
        return action_probs, value
        
    def save(self, path):
        torch.save(self.state_dict(), path)
        
    def load(self, path):
        self.load_state_dict(torch.load(path))

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        identity = x
        out = self.layers(x)
        out = self.layer_norm(out + identity)
        return F.relu(out)