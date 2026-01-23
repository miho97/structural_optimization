import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal, Categorical
import numpy as np
import matplotlib.pyplot as plt
from torch import amp

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RolloutBuffer:
    def __init__(self, max_size, num_envs, channels, width, height, action_dtype=torch.long, buffer_device='cpu'):
        self.num_envs = num_envs
        self.max_size = max_size
        self.ptr = 0
        self.buffer_device = buffer_device

        # States: [max_size, num_envs, channels, width, height] - stored on CPU to save GPU memory
        self.states = torch.zeros((max_size, num_envs, channels, width, height), device=buffer_device)
        self.actions = torch.zeros((max_size, num_envs), dtype=action_dtype, device=buffer_device)  
        self.logprobs = torch.zeros((max_size, num_envs), device=buffer_device)
        self.rewards = torch.zeros((max_size, num_envs), device=buffer_device)
        self.state_values = torch.zeros((max_size, num_envs), device=buffer_device)
        self.is_terminals = torch.zeros((max_size, num_envs), dtype=torch.bool, device=buffer_device)

    def store(self, states, actions, logprobs, rewards, state_values, is_terminals):
        if self.ptr >= self.max_size:
            raise IndexError("RolloutBuffer is full")
        # Move tensors to buffer device (CPU) for storage
        self.states[self.ptr] = states.to(self.buffer_device)
        self.actions[self.ptr] = actions.to(self.buffer_device)
        self.logprobs[self.ptr] = logprobs.to(self.buffer_device)
        self.rewards[self.ptr] = rewards.to(self.buffer_device)
        self.state_values[self.ptr] = state_values.to(self.buffer_device)
        self.is_terminals[self.ptr] = is_terminals.to(self.buffer_device)
        self.ptr += 1

    def clear(self):
        self.ptr = 0


import torch.nn.functional as F

class CNNActorCritic(nn.Module):
    def __init__(self, num_channels, num_actions, width, height, has_continuous_action_space, action_std_init, dropout_rate=0.3):
        super(CNNActorCritic, self).__init__()
        self.has_continuous_action_space = has_continuous_action_space

        # Define CNN layers
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=3, padding=1)  # Output: 32 x W x H
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)            # Output: 64 x W x H
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)           # Output: 128 x W x H
        self.bn3 = nn.BatchNorm2d(128)

        # Compute the size after convolution to flatten
        self.width = width
        self.height = height
        self.flatten_dim = 128 * self.width * self.height

        # Fully connected layers (ReLU + 10% dropout for better gradient flow)
        self.fc_actor = nn.Sequential(
            nn.Linear(self.flatten_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_actions),
            nn.Tanh() if has_continuous_action_space else nn.Softmax(dim=-1)
        )

        self.fc_critic = nn.Sequential(
            nn.Linear(self.flatten_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )

        # Initialize weights
        self.apply(self.weights_init_)

        # For continuous action space
        if self.has_continuous_action_space:
            self.action_var = torch.full((num_actions,), action_std_init ** 2).to(device)

    def weights_init_(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

    def forward(self):
        raise NotImplementedError

    def act(self, states):
        """
        Given states, select actions, log probabilities, and state values.
        """
        # Forward pass through CNN
        x = F.relu(self.bn1(self.conv1(states)))  # [batch_size, 32, W, H]
        x = F.relu(self.bn2(self.conv2(x)))       # [batch_size, 64, W, H]
        x = F.relu(self.bn3(self.conv3(x)))       # [batch_size, 128, W, H]
        x = x.view(x.size(0), -1)                 # [batch_size, 128*W*H]

        # Actor
        action_probs = self.fc_actor(x)           # [batch_size, num_actions]
        if self.has_continuous_action_space:
            action_mean = action_probs             # For continuous, interpret as mean
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var)
            dist = MultivariateNormal(action_mean, cov_mat)
            actions = dist.sample()
            action_logprobs = dist.log_prob(actions)
        else:
            dist = Categorical(action_probs)
            actions = dist.sample()
            action_logprobs = dist.log_prob(actions)

        # Critic
        state_values = self.fc_critic(x).squeeze(-1)  # [batch_size]

        return actions, action_logprobs, state_values

    def evaluate(self, states, actions):
        """
        Given states and actions, evaluate log probabilities, state values, and entropy.
        """
        # Forward pass through CNN
        x = F.relu(self.bn1(self.conv1(states)))  # [batch_size, 32, W, H]
        x = F.relu(self.bn2(self.conv2(x)))       # [batch_size, 64, W, H]
        x = F.relu(self.bn3(self.conv3(x)))       # [batch_size, 128, W, H]
        x = x.view(x.size(0), -1)                 # [batch_size, 128*W*H]

        # Actor
        action_probs = self.fc_actor(x)           # [batch_size, num_actions]
        if self.has_continuous_action_space:
            action_mean = action_probs             # For continuous, interpret as mean
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var)
            dist = MultivariateNormal(action_mean, cov_mat)

            action_logprobs = dist.log_prob(actions)
            dist_entropy = dist.entropy()
        else:
            dist = Categorical(action_probs)
            action_logprobs = dist.log_prob(actions)
            dist_entropy = dist.entropy()

        # Critic
        state_values = self.fc_critic(x).squeeze(-1)  # [batch_size]

        return action_logprobs, state_values, dist_entropy


class SpatialAttention(nn.Module):
    """
    Self-attention over spatial locations.
    Allows each grid cell to attend to all other cells, enabling the network
    to reason about global structure (e.g., load paths from force to support).
    
    Uses gated residual connection (starts at 0) for training stability.
    """
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Pre-norm for stability
        self.norm = nn.LayerNorm(channels)
        
        # Q, K, V projections
        self.to_q = nn.Linear(channels, channels, bias=False)
        self.to_k = nn.Linear(channels, channels, bias=False)
        self.to_v = nn.Linear(channels, channels, bias=False)
        self.proj = nn.Linear(channels, channels, bias=False)
        
        # Learnable gate - starts at 0 so attention has no effect initially
        self.gate = nn.Parameter(torch.zeros(1))
        
        # Initialize projections
        for layer in [self.to_q, self.to_k, self.to_v, self.proj]:
            nn.init.orthogonal_(layer.weight, gain=1.0)
        
    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        
        # Reshape: [B, C, H, W] -> [B, N, C]
        x_flat = x.flatten(2).transpose(1, 2)
        x_norm = self.norm(x_flat)
        
        # Compute Q, K, V separately
        q = self.to_q(x_norm).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(x_norm).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(x_norm).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        # q, k, v: [B, num_heads, N, head_dim]
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn - attn.amax(dim=-1, keepdim=True).detach()  # Numerical stability
        attn = torch.softmax(attn, dim=-1)
        
        # Apply attention
        out = attn @ v  # [B, num_heads, N, head_dim]
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        
        # Reshape back: [B, N, C] -> [B, C, H, W]
        out = out.transpose(1, 2).view(B, C, H, W)
        
        # Gated residual (gate starts at 0, learned during training)
        return x + self.gate.tanh() * out


class TwoPathActorCritic(nn.Module):
    """
    Two-path CNN architecture with separate encoders for:
    - Path 1: Density channel (1 channel) - the changing state
    - Path 2: Boundary conditions (4 channels) - forces and normals (static per episode)
    
    Features spatial self-attention after fusion to reason about global structure
    (load paths from forces to supports).
    
    This forces the network to explicitly encode boundary condition information
    rather than ignoring it in favor of the more dynamic density channel.
    """
    def __init__(self, num_actions, width, height, has_continuous_action_space, action_std_init, dropout_rate=0.1, use_attention=False):
        super(TwoPathActorCritic, self).__init__()
        self.has_continuous_action_space = has_continuous_action_space
        self.width = width
        self.height = height
        self.use_attention = use_attention

        # ============ Path 1: Density Encoder (1 channel) ============
        self.density_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        # ============ Path 2: Boundary Condition Encoder (4 channels: normals_x, normals_y, forces_x, forces_y) ============
        self.bc_encoder = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        # ============ Fusion Layer ============
        # Combine features from both paths
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 64 + 64 = 128
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        
        # ============ Spatial Attention ============
        # Self-attention over spatial locations for global reasoning
        self.spatial_attention = SpatialAttention(channels=128, num_heads=4)
        
        # Flatten dimension after fusion
        self.flatten_dim = 128 * width * height
        
        # ============ Actor Head ============
        self.fc_actor = nn.Sequential(
            nn.Linear(self.flatten_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_actions),
            nn.Tanh() if has_continuous_action_space else nn.Softmax(dim=-1)
        )
        
        # ============ Critic Head ============
        self.fc_critic = nn.Sequential(
            nn.Linear(self.flatten_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1)
        )
        
        # Initialize weights
        self.apply(self.weights_init_)
        
        # For continuous action space
        if self.has_continuous_action_space:
            self.action_var = torch.full((num_actions,), action_std_init ** 2).to(device)
    
    def weights_init_(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
    
    def forward_features(self, states):
        """Extract features from both paths, fuse them, and optionally apply spatial attention."""
        # Split state into density and boundary conditions
        density = states[:, 0:1, :, :]   # Channel 0: densities
        bc = states[:, 1:5, :, :]        # Channels 1-4: normals_x, normals_y, forces_x, forces_y
        
        # Encode each path separately
        feat_density = self.density_encoder(density)   # [batch, 64, W, H]
        feat_bc = self.bc_encoder(bc)                  # [batch, 64, W, H]
        
        # Concatenate and fuse
        fused = torch.cat([feat_density, feat_bc], dim=1)  # [batch, 128, W, H]
        fused = self.fusion_conv(fused)                     # [batch, 128, W, H]
        
        # Apply spatial attention for global reasoning (load paths)
        # Run in float32 to avoid mixed precision issues
        if self.use_attention:
            with torch.cuda.amp.autocast(enabled=False):
                fused = self.spatial_attention(fused.float())
        
        # Flatten
        return fused.view(fused.size(0), -1)  # [batch, 128*W*H]
    
    def act(self, states):
        """Given states, select actions, log probabilities, and state values."""
        x = self.forward_features(states)
        
        # Actor
        action_probs = self.fc_actor(x)
        if self.has_continuous_action_space:
            action_mean = action_probs
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var)
            dist = MultivariateNormal(action_mean, cov_mat)
            actions = dist.sample()
            action_logprobs = dist.log_prob(actions)
        else:
            dist = Categorical(action_probs)
            actions = dist.sample()
            action_logprobs = dist.log_prob(actions)
        
        # Critic
        state_values = self.fc_critic(x).squeeze(-1)
        
        return actions, action_logprobs, state_values
    
    def evaluate(self, states, actions):
        """Given states and actions, evaluate log probabilities, state values, and entropy."""
        x = self.forward_features(states)
        
        # Actor
        action_probs = self.fc_actor(x)
        if self.has_continuous_action_space:
            action_mean = action_probs
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var)
            dist = MultivariateNormal(action_mean, cov_mat)
            action_logprobs = dist.log_prob(actions)
            dist_entropy = dist.entropy()
        else:
            dist = Categorical(action_probs)
            action_logprobs = dist.log_prob(actions)
            dist_entropy = dist.entropy()
        
        # Critic
        state_values = self.fc_critic(x).squeeze(-1)
        
        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, state_channels, width, height, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                 has_continuous_action_space, action_std_init=0.6, gae_lambda=0.95, num_envs=1, device=device,
                 use_two_path=False):
        
        self.has_continuous_action_space = has_continuous_action_space
        self.num_envs = num_envs
        self.use_two_path = use_two_path

        self.gamma = gamma
        self.gae_lambda = gae_lambda  
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.state_channels = state_channels
        self.width = width
        self.height = height

        # Initialize buffer on CPU to save GPU memory
        self.buffer = RolloutBuffer(
            max_size=500,  # Reduced from 10000 - only need enough for a few episodes
            num_envs=num_envs, 
            channels=state_channels, 
            width=width, 
            height=height,
            buffer_device='cpu'
        )

        # Initialize ActorCritic - choose architecture based on use_two_path flag
        if use_two_path:
            print("Using TwoPathActorCritic (force encoder architecture)")
            self.policy = TwoPathActorCritic(
                num_actions=action_dim, 
                width=width, 
                height=height, 
                has_continuous_action_space=has_continuous_action_space, 
                action_std_init=action_std_init
            ).to(device)
            
            # Optimizer for two-path architecture
            self.optimizer = optim.Adam([
                {'params': self.policy.density_encoder.parameters(), 'lr': lr_actor},
                {'params': self.policy.bc_encoder.parameters(), 'lr': lr_actor},
                {'params': self.policy.fusion_conv.parameters(), 'lr': lr_actor},
                {'params': self.policy.fc_actor.parameters(), 'lr': lr_actor},
                {'params': self.policy.fc_critic.parameters(), 'lr': lr_critic}
            ])
            
            # Copy policy for old policy
            self.policy_old = TwoPathActorCritic(
                num_actions=action_dim, 
                width=width, 
                height=height, 
                has_continuous_action_space=has_continuous_action_space, 
                action_std_init=action_std_init
            ).to(device)
        else:
            print("Using CNNActorCritic (single-path architecture)")
            self.policy = CNNActorCritic(
                num_channels=state_channels, 
                num_actions=action_dim, 
                width=width, 
                height=height, 
                has_continuous_action_space=has_continuous_action_space, 
                action_std_init=action_std_init
            ).to(device)
            
            self.optimizer = optim.Adam([
                {'params': self.policy.conv1.parameters(), 'lr': lr_actor},
                {'params': self.policy.bn1.parameters(), 'lr': lr_actor},
                {'params': self.policy.conv2.parameters(), 'lr': lr_actor},
                {'params': self.policy.bn2.parameters(), 'lr': lr_actor},
                {'params': self.policy.conv3.parameters(), 'lr': lr_actor},
                {'params': self.policy.bn3.parameters(), 'lr': lr_actor},
                {'params': self.policy.fc_actor.parameters(), 'lr': lr_actor},
                {'params': self.policy.fc_critic.parameters(), 'lr': lr_critic}
            ])
            
            # Copy policy for old policy
            self.policy_old = CNNActorCritic(
                num_channels=state_channels, 
                num_actions=action_dim, 
                width=width, 
                height=height, 
                has_continuous_action_space=has_continuous_action_space, 
                action_std_init=action_std_init
            ).to(device)

        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.95)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        if device.type == 'cuda':
            self.scaler = amp.GradScaler()
        else:
            self.scaler = None

        # Metrics
        self.actor_losses = []
        self.critic_losses = []
        self.entropies = []
        self.grad_norms = []
        self.rewards = []
        self.advantages = []

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.policy.action_var = torch.full((self.policy.action_var.size(0),), new_action_std ** 2).to(device)
            self.policy_old.action_var = torch.full((self.policy_old.action_var.size(0),), new_action_std ** 2).to(device)
        else:
            print("WARNING: Calling PPO::set_action_std() on discrete action space policy")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        if self.has_continuous_action_space:
            new_action_std = max(self.policy.action_var.sqrt().mean().item() - action_std_decay_rate, min_action_std)
            self.set_action_std(new_action_std)
            print(f"Decayed action std to {new_action_std}")
        else:
            print("WARNING: Calling PPO::decay_action_std() on discrete action space policy")

    def select_action(self, states):
        """
        Select actions using the old policy.
        
        Args:
            states (torch.Tensor): Tensor of shape [num_envs, channels, width, height]
        
        Returns:
            Tuple: (actions, logprobs, state_values)
        """
        self.policy_old.eval()  # Disable dropout, use running BatchNorm stats for inference
        with torch.no_grad():
            actions, logprobs, state_values = self.policy_old.act(states)
        return actions, logprobs, state_values

    def compute_gae(self, rewards, state_values, is_terminals, gamma=0.99, gae_lambda=0.95):
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            rewards (np.ndarray): Rewards array of shape [ptr, num_envs]
            state_values (np.ndarray): State values array of shape [ptr, num_envs]
            is_terminals (np.ndarray): Terminal flags array of shape [ptr, num_envs]
            gamma (float): Discount factor
            gae_lambda (float): GAE lambda parameter
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Returns and advantages arrays
        """
        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)
        last_advantage = np.zeros(self.num_envs)

        for step in reversed(range(self.buffer.ptr)):
            mask = 1.0 - is_terminals[step].astype(float)
            if step + 1 < self.buffer.ptr:
                next_state_value = state_values[step + 1]
            else:
                next_state_value = np.zeros(self.num_envs)
            delta = rewards[step] + gamma * next_state_value * mask - state_values[step]
            last_advantage = delta + gamma * gae_lambda * mask * last_advantage
            advantages[step] = last_advantage
            returns[step] = advantages[step] + state_values[step]
        
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
        return returns, advantages

    def update(self):
        """
        Update the policy using the collected rollout buffer.
        """
        # Convert buffer to numpy for GAE
        rewards = self.buffer.rewards[:self.buffer.ptr].cpu().numpy()  # Shape: [ptr, num_envs]
        state_values = self.buffer.state_values[:self.buffer.ptr].cpu().numpy()  # Shape: [ptr, num_envs]
        is_terminals = self.buffer.is_terminals[:self.buffer.ptr].cpu().numpy()  # Shape: [ptr, num_envs]

        # Compute GAE
        returns, advantages = self.compute_gae(rewards, state_values, is_terminals, self.gamma, self.gae_lambda)

        # Flatten the buffer
        states = self.buffer.states[:self.buffer.ptr].reshape(-1, self.state_channels, self.width, self.height).to(device)  # [ptr*num_envs, channels, W, H]
        actions = self.buffer.actions[:self.buffer.ptr].reshape(-1).to(device)  # [ptr*num_envs]
        logprobs = self.buffer.logprobs[:self.buffer.ptr].reshape(-1).to(device)  # [ptr*num_envs]
        returns = returns.reshape(-1)  # [ptr*num_envs]
        advantages = advantages.reshape(-1)  # [ptr*num_envs]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for epoch in range(self.K_epochs):
            with amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                logprobs_new, state_values_new, dist_entropy = self.policy.evaluate(states, actions)
                state_values_new = state_values_new.view(-1)

                ratios = torch.exp(logprobs_new - logprobs)
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

                loss_actor = -torch.min(surr1, surr2).mean()
                loss_critic = self.MseLoss(state_values_new, returns).mean()
                loss_entropy = -dist_entropy.mean()

                loss = loss_actor + loss_critic + 0.1 * loss_entropy

            self.optimizer.zero_grad()
            if self.scaler:
                self.scaler.scale(loss).backward()
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()

            # Logging
            self.actor_losses.append(loss_actor.item())
            self.critic_losses.append(loss_critic.item())
            self.entropies.append(dist_entropy.mean().item())

        # Update the old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()
        self.scheduler.step()

    def save(self, checkpoint_path):
        torch.save({
            'policy_state_dict': self.policy_old.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_path)

    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.policy_old.load_state_dict(checkpoint['policy_state_dict'])
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def plot_metrics(self, save_path=None):
        epochs = range(1, len(self.actor_losses) + 1)

        plt.figure(figsize=(20, 12))

        plt.subplot(2, 2, 1)
        plt.plot(epochs, self.actor_losses, label='Actor Loss')
        plt.xlabel('Update Steps')
        plt.ylabel('Loss')
        plt.title('Actor Loss')
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(epochs, self.critic_losses, label='Critic Loss')
        plt.xlabel('Update Steps')
        plt.ylabel('Loss')
        plt.title('Critic Loss')
        plt.legend()

        # Plot Entropy
        plt.subplot(2, 2, 3)
        plt.plot(epochs, self.entropies, label='Entropy', color='green')
        plt.xlabel('Update Steps')
        plt.ylabel('Entropy')
        plt.title('Policy Entropy')
        plt.legend()

        # Placeholder for Gradient Norms (if implemented)
        # plt.subplot(2, 2, 4)
        # plt.plot(epochs, self.grad_norms, label='Gradient Norm', color='red')
        # plt.xlabel('Update Steps')
        # plt.ylabel('Gradient Norm')
        # plt.title('Gradient Norms')
        # plt.legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
