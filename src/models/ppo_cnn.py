import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal, Categorical
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch import amp

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# -----------------------------------------------------------------------------
# Rollout Buffer (unchanged)
# -----------------------------------------------------------------------------
class RolloutBuffer:
    def __init__(self, max_size, state_dim, num_envs, action_dtype=torch.long):
        self.num_envs = num_envs
        self.max_size = max_size
        self.ptr = 0
        self.states = torch.zeros((max_size, num_envs, state_dim), device=device)
        self.actions = torch.zeros((max_size, num_envs), dtype=action_dtype, device=device)
        self.logprobs = torch.zeros((max_size, num_envs), device=device)
        self.rewards = torch.zeros((max_size, num_envs), device=device)
        self.state_values = torch.zeros((max_size + 1, num_envs), device=device)
        self.is_terminals = torch.zeros((max_size, num_envs), dtype=torch.bool, device=device)

    def store(self, states, actions, logprobs, rewards, state_values, is_terminals):
        if self.ptr >= self.max_size:
            raise IndexError("RolloutBuffer is full")
        self.states[self.ptr] = states
        self.actions[self.ptr] = actions
        self.logprobs[self.ptr] = logprobs
        self.rewards[self.ptr] = rewards
        self.state_values[self.ptr] = state_values
        self.is_terminals[self.ptr] = is_terminals
        self.ptr += 1

    def clear(self):
        self.ptr = 0

# -----------------------------------------------------------------------------
# CNN-based ActorCritic that treats the full state as a 5-channel image
# -----------------------------------------------------------------------------
class ActorCriticCNN(nn.Module):
    def __init__(self, grid_channels, grid_height, grid_width, action_dim, 
                 has_continuous_action_space, action_std_init, dropout_rate=0.3):
        """
        Here we assume the state is a flattened image of shape (grid_channels*grid_height*grid_width)
        For your setup: grid_channels=5, grid_height=7, grid_width=7 (5*7*7 = 245)
        """
        super(ActorCriticCNN, self).__init__()
        self.has_continuous_action_space = has_continuous_action_space
        self.grid_channels = grid_channels
        self.grid_height = grid_height
        self.grid_width = grid_width

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init ** 2).to(device)

        # CNN branch that processes the entire image
        self.cnn = nn.Sequential(
            nn.Conv2d(grid_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((grid_height // 2, grid_width // 2))
        )
        cnn_out_h = grid_height // 2
        cnn_out_w = grid_width // 2
        cnn_output_size = 64 * cnn_out_h * cnn_out_w

        # Actor head
        if has_continuous_action_space:
            self.actor_fc = nn.Sequential(
                nn.Linear(cnn_output_size, 128),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(128, action_dim),
                nn.Tanh()
            )
        else:
            self.actor_fc = nn.Sequential(
                nn.Linear(cnn_output_size, 512),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(512, action_dim),
                nn.Softmax(dim=-1)
            )

        # Critic head
        self.critic_fc = nn.Sequential(
            nn.Linear(cnn_output_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 1)
        )

        self.apply(self.weights_init_)

    def weights_init_(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def process_state(self, state):
        """
        Expects state as a flat tensor of shape (batch, state_dim) where state_dim = grid_channels*grid_height*grid_width.
        Reshapes it into (batch, grid_channels, grid_height, grid_width) and passes it through the CNN.
        """
        batch_size = state.size(0)
        state_image = state.view(batch_size, self.grid_channels, self.grid_height, self.grid_width)
        features = self.cnn(state_image)
        features = features.view(batch_size, -1)
        return features

    def act(self, states):
        features = self.process_state(states)
        if self.has_continuous_action_space:
            action_mean = self.actor_fc(features)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(states.device)
            dist = MultivariateNormal(action_mean, cov_mat)
            actions = dist.sample()
            action_logprobs = dist.log_prob(actions)
        else:
            action_probs = self.actor_fc(features)
            dist = Categorical(action_probs)
            actions = dist.sample()
            action_logprobs = dist.log_prob(actions)
        state_values = self.critic_fc(features).squeeze(-1)
        return actions, action_logprobs, state_values

    def get_distribution(self, states):
        features = self.process_state(states)
        if self.has_continuous_action_space:
            action_mean = self.actor_fc(features)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(states.device)
            return MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor_fc(features)
            return Categorical(probs=action_probs)

    def evaluate(self, states, actions):
        features = self.process_state(states)
        if self.has_continuous_action_space:
            action_mean = self.actor_fc(features)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(states.device)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor_fc(features)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        state_values = self.critic_fc(features).squeeze(-1)
        return action_logprobs, state_values, dist_entropy

    def forward(self):
        raise NotImplementedError

# -----------------------------------------------------------------------------
# PPO Class (instantiation adjusted for the CNN version)
# -----------------------------------------------------------------------------
class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                 has_continuous_action_space, action_std_init=0.6, gae_lambda=0.95,
                 num_envs=32, device=device,
                 # CNN-specific parameters:
                 grid_channels=5, grid_height=7, grid_width=7):
        self.has_continuous_action_space = has_continuous_action_space
        self.num_envs = num_envs
        if has_continuous_action_space:
            self.action_std = action_std_init
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.state_dim = state_dim

        self.actor_losses = []
        self.critic_losses = []
        self.entropies = []
        self.grad_norms = []
        self.rewards = []
        self.advantages = []
        self.device = device
        self.buffer = RolloutBuffer(max_size=10000, state_dim=state_dim, num_envs=num_envs)

        self.policy = ActorCriticCNN(grid_channels, grid_height, grid_width,
                                     action_dim, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr_actor)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10000, gamma=0.95)
        self.policy_old = ActorCriticCNN(grid_channels, grid_height, grid_width,
                                         action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()
        if device.type == 'cuda':
            self.scaler = amp.GradScaler()
        else:
            self.scaler = None

        self.minibatch_size = 128
        self.use_kl_penalty = False

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("WARNING: set_action_std called on discrete action space.")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        if self.has_continuous_action_space:
            self.action_std = max(self.action_std - action_std_decay_rate, min_action_std)
            self.set_action_std(self.action_std)
        else:
            print("WARNING: decay_action_std called on discrete action space.")

    def select_action(self, states):
        self.policy.eval()
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions, logprobs, state_values = self.policy_old.act(states)
        if not self.has_continuous_action_space:
            actions = actions.cpu().numpy().astype(int)
        else:
            actions = actions.cpu().numpy()
        logprobs = logprobs.detach()
        state_values = state_values.detach()
        return actions, logprobs, state_values

    def compute_gae(self, rewards, state_values, is_terminals, gamma=0.99, gae_lambda=0.95):
        T, N = rewards.shape
        advantages = torch.zeros(T, N, device=rewards.device)
        returns = torch.zeros(T, N, device=rewards.device)
        last_advantage = torch.zeros(N, device=rewards.device)
        for step in reversed(range(T)):
            mask = 1.0 - is_terminals[step].float()
            delta = rewards[step] + gamma * state_values[step + 1] * mask - state_values[step]
            last_advantage = delta + gamma * gae_lambda * mask * last_advantage
            advantages[step] = last_advantage
            returns[step] = advantages[step] + state_values[step]
            self.rewards.append(rewards[step].cpu().numpy())
            self.advantages.append(last_advantage.cpu().numpy())
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages

    def update(self):
        rewards = self.buffer.rewards[:self.buffer.ptr]
        state_values = self.buffer.state_values[:self.buffer.ptr + 1]
        is_terminals = self.buffer.is_terminals[:self.buffer.ptr]
        returns, advantages = self.compute_gae(rewards, state_values, is_terminals, self.gamma, self.gae_lambda)
        states = self.buffer.states[:self.buffer.ptr].reshape(-1, self.buffer.states.size(-1)).to(self.device)
        actions = self.buffer.actions[:self.buffer.ptr].reshape(-1).to(self.device)
        logprobs = self.buffer.logprobs[:self.buffer.ptr].reshape(-1).to(self.device)
        returns = returns.reshape(-1)
        advantages = advantages.reshape(-1)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        old_values = self.buffer.state_values[:self.buffer.ptr].reshape(-1).to(self.device)
        dataset = TensorDataset(states, actions, logprobs, returns, advantages, old_values)
        dataloader = DataLoader(dataset, batch_size=self.minibatch_size, shuffle=True)
        self.policy.train()
        for epoch in range(self.K_epochs):
            grad_norms_batch = []
            kl_list = []
            for batch in dataloader:
                batch_states, batch_actions, batch_logprobs, batch_returns, batch_advantages, batch_old_values = batch
                logprobs_new, state_values_new, dist_entropy = self.policy.evaluate(batch_states, batch_actions)
                state_values_new = state_values_new.view(-1)
                ratios = torch.exp(logprobs_new - batch_logprobs)
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                with torch.no_grad():
                    old_dist = self.policy_old.get_distribution(batch_states)
                    new_dist = self.policy.get_distribution(batch_states)
                    kl_values = torch.distributions.kl.kl_divergence(old_dist, new_dist)
                    kl_list.append(kl_values.mean().item())
                loss_actor = -torch.min(surr1, surr2).mean()
                loss_critic = self.MseLoss(state_values_new, batch_returns).mean()
                loss_entropy = -dist_entropy.mean()
                kl_penalty = 0
                if self.use_kl_penalty:
                    kl_penalty = 0.01 * kl_values.mean()
                loss = loss_actor + 0.5 * loss_critic + 0.04 * loss_entropy + kl_penalty
                self.optimizer.zero_grad()
                loss.backward()
                total_norm = 0.0
                for p in self.policy.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                total_norm = total_norm ** 0.5
                grad_norms_batch.append(total_norm)
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
            self.grad_norms.append(np.mean(grad_norms_batch))
            self.actor_losses.append(loss_actor.item())
            self.critic_losses.append(loss_critic.item())
            self.entropies.append(dist_entropy.mean().item())
            if self.use_kl_penalty:
                avg_kl = np.mean(kl_list)
                target_kl = 0.02
                if avg_kl > target_kl:
                    self.eps_clip = max(self.eps_clip * 0.9, 0.1)
                elif avg_kl < target_kl / 2:
                    self.eps_clip = min(self.eps_clip * 1.1, 0.3)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()
        self.scheduler.step()
        self.policy.eval()

    def save(self, checkpoint_path):
        torch.save({
            'policy_state_dict': self.policy_old.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_path)

    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        self.policy_old.load_state_dict(checkpoint['policy_state_dict'])
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def plot_metrics(self, save_path=None):
        import matplotlib.pyplot as plt
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
        plt.subplot(2, 2, 3)
        plt.plot(epochs, self.entropies, label='Entropy', color='green')
        plt.xlabel('Update Steps')
        plt.ylabel('Entropy')
        plt.title('Policy Entropy')
        plt.legend()
        plt.subplot(2, 2, 4)
        plt.plot(epochs, self.grad_norms, label='Gradient Norm', color='red')
        plt.xlabel('Update Steps')
        plt.ylabel('Gradient Norm')
        plt.title('Gradient Norms')
        plt.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_advantage_distribution_window(self, window_size=100, save_path=None):
        import matplotlib.pyplot as plt
        rolling_advantages = self.advantages[-window_size:]
        plt.figure(figsize=(10, 5))
        plt.hist(rolling_advantages, bins=50, alpha=0.7)
        plt.title(f'Advantage Distribution (Last {window_size} Updates)')
        plt.xlabel('Average Advantage')
        plt.ylabel('Frequency')
        plt.show()
        if save_path:
            plt.savefig(save_path)

    def plot_reward_distribution_window(self, window_size=100, save_path=None):
        import matplotlib.pyplot as plt
        rolling_rewards = self.rewards[-window_size:]
        plt.figure(figsize=(10, 5))
        plt.hist(rolling_rewards, bins=50, alpha=0.7)
        plt.title(f'Reward Distribution (Last {window_size} Updates)')
        plt.xlabel('Average Reward')
        plt.ylabel('Frequency')
        plt.show()
        if save_path:
            plt.savefig(save_path)

    def plot_advantage_vs_reward(self, save_path=None):
        import matplotlib.pyplot as plt
        avg_rewards = [sum(r) / len(r) for r in self.rewards]
        avg_advantages = [sum(a) / len(a) for a in self.advantages]
        plt.figure(figsize=(10, 5))
        plt.scatter(avg_rewards, avg_advantages, alpha=0.5, color='orange')
        plt.xlabel('Average Reward per Update')
        plt.ylabel('Average Advantage per Update')
        plt.title('Average Advantage vs. Average Reward')
        plt.grid(True)
        plt.show()
        if save_path:
            plt.savefig(save_path)
