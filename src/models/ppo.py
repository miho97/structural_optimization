
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt

# use configuration file or .env for device
device = torch.device('cpu')

if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
else:
    pass

class RolloutBuffer:
    def __init__(self, max_size, state_dim, num_envs, action_dtype=torch.long):
        self.num_envs = num_envs
        self.max_size = max_size
        self.ptr = 0

        self.states = torch.zeros((max_size, num_envs, state_dim), device=device)
        self.actions = torch.zeros((max_size, num_envs), dtype=action_dtype, device=device)  
        self.logprobs = torch.zeros((max_size, num_envs), device=device)
        self.rewards = torch.zeros((max_size, num_envs), device=device)
        self.state_values = torch.zeros((max_size, num_envs), device=device)
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
        
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init, dropout_rate=0.3):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init ** 2).to(device)

        # Actor Network
        if has_continuous_action_space:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.LayerNorm(64),
                nn.Tanh(),
                nn.Dropout(dropout_rate),
                nn.Linear(64, 64),
                nn.LayerNorm(64),
                nn.Tanh(),
                nn.Dropout(dropout_rate),
                nn.Linear(64, action_dim),
                nn.Tanh()
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.LayerNorm(128),
                nn.Tanh(),
                nn.Dropout(dropout_rate),
                nn.Linear(128, 128),
                nn.LayerNorm(128),
                nn.Tanh(),
                nn.Dropout(dropout_rate),
                nn.Linear(128, action_dim),
                nn.Softmax(dim=-1)
            )

        # Critic Network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1)
        )

        # Initialize weights
        self.apply(self.weights_init_)

    def weights_init_(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
            torch.nn.init.constant_(m.bias, 0)

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError

    def act(self, states):
        if self.has_continuous_action_space:
            action_mean = self.actor(states)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
            actions = dist.sample()
            action_logprobs = dist.log_prob(actions)
        else:
            action_probs = self.actor(states)
            dist = Categorical(action_probs)
            actions = dist.sample()
            action_logprobs = dist.log_prob(actions)
        
        state_values = self.critic(states).squeeze(-1)  
        return actions, action_logprobs, state_values



    def evaluate(self, states, actions):
        if self.has_continuous_action_space:
            action_mean = self.actor(states)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(states)
            dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        state_values = self.critic(states)

        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space,
                 action_std_init=0.6, gae_lambda=0.95, num_envs=1, device = device):
        
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

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.95)

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")

        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")

        print("--------------------------------------------------------------------------------------------")

    def select_action(self, states):
        states = torch.FloatTensor(states).to(self.device)
        
        actions, logprobs, state_vals = self.policy_old.act(states)
        
        return actions.cpu().detach().numpy(), logprobs.cpu().detach().numpy(), state_vals.cpu().detach().numpy()

    def select_action(self, states):

        states = torch.FloatTensor(states).to(self.device)
        actions, logprobs, state_values = self.policy_old.act(states)

        if not self.has_continuous_action_space:
            actions = actions.cpu().numpy().astype(int)
        else:
            actions = actions.cpu().numpy()

        logprobs = logprobs.detach()
        state_values = state_values.detach()

        return actions, logprobs, state_values



    def compute_gae(self, rewards, state_values, is_terminals, gamma=0.99, gae_lambda=0.95):

        rewards = rewards#.cpu().numpy()
        state_values = state_values#.cpu().numpy()
        is_terminals = is_terminals#.cpu().numpy()


        max_steps = self.buffer.ptr  

        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)

        last_advantage = np.zeros(self.num_envs)

        for step in reversed(range(max_steps)):
            mask = 1.0 - is_terminals[step].astype(float)

            if step + 1 < max_steps:
                next_state_value = state_values[step + 1]
            else:
                next_state_value = np.zeros(self.num_envs)

            delta = rewards[step] + gamma * next_state_value * mask - state_values[step]
            last_advantage = delta + gamma * gae_lambda * mask * last_advantage

            advantages[step] = last_advantage
            returns[step] = advantages[step] + state_values[step]

            # self.rewards.append( rewards[step])
            # self.advantages.append( last_advantage)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)

        return returns, advantages


    def update(self):

        rewards = self.buffer.rewards[:self.buffer.ptr].cpu().numpy()  # Shape: (ptr, num_envs)
        state_values = self.buffer.state_values[:self.buffer.ptr].cpu().numpy()  # Shape: (ptr, num_envs)
        is_terminals = self.buffer.is_terminals[:self.buffer.ptr].cpu().numpy()  # Shape: (ptr, num_envs)
        
        states = self.buffer.states[:self.buffer.ptr].reshape(-1, self.buffer.num_envs, self.state_dim).cpu().numpy()  # Shape: (ptr, num_envs, state_dim)
        actions = self.buffer.actions[:self.buffer.ptr].cpu().numpy()  # Shape: (ptr, num_envs)
        logprobs = self.buffer.logprobs[:self.buffer.ptr].cpu().numpy()  
 
        returns, advantages = self.compute_gae(rewards, state_values, is_terminals, self.gamma, self.gae_lambda)

        states = self.buffer.states[:self.buffer.ptr].reshape(-1, self.buffer.states.size(-1)).to(self.device)  # [max_size*num_envs, state_dim]
        actions = self.buffer.actions[:self.buffer.ptr].reshape(-1).to(self.device)  # [max_size*num_envs]
        logprobs = self.buffer.logprobs[:self.buffer.ptr].reshape(-1).to(self.device)  # [max_size*num_envs]
        returns = returns.reshape(-1)  # [max_size*num_envs]
        advantages = advantages.reshape(-1)  # [max_size*num_envs]

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for epoch in range(self.K_epochs):
            logprobs_new, state_values_new, dist_entropy = self.policy.evaluate(states, actions)
            state_values_new = state_values_new.view(-1)


            ratios = torch.exp(logprobs_new - logprobs)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages


            loss_actor = -torch.min(surr1, surr2).mean()

            loss_critic = self.MseLoss(state_values_new, returns).mean()

            loss_entropy = -dist_entropy.mean()

            loss = 2 * loss_actor +  loss_critic + 0.05 * loss_entropy

            self.optimizer.zero_grad()
            loss.backward()

            # Compute gradient norm
            total_norm = 0.0
            for p in self.policy.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            self.grad_norms.append(total_norm)

            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.2)  
            self.optimizer.step()

            self.actor_losses.append(loss_actor.item())
            self.critic_losses.append(loss_critic.item())
            self.entropies.append(dist_entropy.mean().item())

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()
        self.scheduler.step()


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

        # Plot Gradient Norms
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
        rolling_advantages = self.advantages[-window_size:]
        # print( self.advantages)
        plt.figure(figsize=(10, 5))
        plt.hist(rolling_advantages, bins=50,  alpha=0.7)
        plt.title(f'Advantage Distribution (Last {window_size} Updates)')
        plt.xlabel('Average Advantage')
        plt.ylabel('Frequency')
        plt.show()
        if save_path:
            plt.savefig(save_path)

    def plot_reward_distribution_window(self, window_size=100, save_path=None):
        rolling_rewards = self.rewards[-window_size:]
        print( self.rewards)
        plt.figure(figsize=(10, 5))
        plt.hist(rolling_rewards, bins=50, alpha=0.7)
        plt.title(f'Reward Distribution (Last {window_size} Updates)')
        plt.xlabel('Average Reward')
        plt.ylabel('Frequency')
        plt.show()
        if save_path:
            plt.savefig(save_path)

    def plot_advantage_vs_reward(self, save_path=None):
        # Compute average rewards and advantages per update
        avg_rewards = [sum(update_rewards) / len(update_rewards) for update_rewards in self.rewards]
        avg_advantages = [sum(update_advantages) / len(update_advantages) for update_advantages in self.advantages]
        
        plt.figure(figsize=(10, 5))
        plt.scatter(avg_rewards, avg_advantages, alpha=0.5, color='orange')
        plt.xlabel('Average Reward per Update')
        plt.ylabel('Average Advantage per Update')
        plt.title('Average Advantage vs. Average Reward')
        plt.grid(True)
        plt.show()
        
        if save_path:
            plt.savefig(save_path)



