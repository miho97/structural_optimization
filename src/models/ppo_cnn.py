"""Fully convolutional PPO for grid-based structural optimization."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Independent, Normal


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RolloutBuffer:
    """Fixed-horizon rollout storage kept on CPU."""

    def __init__(
        self,
        max_size,
        num_envs,
        channels,
        height,
        width,
        action_dim,
        continuous_actions=False,
        buffer_device="cpu",
    ):
        if max_size < 1 or num_envs < 1:
            raise ValueError("max_size and num_envs must be positive")
        self.num_envs = int(num_envs)
        self.max_size = int(max_size)
        self.ptr = 0
        self.buffer_device = torch.device(buffer_device)
        self.continuous_actions = bool(continuous_actions)

        self.states = torch.empty(
            (max_size, num_envs, channels, height, width),
            dtype=torch.float32,
            device=self.buffer_device,
        )
        if self.continuous_actions:
            self.actions = torch.empty(
                (max_size, num_envs, action_dim),
                dtype=torch.float32,
                device=self.buffer_device,
            )
        else:
            self.actions = torch.empty(
                (max_size, num_envs), dtype=torch.long, device=self.buffer_device
            )
        self.logprobs = torch.empty((max_size, num_envs), device=self.buffer_device)
        self.rewards = torch.empty((max_size, num_envs), device=self.buffer_device)
        self.state_values = torch.empty((max_size, num_envs), device=self.buffer_device)
        self.is_terminals = torch.empty(
            (max_size, num_envs), dtype=torch.bool, device=self.buffer_device
        )

    @property
    def is_full(self):
        return self.ptr == self.max_size

    def store(self, states, actions, logprobs, rewards, state_values, is_terminals):
        if self.is_full:
            raise IndexError(
                f"rollout buffer is full ({self.max_size} steps); update PPO before storing more"
            )
        values = (states, actions, logprobs, rewards, state_values, is_terminals)
        if any(not isinstance(value, torch.Tensor) for value in values):
            raise TypeError("all rollout values must be torch tensors")
        self.states[self.ptr].copy_(states.detach().to(self.buffer_device))
        self.actions[self.ptr].copy_(actions.detach().to(self.buffer_device))
        self.logprobs[self.ptr].copy_(logprobs.detach().to(self.buffer_device))
        self.rewards[self.ptr].copy_(rewards.detach().to(self.buffer_device))
        self.state_values[self.ptr].copy_(state_values.detach().to(self.buffer_device))
        self.is_terminals[self.ptr].copy_(is_terminals.detach().to(self.buffer_device))
        self.ptr += 1

    def clear(self):
        self.ptr = 0


class DilatedSpatialContext(nn.Module):
    """Expand the actor receptive field without resolution-dependent weights."""

    def __init__(self, channels, dilations=(2, 4, 8, 16)):
        super().__init__()
        self.dilations = tuple(dilations)
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation,
                ),
                nn.GroupNorm(16, channels),
                nn.ReLU(),
            )
            for dilation in self.dilations
        )

    def forward(self, features):
        for block in self.blocks:
            # Residual propagation retains precise local information while
            # each dilation communicates over an increasingly large radius.
            features = features + block(features)
        return features


class _SpatialActorCritic(nn.Module):
    """Distribution/value behavior shared by both convolutional encoders."""

    feature_channels: int

    def __init__(
        self,
        num_actions,
        observation_height,
        observation_width,
        has_continuous_action_space,
        action_std_init,
        target_density=0.4,
    ):
        super().__init__()
        self.num_actions = int(num_actions)
        self.observation_height = int(observation_height)
        self.observation_width = int(observation_width)
        self.grid_height = self.observation_height - 1
        self.grid_width = self.observation_width - 1
        self.has_continuous_action_space = bool(has_continuous_action_space)
        self.target_density = float(target_density)
        expected_actions = 2 * self.grid_height * self.grid_width
        if self.grid_height < 1 or self.grid_width < 1:
            raise ValueError("observations must include at least a 1x1 element grid plus nodes")
        if self.num_actions != expected_actions:
            raise ValueError(
                f"expected {expected_actions} actions for a {self.grid_height}x{self.grid_width} "
                f"grid, got {self.num_actions}"
            )
        if action_std_init <= 0:
            raise ValueError("action_std_init must be positive")
        if not 0.001 < self.target_density < 1.0:
            raise ValueError("target_density must lie strictly between 0.001 and 1")
        self.register_buffer("action_std", torch.tensor(float(action_std_init)))

    def _finish_heads(self):
        self.actor_head = nn.Conv2d(self.feature_channels, 2, kernel_size=1)
        self.critic_pool = nn.AdaptiveAvgPool2d(1)
        self.critic_head = nn.Sequential(
            nn.Linear(self.feature_channels, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )
        self.apply(self._initialize_layer)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.zeros_(self.actor_head.bias)
        nn.init.orthogonal_(self.critic_head[-1].weight, gain=1.0)
        nn.init.zeros_(self.critic_head[-1].bias)

    @staticmethod
    def _initialize_layer(layer):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(layer.weight, gain=math.sqrt(2))
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def _validate_states(self, states):
        if states.ndim != 4:
            raise ValueError("states must have shape (batch, channels, height, width)")
        if tuple(states.shape[-2:]) != (
            self.observation_height,
            self.observation_width,
        ):
            raise ValueError(
                f"expected spatial shape {(self.observation_height, self.observation_width)}, "
                f"got {tuple(states.shape[-2:])}"
            )

    def forward_features(self, states):
        raise NotImplementedError

    def action_output(self, features):
        # The observation has one padded row and column for nodal channels.
        # Cropping guarantees that padded locations can never become actions.
        output = self.actor_head(features)[
            :, :, : self.grid_height, : self.grid_width
        ].flatten(start_dim=1)
        if output.shape[1] != self.num_actions:
            raise RuntimeError("actor output/action-space mismatch")
        return torch.tanh(output) if self.has_continuous_action_space else output

    def value_from_features(self, features):
        pooled = self.critic_pool(features).flatten(start_dim=1)
        return self.critic_head(pooled).squeeze(-1)

    def valid_action_mask(self, states):
        """Mask both assignments for cells already decided void or solid."""
        densities = states[
            :, 0, : self.grid_height, : self.grid_width
        ].flatten(start_dim=1)
        tolerance = 1e-6
        undecided = (densities > 0.001 + tolerance) & (
            densities < 1.0 - tolerance
        )
        undecided_count = undecided.sum(dim=1)
        total_cells = self.grid_height * self.grid_width
        continuous_budget = (
            self.target_density * total_cells - 0.001 * total_cells
        ) / (1.0 - 0.001)
        solid_budget = int(round(continuous_budget))
        solid_budget = min(max(solid_budget, 0), total_cells)
        solid_count = (densities >= 1.0 - tolerance).sum(dim=1)

        allow_solid = (solid_count < solid_budget).unsqueeze(1)
        allow_void = (
            solid_count + undecided_count - 1 >= solid_budget
        ).unsqueeze(1)
        can_assign_void = undecided & allow_void
        can_assign_solid = undecided & allow_solid
        action_mask = torch.cat((can_assign_void, can_assign_solid), dim=1)
        if action_mask.shape[1] != self.num_actions:
            raise RuntimeError("density-derived action-mask size mismatch")
        if not bool(action_mask.any(dim=1).all()):
            raise RuntimeError("the policy was asked to act on a completed design")
        return action_mask

    def distribution(self, actor_output, states=None):
        if self.has_continuous_action_space:
            std = self.action_std.expand_as(actor_output)
            return Independent(Normal(actor_output, std), 1)
        if states is None:
            raise ValueError("states are required to mask invalid discrete actions")
        action_mask = self.valid_action_mask(states)
        masked_logits = actor_output.masked_fill(
            ~action_mask, torch.finfo(actor_output.dtype).min
        )
        return Categorical(logits=masked_logits)

    def act(self, states, deterministic=False):
        self._validate_states(states)
        features = self.forward_features(states)
        actor_output = self.action_output(features)
        distribution = self.distribution(actor_output, states)
        if deterministic:
            if self.has_continuous_action_space:
                actions = actor_output
            else:
                actions = distribution.probs.argmax(dim=-1)
        else:
            actions = distribution.sample()
        return actions, distribution.log_prob(actions), self.value_from_features(features)

    def evaluate(self, states, actions):
        self._validate_states(states)
        features = self.forward_features(states)
        distribution = self.distribution(self.action_output(features), states)
        return (
            distribution.log_prob(actions),
            self.value_from_features(features),
            distribution.entropy(),
        )

    def value(self, states):
        self._validate_states(states)
        return self.value_from_features(self.forward_features(states))

    def set_action_std(self, new_action_std):
        if new_action_std <= 0:
            raise ValueError("action standard deviation must be positive")
        self.action_std.fill_(float(new_action_std))


class CNNActorCritic(_SpatialActorCritic):
    """Single-path, fully convolutional actor with a pooled critic."""

    def __init__(
        self,
        num_channels,
        num_actions,
        width,
        height,
        has_continuous_action_space,
        action_std_init,
        target_density=0.4,
        dropout_rate=0.0,
    ):
        del dropout_rate
        super().__init__(
            num_actions,
            observation_height=width,
            observation_width=height,
            has_continuous_action_space=has_continuous_action_space,
            action_std_init=action_std_init,
            target_density=target_density,
        )
        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(16, 128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.GroupNorm(16, 128),
            nn.ReLU(),
        )
        self.feature_channels = 128
        self.spatial_context = DilatedSpatialContext(self.feature_channels)
        self.receptive_field = 7 + 2 * sum(self.spatial_context.dilations)
        self._finish_heads()

    def forward_features(self, states):
        return self.spatial_context(self.encoder(states))


class TwoPathActorCritic(_SpatialActorCritic):
    """Separate density and boundary-condition encoders with spatial fusion."""

    def __init__(
        self,
        num_actions,
        width,
        height,
        has_continuous_action_space,
        action_std_init,
        target_density=0.4,
        dropout_rate=0.0,
    ):
        del dropout_rate
        super().__init__(
            num_actions,
            observation_height=width,
            observation_width=height,
            has_continuous_action_space=has_continuous_action_space,
            action_std_init=action_std_init,
            target_density=target_density,
        )
        self.density_encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(16, 128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.GroupNorm(16, 128),
            nn.ReLU(),
        )
        self.bc_encoder = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(16, 128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.GroupNorm(16, 128),
            nn.ReLU(),
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.GroupNorm(16, 256),
            nn.ReLU(),
        )
        self.feature_channels = 256
        self.spatial_context = DilatedSpatialContext(self.feature_channels)
        self.receptive_field = 9 + 2 * sum(self.spatial_context.dilations)
        self._finish_heads()

    def forward_features(self, states):
        density_features = self.density_encoder(states[:, 0:1])
        boundary_features = self.bc_encoder(states[:, 1:5])
        fused = self.fusion_conv(
            torch.cat((density_features, boundary_features), dim=1)
        )
        return self.spatial_context(fused)


class PPO:
    def __init__(
        self,
        state_channels,
        width,
        height,
        action_dim,
        lr_actor,
        lr_critic,
        gamma,
        K_epochs,
        eps_clip,
        has_continuous_action_space,
        action_std_init=0.6,
        gae_lambda=0.95,
        num_envs=1,
        device=device,
        use_two_path=False,
        rollout_steps=128,
        minibatch_size=1024,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        target_density=0.4,
    ):
        self.device = torch.device(device)
        self.has_continuous_action_space = bool(has_continuous_action_space)
        self.num_envs = int(num_envs)
        self.use_two_path = bool(use_two_path)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.eps_clip = float(eps_clip)
        self.K_epochs = int(K_epochs)
        self.state_channels = int(state_channels)
        # These legacy argument names are observation height and width because
        # callers pass state.shape[1] and state.shape[2].
        self.height = int(width)
        self.width = int(height)
        self.action_dim = int(action_dim)
        self.rollout_steps = int(rollout_steps)
        requested_minibatch_size = int(minibatch_size)
        # Bound activation memory by spatial pixels. This keeps the convenient
        # 1024-sample batches on 6x6 while automatically reducing them on
        # large grids (about 100 samples on 50x50).
        spatial_batch_limit = max(1, 65_536 // (self.height * self.width))
        self.minibatch_size = min(requested_minibatch_size, spatial_batch_limit)
        self.value_coef = float(value_coef)
        self.entropy_coef = float(entropy_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.target_density = float(target_density)
        if self.rollout_steps < 1 or self.minibatch_size < 1 or self.K_epochs < 1:
            raise ValueError("rollout_steps, minibatch_size, and K_epochs must be positive")
        if self.minibatch_size != requested_minibatch_size:
            print(
                f"Reducing minibatch size from {requested_minibatch_size} to "
                f"{self.minibatch_size} for {self.height}x{self.width} observations"
            )

        self.buffer = RolloutBuffer(
            max_size=self.rollout_steps,
            num_envs=self.num_envs,
            channels=self.state_channels,
            height=self.height,
            width=self.width,
            action_dim=self.action_dim,
            continuous_actions=self.has_continuous_action_space,
            buffer_device="cpu",
        )

        model_class = TwoPathActorCritic if self.use_two_path else CNNActorCritic
        model_kwargs = dict(
            num_actions=self.action_dim,
            width=self.height,
            height=self.width,
            has_continuous_action_space=self.has_continuous_action_space,
            action_std_init=action_std_init,
            target_density=self.target_density,
        )
        if model_class is CNNActorCritic:
            model_kwargs["num_channels"] = self.state_channels
        architecture_name = "TwoPathActorCritic" if self.use_two_path else "CNNActorCritic"
        print(f"Using fully convolutional {architecture_name}")
        self.policy = model_class(**model_kwargs).to(self.device)
        self.policy_old = model_class(**model_kwargs).to(self.device)

        if self.use_two_path:
            actor_parameters = list(self.policy.density_encoder.parameters())
            actor_parameters += list(self.policy.bc_encoder.parameters())
            actor_parameters += list(self.policy.fusion_conv.parameters())
        else:
            actor_parameters = list(self.policy.encoder.parameters())
        actor_parameters += list(self.policy.spatial_context.parameters())
        actor_parameters += list(self.policy.actor_head.parameters())
        critic_parameters = list(self.policy.critic_head.parameters())
        self.optimizer = optim.Adam(
            (
                {"params": actor_parameters, "lr": lr_actor},
                {"params": critic_parameters, "lr": lr_critic},
            )
        )
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.95)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.policy_old.eval()
        self.value_loss = nn.MSELoss()
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.device.type == "cuda"
        )

        self.actor_losses = []
        self.critic_losses = []
        self.entropies = []
        self.grad_norms = []

    def set_action_std(self, new_action_std):
        if not self.has_continuous_action_space:
            raise RuntimeError("action standard deviation only applies to continuous policies")
        self.policy.set_action_std(new_action_std)
        self.policy_old.set_action_std(new_action_std)

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        if not self.has_continuous_action_space:
            raise RuntimeError("action standard deviation only applies to continuous policies")
        new_std = max(
            self.policy.action_std.item() - action_std_decay_rate, min_action_std
        )
        self.set_action_std(new_std)
        return new_std

    def select_action(self, states, deterministic=False):
        self.policy_old.eval()
        states = states.to(self.device, dtype=torch.float32)
        with torch.no_grad():
            return self.policy_old.act(states, deterministic=deterministic)

    def estimate_values(self, states):
        self.policy_old.eval()
        states = states.to(self.device, dtype=torch.float32)
        with torch.no_grad():
            return self.policy_old.value(states)

    def compute_gae(
        self,
        rewards,
        state_values,
        is_terminals,
        last_state_values=None,
        gamma=None,
        gae_lambda=None,
    ):
        """Compute GAE and bootstrap non-terminal rollout boundaries."""
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device="cpu")
        state_values = torch.as_tensor(state_values, dtype=torch.float32, device="cpu")
        is_terminals = torch.as_tensor(is_terminals, dtype=torch.bool, device="cpu")
        if rewards.ndim != 2 or rewards.shape != state_values.shape:
            raise ValueError("rewards and state_values must have shape (steps, num_envs)")
        if is_terminals.shape != rewards.shape:
            raise ValueError("terminal flags must match rewards")
        if rewards.shape[1] != self.num_envs:
            raise ValueError("rollout num_envs does not match PPO configuration")
        if last_state_values is None:
            last_state_values = torch.zeros(self.num_envs)
        last_state_values = torch.as_tensor(
            last_state_values, dtype=torch.float32, device="cpu"
        ).reshape(self.num_envs)
        gamma = self.gamma if gamma is None else gamma
        gae_lambda = self.gae_lambda if gae_lambda is None else gae_lambda

        advantages = torch.zeros_like(rewards)
        last_advantage = torch.zeros(self.num_envs)
        for step in reversed(range(rewards.shape[0])):
            next_value = (
                last_state_values
                if step == rewards.shape[0] - 1
                else state_values[step + 1]
            )
            non_terminal = (~is_terminals[step]).float()
            delta = (
                rewards[step]
                + gamma * next_value * non_terminal
                - state_values[step]
            )
            last_advantage = (
                delta + gamma * gae_lambda * non_terminal * last_advantage
            )
            advantages[step] = last_advantage
        returns = advantages + state_values
        return returns, advantages

    def update(self, last_state_values=None):
        """Run a minibatched PPO update and clear the rollout buffer."""
        if self.buffer.ptr == 0:
            raise RuntimeError("cannot update PPO with an empty rollout")
        rollout_length = self.buffer.ptr
        rewards = self.buffer.rewards[:rollout_length]
        state_values = self.buffer.state_values[:rollout_length]
        terminals = self.buffer.is_terminals[:rollout_length]
        returns, advantages = self.compute_gae(
            rewards, state_values, terminals, last_state_values=last_state_values
        )

        states = self.buffer.states[:rollout_length].reshape(
            -1, self.state_channels, self.height, self.width
        )
        if self.has_continuous_action_space:
            actions = self.buffer.actions[:rollout_length].reshape(-1, self.action_dim)
        else:
            actions = self.buffer.actions[:rollout_length].reshape(-1)
        old_logprobs = self.buffer.logprobs[:rollout_length].reshape(-1)
        returns = returns.reshape(-1)
        advantages = advantages.reshape(-1)
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )

        num_samples = states.shape[0]
        batch_size = min(self.minibatch_size, num_samples)
        actor_epoch_losses = []
        critic_epoch_losses = []
        entropy_values = []
        gradient_norms = []
        self.policy.train()

        for _ in range(self.K_epochs):
            permutation = torch.randperm(num_samples)
            for start in range(0, num_samples, batch_size):
                indices = permutation[start : start + batch_size]
                batch_states = states[indices].to(self.device)
                batch_actions = actions[indices].to(self.device)
                batch_old_logprobs = old_logprobs[indices].to(self.device)
                batch_returns = returns[indices].to(self.device)
                batch_advantages = advantages[indices].to(self.device)

                with torch.autocast(
                    device_type=self.device.type, enabled=self.device.type == "cuda"
                ):
                    new_logprobs, new_values, entropy = self.policy.evaluate(
                        batch_states, batch_actions
                    )
                    ratios = torch.exp(new_logprobs - batch_old_logprobs)
                    surrogate_1 = ratios * batch_advantages
                    surrogate_2 = torch.clamp(
                        ratios, 1 - self.eps_clip, 1 + self.eps_clip
                    ) * batch_advantages
                    actor_loss = -torch.min(surrogate_1, surrogate_2).mean()
                    critic_loss = self.value_loss(new_values, batch_returns)
                    entropy_mean = entropy.mean()
                    loss = (
                        actor_loss
                        + self.value_coef * critic_loss
                        - self.entropy_coef * entropy_mean
                    )

                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()

                actor_epoch_losses.append(float(actor_loss.detach().cpu()))
                critic_epoch_losses.append(float(critic_loss.detach().cpu()))
                entropy_values.append(float(entropy_mean.detach().cpu()))
                gradient_norms.append(float(gradient_norm.detach().cpu()))

        metrics = {
            "actor_loss": sum(actor_epoch_losses) / len(actor_epoch_losses),
            "critic_loss": sum(critic_epoch_losses) / len(critic_epoch_losses),
            "entropy": sum(entropy_values) / len(entropy_values),
            "grad_norm": sum(gradient_norms) / len(gradient_norms),
            "rollout_steps": rollout_length,
            "samples": num_samples,
        }
        self.actor_losses.append(metrics["actor_loss"])
        self.critic_losses.append(metrics["critic_loss"])
        self.entropies.append(metrics["entropy"])
        self.grad_norms.append(metrics["grad_norm"])

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.policy_old.eval()
        self.buffer.clear()
        self.scheduler.step()
        return metrics

    def save(self, checkpoint_path):
        torch.save(
            {
                "policy_state_dict": self.policy_old.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "architecture": "two_path" if self.use_two_path else "single_path",
                "observation_shape": (self.state_channels, self.height, self.width),
                "action_dim": self.action_dim,
                "target_density": self.target_density,
            },
            checkpoint_path,
        )

    def load(self, checkpoint_path):
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=True
        )
        saved_target_density = checkpoint.get("target_density")
        if saved_target_density is not None and not math.isclose(
            float(saved_target_density), self.target_density, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                "checkpoint target density does not match the current PPO agent"
            )
        self.policy_old.load_state_dict(checkpoint["policy_state_dict"])
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.policy_old.eval()

    def plot_metrics(self, save_path=None):
        updates = range(1, len(self.actor_losses) + 1)
        figure, axes = plt.subplots(2, 2, figsize=(16, 10))
        axes[0, 0].plot(updates, self.actor_losses)
        axes[0, 0].set_title("Actor loss")
        axes[0, 1].plot(updates, self.critic_losses)
        axes[0, 1].set_title("Critic loss")
        axes[1, 0].plot(updates, self.entropies)
        axes[1, 0].set_title("Policy entropy")
        axes[1, 1].plot(updates, self.grad_norms)
        axes[1, 1].set_title("Gradient norm")
        for axis in axes.flat:
            axis.set_xlabel("PPO update")
            axis.grid(alpha=0.25)
        figure.tight_layout()
        if save_path:
            figure.savefig(save_path)
        plt.show()
