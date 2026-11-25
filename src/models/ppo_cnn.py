"""Fully convolutional PPO for grid-based structural optimization.

The policy reads the observation layout defined in ``utils.layout`` and emits
two logits per physical cell (void, solid) plus one stop logit. Invalid
actions are masked from the observation itself, so the mask always matches
the environment and the network is independent of grid size and target
volume.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from utils import layout


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
        action_dim=None,
        continuous_actions=False,
        buffer_device="cpu",
    ):
        del action_dim
        if continuous_actions:
            raise ValueError("the grid environment has discrete actions only")
        if max_size < 1 or num_envs < 1:
            raise ValueError("max_size and num_envs must be positive")
        self.num_envs = int(num_envs)
        self.max_size = int(max_size)
        self.ptr = 0
        self.buffer_device = torch.device(buffer_device)

        shape = (self.max_size, self.num_envs)
        self.states = torch.empty(
            (*shape, channels, height, width), dtype=torch.float32, device=self.buffer_device
        )
        self.actions = torch.empty(shape, dtype=torch.long, device=self.buffer_device)
        self.logprobs = torch.empty(shape, device=self.buffer_device)
        self.rewards = torch.empty(shape, device=self.buffer_device)
        self.state_values = torch.empty(shape, device=self.buffer_device)
        self.is_terminals = torch.empty(shape, dtype=torch.bool, device=self.buffer_device)
        # False marks a transition that must not be trained on, such as the
        # reset step a vector environment inserts after an episode ends.
        self.valid = torch.ones(shape, dtype=torch.bool, device=self.buffer_device)

    @property
    def is_full(self):
        return self.ptr == self.max_size

    def store(self, states, actions, logprobs, rewards, state_values, is_terminals, valid=None):
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
        if valid is None:
            self.valid[self.ptr].fill_(True)
        else:
            self.valid[self.ptr].copy_(
                torch.as_tensor(valid, dtype=torch.bool).to(self.buffer_device)
            )
        self.ptr += 1

    def clear(self):
        self.ptr = 0


def _groups(channels):
    """GroupNorm group count: 8 below 128 channels, 16 from there on."""
    groups = 8 if channels < 128 else 16
    while channels % groups:
        groups //= 2
    return max(groups, 1)


def _conv_block(in_channels, out_channels):
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.GroupNorm(_groups(out_channels), out_channels),
        nn.ReLU(),
    ]


class DilatedSpatialContext(nn.Module):
    """Expand the receptive field without resolution-dependent weights."""

    def __init__(self, channels, dilations=(2, 4, 8, 16)):
        super().__init__()
        self.dilations = tuple(dilations)
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
                nn.GroupNorm(_groups(channels), channels),
                nn.ReLU(),
            )
            for dilation in self.dilations
        )

    def forward(self, features):
        for block in self.blocks:
            # Residual propagation keeps precise local information while each
            # dilation communicates over an increasingly large radius.
            features = features + block(features)
        return features


class _SpatialActorCritic(nn.Module):
    """Distribution and value behavior shared by both encoders."""

    feature_channels: int

    def __init__(self, num_actions, observation_height, observation_width, num_channels):
        super().__init__()
        self.num_actions = int(num_actions)
        self.observation_height = int(observation_height)
        self.observation_width = int(observation_width)
        self.grid_height = self.observation_height - 1
        self.grid_width = self.observation_width - 1
        self.num_channels = int(num_channels)
        if self.grid_height < 1 or self.grid_width < 1:
            raise ValueError("observations must include at least a 1x1 element grid plus nodes")
        self.num_cells = self.grid_height * self.grid_width
        expected_actions = 2 * self.num_cells + 1
        if self.num_actions != expected_actions:
            raise ValueError(
                f"expected {expected_actions} actions (void and solid per cell, plus stop) "
                f"for a {self.grid_height}x{self.grid_width} grid, got {self.num_actions}"
            )
        if self.num_channels != layout.NUM_CHANNELS:
            raise ValueError(
                f"observations must have {layout.NUM_CHANNELS} channels "
                f"(layout version {layout.LAYOUT_VERSION}), got {self.num_channels}"
            )

    def _finish_heads(self):
        pooled = 2 * self.feature_channels  # average and max pooling
        self.actor_head = nn.Conv2d(self.feature_channels, 2, kernel_size=1)
        self.stop_head = nn.Linear(pooled, 1)
        self.critic_head = nn.Sequential(nn.Linear(pooled, 256), nn.Tanh(), nn.Linear(256, 1))
        self.apply(self._initialize_layer)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.zeros_(self.actor_head.bias)
        nn.init.orthogonal_(self.stop_head.weight, gain=0.01)
        nn.init.zeros_(self.stop_head.bias)
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
        if states.shape[1] != self.num_channels:
            raise ValueError(
                f"expected {self.num_channels} observation channels, got {states.shape[1]}"
            )
        if tuple(states.shape[-2:]) != (self.observation_height, self.observation_width):
            raise ValueError(
                f"expected spatial shape {(self.observation_height, self.observation_width)}, "
                f"got {tuple(states.shape[-2:])}"
            )

    def forward_features(self, states):
        raise NotImplementedError

    def _pooled(self, features):
        return torch.cat((features.mean(dim=(2, 3)), features.amax(dim=(2, 3))), dim=1)

    def action_output(self, features):
        """Logits for ``2N`` cell actions followed by the stop action, in float32."""
        # The observation has one padded row and column for nodal channels.
        # Cropping guarantees that padded locations can never become actions.
        cells = self.actor_head(features)[:, :, : self.grid_height, : self.grid_width]
        stop = self.stop_head(self._pooled(features))
        return torch.cat((cells.flatten(start_dim=1), stop), dim=1).float()

    def value_from_features(self, features):
        return self.critic_head(self._pooled(features)).squeeze(-1).float()

    def _cells(self, states, channel):
        return states[:, channel, : self.grid_height, : self.grid_width].flatten(start_dim=1)

    def valid_action_mask(self, states):
        """Rebuild the environment's action mask from the observation."""
        undecided = self._cells(states, layout.UNDECIDED) > 0.5
        solid = self._cells(states, layout.SOLID) > 0.5
        void = self._cells(states, layout.VOID) > 0.5
        pending = self._cells(states, layout.PENDING) > 0.5
        budget_left = torch.round(states[:, layout.BUDGET_LEFT, 0, 0] * self.num_cells)
        swaps_left = states[:, layout.SWAPS_LEFT, 0, 0] > 1e-6

        undecided_count = undecided.sum(dim=1)
        constructing = undecided_count > 0
        allow_solid = constructing & (budget_left > 0)
        allow_void = constructing & (undecided_count - 1 >= budget_left)
        can_void = undecided & allow_void.unsqueeze(1)
        can_solid = undecided & allow_solid.unsqueeze(1)

        has_pending = pending.any(dim=1)
        idle = ~constructing & ~has_pending
        swap_possible = idle & swaps_left & solid.any(dim=1) & void.any(dim=1)
        # A finished design starts a swap by removing one of its solid cells.
        can_void = can_void | (solid & swap_possible.unsqueeze(1))
        # The removed cell cannot be refilled, so every swap changes the design.
        can_solid = can_solid | (void & ~pending & (~constructing & has_pending).unsqueeze(1))
        # Stopping is allowed whenever a finished design has no pending swap.
        action_mask = torch.cat((can_void, can_solid, idle.unsqueeze(1)), dim=1)
        if not bool(action_mask.any(dim=1).all()):
            raise RuntimeError("an observation has no valid action")
        return action_mask

    def distribution(self, logits, states):
        action_mask = self.valid_action_mask(states)
        masked_logits = logits.masked_fill(~action_mask, torch.finfo(logits.dtype).min)
        return Categorical(logits=masked_logits)

    def act(self, states, deterministic=False):
        self._validate_states(states)
        features = self.forward_features(states)
        distribution = self.distribution(self.action_output(features), states)
        if deterministic:
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


class CNNActorCritic(_SpatialActorCritic):
    """Single-path, fully convolutional actor with a pooled critic.

    ``width`` and ``height`` keep their historical meaning in this module:
    they are the observation's first and second spatial sizes, that is
    ``grid_height + 1`` and ``grid_width + 1``.
    """

    def __init__(
        self,
        num_channels,
        num_actions,
        width,
        height,
        has_continuous_action_space=False,
        action_std_init=None,
        target_density=None,
        dropout_rate=0.0,
        base_channels=64,
    ):
        del action_std_init, target_density, dropout_rate
        if has_continuous_action_space:
            raise ValueError("the grid environment has discrete actions only")
        super().__init__(num_actions, width, height, num_channels)
        hidden = 2 * base_channels
        self.encoder = nn.Sequential(
            *_conv_block(num_channels, base_channels),
            *_conv_block(base_channels, hidden),
            *_conv_block(hidden, hidden),
        )
        self.feature_channels = hidden
        self.spatial_context = DilatedSpatialContext(hidden)
        self.receptive_field = 7 + 2 * sum(self.spatial_context.dilations)
        self._finish_heads()

    def forward_features(self, states):
        return self.spatial_context(self.encoder(states))


class TwoPathActorCritic(_SpatialActorCritic):
    """Separate topology and boundary-condition encoders with spatial fusion."""

    def __init__(
        self,
        num_actions,
        width,
        height,
        has_continuous_action_space=False,
        action_std_init=None,
        target_density=None,
        dropout_rate=0.0,
        num_channels=layout.NUM_CHANNELS,
        base_channels=64,
    ):
        del action_std_init, target_density, dropout_rate
        if has_continuous_action_space:
            raise ValueError("the grid environment has discrete actions only")
        super().__init__(num_actions, width, height, num_channels)
        hidden = 2 * base_channels
        self.register_buffer(
            "topology_index", torch.tensor(layout.TOPOLOGY_CHANNELS), persistent=False
        )
        self.register_buffer(
            "boundary_index", torch.tensor(layout.BOUNDARY_CHANNELS), persistent=False
        )
        self.density_encoder = nn.Sequential(
            *_conv_block(len(layout.TOPOLOGY_CHANNELS), base_channels),
            *_conv_block(base_channels, hidden),
            *_conv_block(hidden, hidden),
        )
        self.bc_encoder = nn.Sequential(
            *_conv_block(len(layout.BOUNDARY_CHANNELS), base_channels),
            *_conv_block(base_channels, hidden),
            *_conv_block(hidden, hidden),
        )
        self.fusion_conv = nn.Sequential(*_conv_block(2 * hidden, 2 * hidden))
        self.feature_channels = 2 * hidden
        self.spatial_context = DilatedSpatialContext(self.feature_channels)
        self.receptive_field = 9 + 2 * sum(self.spatial_context.dilations)
        self._finish_heads()

    def forward_features(self, states):
        topology = self.density_encoder(states.index_select(1, self.topology_index))
        boundary = self.bc_encoder(states.index_select(1, self.boundary_index))
        return self.spatial_context(self.fusion_conv(torch.cat((topology, boundary), dim=1)))


class PPO:
    """Proximal policy optimization with GAE and minibatched updates.

    ``width`` and ``height`` are the observation's spatial sizes in the order
    callers already pass them: ``state.shape[1]`` and ``state.shape[2]``.

    Schedules are active only when ``total_updates`` is given: the learning
    rates then decay linearly to zero and the entropy coefficient moves
    linearly from ``entropy_coef`` to ``entropy_coef_final``.
    """

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
        has_continuous_action_space=False,
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
        target_density=None,
        entropy_coef_final=None,
        total_updates=None,
        target_kl=None,
        base_channels=64,
        use_amp=None,
    ):
        del action_std_init
        if has_continuous_action_space:
            raise ValueError("the grid environment has discrete actions only")
        self.device = torch.device(device)
        self.has_continuous_action_space = False
        self.num_envs = int(num_envs)
        self.use_two_path = bool(use_two_path)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.eps_clip = float(eps_clip)
        self.K_epochs = int(K_epochs)
        self.state_channels = int(state_channels)
        # Legacy argument names: these are observation height and width.
        self.height = int(width)
        self.width = int(height)
        self.action_dim = int(action_dim)
        self.rollout_steps = int(rollout_steps)
        self.base_channels = int(base_channels)
        requested_minibatch_size = int(minibatch_size)
        # Bound activation memory by spatial pixels: 1024-sample batches on
        # 6x6, automatically smaller on large grids (about 25 on 50x50).
        spatial_batch_limit = max(1, 65_536 // (self.height * self.width))
        self.minibatch_size = min(requested_minibatch_size, spatial_batch_limit)
        self.value_coef = float(value_coef)
        self.entropy_coef_start = float(entropy_coef)
        self.entropy_coef_final = (
            self.entropy_coef_start if entropy_coef_final is None else float(entropy_coef_final)
        )
        self.total_updates = None if total_updates is None else max(1, int(total_updates))
        self.target_kl = None if target_kl is None else float(target_kl)
        self.max_grad_norm = float(max_grad_norm)
        self.target_density = None if target_density is None else float(target_density)
        self.updates_done = 0
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
            buffer_device="cpu",
        )

        model_class = TwoPathActorCritic if self.use_two_path else CNNActorCritic
        model_kwargs = dict(
            num_actions=self.action_dim,
            width=self.height,
            height=self.width,
            num_channels=self.state_channels,
            base_channels=self.base_channels,
        )
        print(f"Using fully convolutional {model_class.__name__}")
        self.policy = model_class(**model_kwargs).to(self.device)
        self.policy_old = model_class(**model_kwargs).to(self.device)

        critic_parameters = list(self.policy.critic_head.parameters())
        critic_ids = {id(parameter) for parameter in critic_parameters}
        actor_parameters = [
            parameter for parameter in self.policy.parameters() if id(parameter) not in critic_ids
        ]
        self.optimizer = optim.Adam(
            (
                {"params": actor_parameters, "lr": lr_actor},
                {"params": critic_parameters, "lr": lr_critic},
            )
        )
        if self.total_updates is None:
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.95)
        else:
            total = self.total_updates
            self.scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer, lambda update: max(0.0, 1.0 - update / total)
            )
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.policy_old.eval()
        self.value_loss = nn.MSELoss()
        self.use_amp = (self.device.type == "cuda") if use_amp is None else bool(use_amp)
        self.use_amp = self.use_amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.actor_losses = []
        self.critic_losses = []
        self.entropies = []
        self.grad_norms = []

    @property
    def entropy_coef(self):
        """Current entropy coefficient under the linear schedule."""
        if self.total_updates is None:
            return self.entropy_coef_start
        progress = min(1.0, self.updates_done / self.total_updates)
        return self.entropy_coef_start + progress * (
            self.entropy_coef_final - self.entropy_coef_start
        )

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
                last_state_values if step == rewards.shape[0] - 1 else state_values[step + 1]
            )
            non_terminal = (~is_terminals[step]).float()
            delta = rewards[step] + gamma * next_value * non_terminal - state_values[step]
            last_advantage = delta + gamma * gae_lambda * non_terminal * last_advantage
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

        # Invalid transitions (vector-environment reset steps) are dropped
        # here. They never feed an earlier advantage, because the step before
        # each of them is terminal.
        keep = self.buffer.valid[:rollout_length].reshape(-1).nonzero(as_tuple=True)[0]
        if keep.numel() == 0:
            raise RuntimeError("the rollout contains no valid transitions")
        states = self.buffer.states[:rollout_length].reshape(
            -1, self.state_channels, self.height, self.width
        )[keep]
        actions = self.buffer.actions[:rollout_length].reshape(-1)[keep]
        old_logprobs = self.buffer.logprobs[:rollout_length].reshape(-1)[keep]
        old_values = state_values.reshape(-1)[keep]
        returns = returns.reshape(-1)[keep]
        advantages = advantages.reshape(-1)[keep]

        return_variance = float(returns.var(unbiased=False))
        explained_variance = (
            float("nan")
            if return_variance < 1e-12
            else 1.0 - float((returns - old_values).var(unbiased=False)) / return_variance
        )
        value_offset = float((old_values - returns).mean())
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        num_samples = states.shape[0]
        batch_size = min(self.minibatch_size, num_samples)
        entropy_coef = self.entropy_coef
        actor_losses, critic_losses, entropies, gradient_norms = [], [], [], []
        kls, clip_fractions = [], []
        stopped_early = False
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

                with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                    # Convolutions may run in half precision; the logits and
                    # values come back in float32, so the distribution, the
                    # ratio and the losses are all computed in float32.
                    new_logprobs, new_values, entropy = self.policy.evaluate(
                        batch_states, batch_actions
                    )
                log_ratio = new_logprobs - batch_old_logprobs
                ratios = torch.exp(log_ratio)
                with torch.no_grad():
                    approx_kl = float(((ratios - 1.0) - log_ratio).mean())
                    clip_fraction = float(((ratios - 1.0).abs() > self.eps_clip).float().mean())
                if self.target_kl is not None and approx_kl > 1.5 * self.target_kl:
                    stopped_early = True
                    break
                surrogate_1 = ratios * batch_advantages
                surrogate_2 = (
                    torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                )
                actor_loss = -torch.min(surrogate_1, surrogate_2).mean()
                critic_loss = self.value_loss(new_values, batch_returns)
                entropy_mean = entropy.mean()
                loss = actor_loss + self.value_coef * critic_loss - entropy_coef * entropy_mean

                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()

                actor_losses.append(float(actor_loss.detach()))
                critic_losses.append(float(critic_loss.detach()))
                entropies.append(float(entropy_mean.detach()))
                gradient_norms.append(float(gradient_norm))
                kls.append(approx_kl)
                clip_fractions.append(clip_fraction)
            if stopped_early:
                break

        def mean(values):
            return sum(values) / len(values) if values else float("nan")

        metrics = {
            "actor_loss": mean(actor_losses),
            "critic_loss": mean(critic_losses),
            "entropy": mean(entropies),
            "grad_norm": mean(gradient_norms),
            "approx_kl": mean(kls),
            "clip_fraction": mean(clip_fractions),
            "explained_variance": explained_variance,
            "value_offset": value_offset,
            "entropy_coef": entropy_coef,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "gradient_steps": len(actor_losses),
            "stopped_early": stopped_early,
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
        self.updates_done += 1
        return metrics

    def save(self, checkpoint_path):
        torch.save(
            {
                "policy_state_dict": self.policy_old.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "architecture": "two_path" if self.use_two_path else "single_path",
                "base_channels": self.base_channels,
                "layout_version": layout.LAYOUT_VERSION,
                "observation_shape": (self.state_channels, self.height, self.width),
                "action_dim": self.action_dim,
                "updates_done": self.updates_done,
                "target_density": self.target_density,
            },
            checkpoint_path,
        )

    def load(self, checkpoint_path, weights_only=False):
        """Load a checkpoint. ``weights_only=True`` skips optimizer state.

        The network is fully convolutional, so weights trained on one grid
        size load into an agent built for another. Use ``weights_only=True``
        in that case and for evaluation.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        if checkpoint.get("layout_version") != layout.LAYOUT_VERSION:
            raise ValueError(
                "this checkpoint was trained with an older observation and action layout "
                f"and cannot be loaded (expected layout version {layout.LAYOUT_VERSION}). "
                "Retrain with the current environment."
            )
        architecture = "two_path" if self.use_two_path else "single_path"
        if checkpoint.get("architecture") != architecture:
            raise ValueError(
                f"checkpoint architecture is {checkpoint.get('architecture')!r}, "
                f"but this agent is {architecture!r}"
            )
        if checkpoint.get("base_channels", 64) != self.base_channels:
            raise ValueError(
                f"checkpoint has base_channels={checkpoint.get('base_channels', 64)}, "
                f"but this agent has base_channels={self.base_channels}"
            )
        self.policy_old.load_state_dict(checkpoint["policy_state_dict"])
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        if not weights_only:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.updates_done = int(checkpoint.get("updates_done", 0))
        self.policy_old.eval()

    def plot_metrics(self, save_path=None):
        import matplotlib.pyplot as plt

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
