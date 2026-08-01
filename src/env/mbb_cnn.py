"""Gymnasium environment for density-based structural optimization.

Coordinate convention
---------------------
``width`` is the number of finite elements along X (columns) and ``height`` is
the number along Y (rows). Density fields and CNN images therefore use
``(height, width)`` / ``(channels, height, width)`` ordering. The FEM helper
stores nodal data in ``(x, y, dof)`` ordering, so nodal channels are transposed
exactly once when the observation is constructed.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from utils import fem


def generate_random_beam(width, height, density=0.4, rng=None):
    """Generate a stable random support/load configuration.

    The returned nodal arrays intentionally follow the FEM convention
    ``(width + 1, height + 1, 2)``. Pass a NumPy ``Generator`` to make the
    result reproducible without mutating global random state.
    """
    if width < 1 or height < 1:
        raise ValueError("width and height must both be positive")

    rng = np.random.default_rng() if rng is None else rng
    normals = np.zeros((width + 1, height + 1, 2), dtype=np.float64)
    forces = np.zeros((width + 1, height + 1, 2), dtype=np.float64)

    template = rng.choice(("cantilever", "simply_supported", "corner_supported"))

    if template == "cantilever":
        fixed_edge = rng.choice(("left", "right"))
        fixed_x = 0 if fixed_edge == "left" else width
        force_x = width if fixed_edge == "left" else 0
        normals[fixed_x, :, 0] = 1
        normals[fixed_x, 0, 1] = 1
        normals[fixed_x, height, 1] = 1
        force_y = int(rng.integers(0, height + 1))
        forces[force_x, force_y, 1] = -1

    elif template == "simply_supported":
        # Pin/roller supports at one horizontal edge and a downward load at
        # the opposite edge.
        normals[0, height, 0:2] = 1
        normals[width, height, 1] = 1
        force_x = int(rng.integers(1, width)) if width > 1 else 0
        forces[force_x, 0, 1] = -1

    else:  # corner_supported
        fixed_x = 0 if rng.random() < 0.5 else width
        opposite_x = width - fixed_x
        normals[fixed_x, :, 0] = 1
        corner_y = int(rng.choice((0, height)))
        normals[opposite_x, corner_y, 1] = 1
        force_y = int(rng.integers(0, height + 1))
        forces[fixed_x, force_y, 1] = -1

    return normals, forces, density


class BeamOptimizationEnv(gym.Env):
    """Sequential density-editing environment backed by sparse FEM."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        width=4,
        height=4,
        density=0.4,
        step_size=0.5,
        optimal_density=None,
        reward_weights=None,
        beam_type=1,
        custom_normals=None,
        custom_forces=None,
        use_random_beam=False,
        max_steps=None,
    ):
        super().__init__()
        if width < 1 or height < 1:
            raise ValueError("width and height must both be positive")
        if not 0 < density <= 1:
            raise ValueError("density must be in (0, 1]")
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        if (custom_normals is None) != (custom_forces is None):
            raise ValueError("custom_normals and custom_forces must be supplied together")

        self.device = torch.device("cpu")
        self.width = int(width)
        self.height = int(height)
        # ``optimal_density`` is kept for compatibility. There is one
        # authoritative volume target internally.
        self.target_density = float(density if optimal_density is None else optimal_density)
        if not 0 < self.target_density <= 1:
            raise ValueError("optimal_density must be in (0, 1]")
        self.density = self.target_density
        self.optimal_density = self.target_density
        self.step_size_initial = float(step_size)
        self.step_size_final = min(0.05, self.step_size_initial)
        self.max_steps = int(max_steps or self.width * self.height * 3)
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")

        self.action_space = gym.spaces.Discrete(2 * self.width * self.height)
        observation_shape = (5, self.height + 1, self.width + 1)
        obs_low = np.zeros(observation_shape, dtype=np.float32)
        obs_high = np.ones(observation_shape, dtype=np.float32)
        obs_low[3:5] = -1.0
        self.observation_space = gym.spaces.Box(obs_low, obs_high, dtype=np.float32)
        self.state_dim = int(np.prod(observation_shape))

        self.beam_type = int(beam_type)
        self.use_random_beam = bool(use_random_beam)
        self.custom_normals = custom_normals
        self.custom_forces = custom_forces
        self.beam_functions = {
            1: fem.mbb_beam_1,
            2: fem.mbb_beam_2,
            3: fem.mbb_beam_3,
            4: fem.mbb_beam_4,
            5: fem.mbb_beam_5,
            6: fem.mbb_beam_6,
            7: fem.mbb_beam_7,
            8: fem.mbb_beam_8,
            9: fem.mbb_beam_9,
        }
        if not self.use_random_beam and custom_normals is None and self.beam_type not in self.beam_functions:
            raise ValueError(f"beam_type must be one of {sorted(self.beam_functions)}")

        weights = reward_weights or {}
        self.w_compliance = float(weights.get("w_compliance", 1.0))
        self.w_density_low = float(weights.get("w_density_low", 1.0))
        # The violation is a relative fraction rather than a cell count. A
        # weight of five makes feasibility dominant without changing meaning
        # as grid resolution changes.
        self.w_total_mass = float(weights.get("w_total_mass", 5.0))
        self.w_grey = float(weights.get("w_grey", 0.5))
        self.improvement_scale = float(weights.get("w_improvement", 0.5))

        if custom_normals is not None:
            normals, forces = custom_normals, custom_forces
        elif self.use_random_beam:
            normals, forces, _ = generate_random_beam(
                self.width, self.height, self.target_density, self.np_random
            )
        else:
            normals, forces, _ = self.beam_functions[self.beam_type](
                self.width, self.height, self.target_density
            )
        self._set_problem(normals, forces)
        self.reset(options={"regenerate_beam": False})

    def _set_problem(self, normals, forces):
        expected_shape = (self.width + 1, self.height + 1, 2)
        normals = np.asarray(normals, dtype=np.float32)
        forces = np.asarray(forces, dtype=np.float32)
        if normals.shape != expected_shape or forces.shape != expected_shape:
            raise ValueError(
                f"normals and forces must both have shape {expected_shape}; "
                f"got {normals.shape} and {forces.shape}"
            )
        if not np.isfinite(normals).all() or not np.isfinite(forces).all():
            raise ValueError("normals and forces must be finite")
        if not np.any(normals):
            raise ValueError("the structure must have at least one constrained degree of freedom")
        if not np.any(forces):
            raise ValueError("the structure must have at least one non-zero load")
        if np.any(normals < 0) or np.any(normals > 1):
            raise ValueError("normal/support indicators must lie in [0, 1]")
        if np.any(forces < -1) or np.any(forces > 1):
            raise ValueError("force values must lie in [-1, 1]")

        self.normals = torch.as_tensor(normals, dtype=torch.float32, device=self.device)
        self.forces = torch.as_tensor(forces, dtype=torch.float32, device=self.device)
        self.args = fem.get_args(normals, forces, self.target_density)
        base_density = np.full((self.height, self.width), self.target_density, dtype=np.float64)
        base_compliance, _ = fem.optim(self.args, x=base_density, verbose=False)
        self.base_compliance_value = float(base_compliance)
        if not np.isfinite(self.base_compliance_value) or self.base_compliance_value <= 0:
            raise ValueError("the supplied support/load problem has invalid base compliance")
        # Compatibility for existing consumers while reward code uses the
        # unambiguous scalar above.
        self.base_compliance = {self.beam_type: self.base_compliance_value}

    def construct_cnn_state(self, densities, normals=None, forces=None):
        """Build a ``(5, height+1, width+1)`` observation."""
        densities = torch.as_tensor(densities, dtype=torch.float32, device=self.device)
        if densities.numel() != self.width * self.height:
            raise ValueError("density tensor has the wrong number of elements")
        densities = densities.reshape(self.height, self.width)
        densities_padded = F.pad(densities, (0, 1, 0, 1), value=0.0)

        normals = self.normals if normals is None else normals
        forces = self.forces if forces is None else forces
        # FEM nodal arrays are [x, y, dof]; image channels are [y, x].
        normals_yx = normals.reshape(self.width + 1, self.height + 1, 2).permute(1, 0, 2)
        forces_yx = forces.reshape(self.width + 1, self.height + 1, 2).permute(1, 0, 2)
        return torch.stack(
            (
                densities_padded,
                normals_yx[..., 0],
                normals_yx[..., 1],
                forces_yx[..., 0],
                forces_yx[..., 1],
            ),
            dim=0,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if self.use_random_beam and options.get("regenerate_beam", True):
            normals, forces, _ = generate_random_beam(
                self.width, self.height, self.target_density, self.np_random
            )
            self._set_problem(normals, forces)

        densities = torch.full(
            (self.height, self.width),
            self.target_density,
            dtype=torch.float32,
            device=self.device,
        )
        initial_compliance, volume_fraction = fem.optim(
            self.args, x=densities.numpy(), verbose=False
        )
        self.current_step = 0
        self.current_compliance = float(initial_compliance)
        self.previous_compliance = float(initial_compliance)
        self.current_constraint = float(volume_fraction)
        self.best_compliance_ratio = self.current_compliance / self.base_compliance_value
        self.reward = 0.0
        self.state = self.construct_cnn_state(densities)
        info = {
            "compliance": self.current_compliance,
            "base_compliance": self.base_compliance_value,
            "compliance_ratio": self.best_compliance_ratio,
            "volume_fraction": self.current_constraint,
            "target_volume_fraction": self.target_density,
        }
        return self.state.numpy().copy(), info

    def density_field(self):
        """Return a copy of the physical ``(height, width)`` density field."""
        return self.state[0, : self.height, : self.width].numpy().copy()

    def check_connectivity(self, densities=None):
        """Return the fraction of solid cells outside the largest component.

        This is a diagnostic only. Compliance already captures whether loads
        can reach supports; this simple component measure is not added to the
        reward because it cannot establish load-to-support connectivity.
        """
        grid = self.density_field() if densities is None else np.asarray(densities)
        grid = grid.reshape(self.height, self.width)
        solid = grid > 0.2
        visited = np.zeros_like(solid, dtype=bool)
        components = []
        for row in range(self.height):
            for col in range(self.width):
                if not solid[row, col] or visited[row, col]:
                    continue
                stack = [(row, col)]
                visited[row, col] = True
                size = 0
                while stack:
                    current_row, current_col = stack.pop()
                    size += 1
                    for next_row, next_col in (
                        (current_row - 1, current_col),
                        (current_row + 1, current_col),
                        (current_row, current_col - 1),
                        (current_row, current_col + 1),
                    ):
                        if (
                            0 <= next_row < self.height
                            and 0 <= next_col < self.width
                            and solid[next_row, next_col]
                            and not visited[next_row, next_col]
                        ):
                            visited[next_row, next_col] = True
                            stack.append((next_row, next_col))
                components.append(size)
        if len(components) <= 1:
            return 0.0
        return float((sum(components) - max(components)) / max(sum(components), 1))

    def calculate_reward(self):
        """Evaluate scale-invariant physical and density reward terms."""
        densities = self.density_field()
        compliance, volume_fraction = fem.optim(self.args, x=densities, verbose=False)
        compliance = float(compliance)
        volume_fraction = float(volume_fraction)
        self.current_compliance = compliance
        self.current_constraint = volume_fraction

        if np.isfinite(compliance) and compliance > 0:
            compliance_ratio = compliance / self.base_compliance_value
        else:
            compliance_ratio = 1e12
        # A log ratio preserves the ordering of the compliance objective while
        # keeping near-singular structures from producing billion-scale PPO
        # targets. One reward unit corresponds to one decade of compliance.
        log_compliance_ratio = float(np.log10(max(compliance_ratio, 1e-12)))
        r_compliance = -self.w_compliance * log_compliance_ratio

        grey_fraction = float(np.mean((densities > 0.2) & (densities < 0.8)))
        r_grey = -self.w_grey * grey_fraction
        density_extremeness = float(np.mean((0.5 - densities) ** 2))
        r_density_low = self.w_density_low * density_extremeness

        # Relative volume violation is dimensionless and therefore has the
        # same meaning on 6x6 and 50x50 grids.
        volume_violation = max(volume_fraction - self.target_density, 0.0) / self.target_density
        r_mass = -self.w_total_mass * volume_violation
        reward = r_compliance + r_grey + r_density_low + r_mass
        info = {
            "raw_reward": float(reward),
            "compliance": compliance,
            "base_compliance": self.base_compliance_value,
            "compliance_ratio": float(compliance_ratio),
            "log_compliance_ratio": log_compliance_ratio,
            "volume_fraction": volume_fraction,
            "target_volume_fraction": self.target_density,
            "volume_violation": float(volume_violation),
            "constraint": volume_fraction,
            "r_compliance": float(r_compliance),
            "r_grey": float(r_grey),
            "r_density_low": float(r_density_low),
            "r_mass": float(r_mass),
        }
        return float(reward), info

    def step(self, action):
        """Increase or decrease one physical cell and evaluate the result."""
        action = int(action)
        total_cells = self.width * self.height
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action}; expected [0, {2 * total_cells})")

        cell = action % total_cells
        row, col = divmod(cell, self.width)
        direction = "increase" if action >= total_cells else "decrease"
        progress = np.clip(self.current_step / self.max_steps, 0.0, 1.0)
        self.step_size = max(
            self.step_size_initial
            - (self.step_size_initial - self.step_size_final) * progress,
            self.step_size_final,
        )

        compliance_before = self.previous_compliance
        old_density = float(self.state[0, row, col])
        delta = self.step_size if direction == "increase" else -self.step_size
        new_density = float(np.clip(old_density + delta, 0.001, 1.0))
        changed = not np.isclose(new_density, old_density)
        self.state[0, row, col] = new_density
        reward_action = 0.05 if changed else -0.05
        self.current_step += 1

        main_reward, reward_info = self.calculate_reward()
        compliance_after = self.current_compliance
        self.previous_compliance = compliance_after
        if (
            np.isfinite(compliance_before)
            and np.isfinite(compliance_after)
            and compliance_before > 0
            and compliance_after > 0
        ):
            compliance_diff = compliance_before - compliance_after
            reward_improvement = (
                self.improvement_scale
                * np.log10(compliance_before / compliance_after)
            )
        else:
            compliance_diff = 0.0
            reward_improvement = 0.0

        reward = float(np.clip(reward_action + main_reward + reward_improvement, -10.0, 10.0))
        compliance_ratio = reward_info["compliance_ratio"]
        is_new_best = compliance_ratio < self.best_compliance_ratio
        if is_new_best:
            self.best_compliance_ratio = compliance_ratio
        self.reward = reward
        terminated = self.current_step >= self.max_steps
        info = {
            **reward_info,
            "compliance_before": float(compliance_before),
            "compliance_after": float(compliance_after),
            "compliance_diff": float(compliance_diff),
            "reward_improvement": float(reward_improvement),
            "reward_action": float(reward_action),
            "reward_main": float(main_reward),
            "best_compliance_ratio": float(self.best_compliance_ratio),
            "is_new_best": bool(is_new_best),
            "direction": direction,
            "cell": int(cell),
            "row": int(row),
            "col": int(col),
            "step_size": float(self.step_size),
            "step": self.current_step,
        }
        return self.state.numpy().copy(), reward, terminated, False, info

    def render(self):
        grid = self.density_field()
        plt.figure(figsize=(6, 6))
        plt.imshow(grid, cmap="viridis", interpolation="nearest", vmin=0, vmax=1)
        plt.colorbar()
        plt.title(
            f"Step: {self.current_step} | Reward: {self.reward:.3f} | "
            f"Compliance: {self.current_compliance:.3f}"
        )
        plt.xlabel("X / width")
        plt.ylabel("Y / height")
        plt.show()

    def close(self):
        plt.close("all")
