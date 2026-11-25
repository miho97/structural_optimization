"""Gymnasium environment for density-based structural optimization.

Coordinate convention
---------------------
``width`` is the number of finite elements along X (columns) and ``height`` is
the number along Y (rows). Density fields and images use ``(height, width)`` /
``(channels, height + 1, width + 1)`` ordering. The FEM helpers store nodal
data in ``(x, y, dof)`` ordering, so nodal channels are transposed exactly once
when the observation is built.

Episode
-------
1. Construction: every cell starts undecided at the target density and is
   assigned exactly once to void or solid. The action mask enforces the solid
   budget, so a finished construction always has the target volume.
2. Refinement: a swap removes one solid cell and then fills a different void
   cell, which preserves the volume. The stop action ends the episode, and at
   most ``max_refinement_swaps`` swaps are allowed.

Actions are ``0 .. N-1`` (make cell void), ``N .. 2N-1`` (make cell solid) and
``2N`` (stop), with ``N = width * height``.

Reward
------
The objective is ``log10(C / C0)`` plus a cost per corner-only contact, where
``C0`` is the compliance of the uniform starting field. A step pays the
decrease in that objective. The removal half of a swap pays nothing and the
addition half pays for the pair, so an under-filled design is never scored.
Summed over an episode the reward equals the total drop in the objective.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from env.domains import (
    MAXIMUM_DENSITY,
    MINIMUM_DENSITY,
    legacy_random_beam,
    sample_domain,
    solid_cell_budget,
)
from utils import fem, layout
from utils.fem_fast import FastFEM

# Kept under its original name for existing notebooks.
generate_random_beam = legacy_random_beam

UNDECIDED, SOLID, VOID = 0, 1, 2
PHASES = ("construction", "refinement_remove", "refinement_add", "complete")

# Above this many decades of compliance ratio the objective keeps only a
# shallow slope. A disconnected design then costs about 2.3 to 2.5 instead of
# 5 to 7, which keeps value targets in a sane range without making all
# disconnected designs look identical.
LOG_RATIO_KNEE = 2.0
LOG_RATIO_TAIL_SLOPE = 0.1

DEFAULT_REWARD_WEIGHTS = {
    "w_compliance": 1.0,
    # Cost per pair of solid cells that touch only at a corner while both
    # cells beside that corner are void. Four-node elements transmit load
    # through such a point contact, which no real structure can do.
    # 0.02 decades is about 5% of compliance. Set to 0 to disable.
    "w_contact": 0.02,
    # Legacy terms. The action mask already enforces the volume and every
    # finished design is binary, so both default to zero.
    "w_total_mass": 0.0,
    "w_binary": 0.0,
}


def count_corner_contacts(status):
    """Number of solid pairs joined only through a corner between two voids."""
    solid = status == SOLID
    void = status == VOID
    main = solid[:-1, :-1] & solid[1:, 1:] & void[:-1, 1:] & void[1:, :-1]
    anti = solid[:-1, 1:] & solid[1:, :-1] & void[:-1, :-1] & void[1:, 1:]
    return int(np.count_nonzero(main) + np.count_nonzero(anti))


def soft_log_ratio(compliance, base_compliance):
    """``log10(C / C0)`` with a shallow slope above ``LOG_RATIO_KNEE``."""
    if not np.isfinite(compliance) or compliance <= 0:
        raw = 12.0
    else:
        raw = float(np.log10(max(compliance / base_compliance, 1e-12)))
    if raw > LOG_RATIO_KNEE:
        return LOG_RATIO_KNEE + LOG_RATIO_TAIL_SLOPE * (raw - LOG_RATIO_KNEE)
    return raw


class BeamOptimizationEnv(gym.Env):
    """Sequential density-editing environment backed by a banded FEM solver."""

    metadata = {"render_modes": ["human"]}
    minimum_density = MINIMUM_DENSITY
    maximum_density = MAXIMUM_DENSITY

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
        max_refinement_swaps=None,
        random_domain="mixed",
        density_range=None,
    ):
        super().__init__()
        del step_size  # accepted for compatibility; assignments are explicit
        if width < 1 or height < 1:
            raise ValueError("width and height must both be positive")
        if (custom_normals is None) != (custom_forces is None):
            raise ValueError("custom_normals and custom_forces must be supplied together")
        if random_domain not in ("legacy", "rich", "mixed"):
            raise ValueError("random_domain must be 'legacy', 'rich' or 'mixed'")

        self.width = int(width)
        self.height = int(height)
        self.total_cells = self.width * self.height
        self.use_random_beam = bool(use_random_beam)
        self.random_domain = random_domain
        self.beam_type = int(beam_type)
        self.custom_normals = custom_normals
        self.custom_forces = custom_forces
        if density_range is not None:
            low, high = (float(value) for value in density_range)
            if not self.minimum_density < low <= high < self.maximum_density:
                raise ValueError("density_range must lie strictly between the density bounds")
            if not self.use_random_beam:
                raise ValueError("density_range needs use_random_beam=True")
            density_range = (low, high)
        self.density_range = density_range

        target = float(density if optimal_density is None else optimal_density)
        self._check_target_density(target)

        if max_refinement_swaps is None:
            if max_steps is None:
                max_refinement_swaps = max(1, self.total_cells // 4)
            else:
                extra = int(max_steps) - self.total_cells
                if extra < 0:
                    raise ValueError("max_steps must allow one assignment per physical cell")
                if extra % 2:
                    raise ValueError(
                        "steps after construction must come in remove/add refinement pairs"
                    )
                max_refinement_swaps = extra // 2
        self.max_refinement_swaps = int(max_refinement_swaps)
        if self.max_refinement_swaps < 0:
            raise ValueError("max_refinement_swaps must not be negative")
        self.max_steps = self.total_cells + 2 * self.max_refinement_swaps

        self.stop_action = 2 * self.total_cells
        self.action_space = gym.spaces.Discrete(2 * self.total_cells + 1)
        observation_shape = (layout.NUM_CHANNELS, self.height + 1, self.width + 1)
        self.observation_space = gym.spaces.Box(-1.0, 1.0, observation_shape, dtype=np.float32)
        self.state_dim = int(np.prod(observation_shape))

        self.beam_functions = {
            index: getattr(fem, f"mbb_beam_{index}") for index in range(1, 10)
        }
        if (
            not self.use_random_beam
            and custom_normals is None
            and self.beam_type not in self.beam_functions
        ):
            raise ValueError(f"beam_type must be one of {sorted(self.beam_functions)}")

        weights = dict(DEFAULT_REWARD_WEIGHTS)
        unknown = set(reward_weights or {}) - set(weights)
        if unknown:
            raise ValueError(f"unknown reward weights: {sorted(unknown)}")
        weights.update(reward_weights or {})
        self.w_compliance = float(weights["w_compliance"])
        self.w_contact = float(weights["w_contact"])
        self.w_total_mass = float(weights["w_total_mass"])
        self.w_binary = float(weights["w_binary"])

        self._obs = np.zeros(observation_shape, dtype=np.float32)
        self.domain_meta = {}
        self._set_target_density(target)
        if custom_normals is not None:
            normals, forces = custom_normals, custom_forces
        elif self.use_random_beam:
            normals, forces = self._sample_problem()
        else:
            normals, forces, _ = self.beam_functions[self.beam_type](
                self.width, self.height, self.target_density
            )
        self._set_problem(normals, forces)
        self.reset(options={"regenerate_beam": False})

    # ------------------------------------------------------------------
    # Problem setup
    # ------------------------------------------------------------------
    def _check_target_density(self, target):
        if not self.minimum_density < target < self.maximum_density:
            raise ValueError("target density must lie strictly between the void and solid bounds")

    def _set_target_density(self, target):
        self.target_density = float(target)
        # Compatibility aliases.
        self.density = self.target_density
        self.optimal_density = self.target_density
        self.solid_cell_budget = solid_cell_budget(self.width, self.height, self.target_density)

    def _sample_problem(self):
        if self.density_range is not None:
            low, high = self.density_range
            self._set_target_density(self.np_random.uniform(low, high))
        normals, forces, meta = sample_domain(
            self.width, self.height, self.target_density, self.np_random, self.random_domain
        )
        self.domain_meta = meta
        return normals, forces

    def _set_problem(self, normals, forces):
        expected_shape = (self.width + 1, self.height + 1, 2)
        normals = np.asarray(normals, dtype=np.float64)
        forces = np.asarray(forces, dtype=np.float64)
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

        self.normals = normals
        self.forces = forces
        self.solver = FastFEM(normals, forces)
        uniform = np.full((self.height, self.width), self.target_density)
        base_compliance, base_energy = self.solver.solve(uniform, return_energy=True)
        if not np.isfinite(base_compliance) or base_compliance <= 0:
            raise ValueError("the supplied support/load problem has invalid base compliance")
        self.base_compliance_value = float(base_compliance)
        self._base_energy = base_energy
        self._args = None  # autograd problem description, built on demand

        # FEM nodal arrays are [x, y, dof]; image channels are [y, x].
        self._obs[layout.SUPPORT_X] = normals[:, :, 0].T
        self._obs[layout.SUPPORT_Y] = normals[:, :, 1].T
        self._obs[layout.FORCE_X] = forces[:, :, 0].T
        self._obs[layout.FORCE_Y] = forces[:, :, 1].T
        self._obs[layout.TARGET] = self.solid_cell_budget / self.total_cells

    @property
    def args(self):
        """``utils.fem`` problem description, for MMA and gradient code."""
        if self._args is None:
            self._args = fem.get_args(self.normals, self.forces, self.target_density)
        return self._args

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------
    @property
    def phase(self):
        return PHASES[self._phase_id]

    def _update_phase(self):
        if self._terminated:
            self._phase_id = 3
        elif self._undecided_count > 0:
            self._phase_id = 0
        elif self.pending_removal_cell is not None:
            self._phase_id = 2
        else:
            self._phase_id = 1

    def _swap_possible(self):
        return (
            self.completed_refinement_swaps < self.max_refinement_swaps
            and 0 < self._solid_count < self.total_cells
        )

    def _update_mask(self):
        cells = self.total_cells
        mask = np.zeros(2 * cells + 1, dtype=bool)
        if not self._terminated:
            status = self._status.reshape(-1)
            if self._undecided_count > 0:
                undecided = status == UNDECIDED
                # Void is allowed only while the remaining undecided cells can
                # still fill the budget; solid only until the budget is full.
                if self._solid_count + self._undecided_count - 1 >= self.solid_cell_budget:
                    mask[:cells] = undecided
                if self._solid_count < self.solid_cell_budget:
                    mask[cells : 2 * cells] = undecided
            elif self.pending_removal_cell is not None:
                mask[cells : 2 * cells] = status == VOID
                mask[cells + self.pending_removal_cell] = False
            else:
                if self._swap_possible():
                    mask[:cells] = status == SOLID
                mask[self.stop_action] = True
        self._mask = mask

    def valid_action_mask(self):
        """Boolean mask over the ``2N + 1`` actions for the current state."""
        return self._mask.copy()

    def density_field(self):
        """Copy of the physical ``(height, width)`` density field."""
        return self._density.copy()

    def best_density_field(self):
        """Best completed design of this episode, or ``None`` before one exists."""
        return None if self._best_density is None else self._best_density.copy()

    def _refresh_observation(self, energy):
        obs = self._obs
        h, w = self.height, self.width
        status = self._status
        obs[layout.UNDECIDED, :h, :w] = status == UNDECIDED
        obs[layout.SOLID, :h, :w] = status == SOLID
        obs[layout.VOID, :h, :w] = status == VOID
        obs[layout.PENDING] = 0.0
        if self.pending_removal_cell is not None:
            row, col = divmod(self.pending_removal_cell, w)
            obs[layout.PENDING, row, col] = 1.0
        mean_energy = float(np.mean(energy))
        if np.isfinite(mean_energy) and mean_energy > 0:
            scaled = np.log10(np.maximum(energy, 1e-300) / mean_energy)
            obs[layout.ENERGY, :h, :w] = np.clip(scaled, -3.0, 3.0) / 3.0
        else:
            obs[layout.ENERGY, :h, :w] = 0.0
        obs[layout.BUDGET_LEFT] = (self.solid_cell_budget - self._solid_count) / self.total_cells
        swaps_left = self.max_refinement_swaps - self.completed_refinement_swaps
        obs[layout.SWAPS_LEFT] = swaps_left / max(self.max_refinement_swaps, 1)

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------
    def _objective(self, compliance, contacts, density):
        cost = self.w_compliance * soft_log_ratio(compliance, self.base_compliance_value)
        cost += self.w_contact * contacts
        if self.w_total_mass:
            volume_error = (float(np.mean(density)) - self.target_density) / self.target_density
            cost += self.w_total_mass * volume_error**2
        if self.w_binary:
            cost += self.w_binary * float(np.mean(4.0 * density * (1.0 - density)))
        return float(cost)

    def evaluate_design(self, density):
        """Score an arbitrary density field on the current problem.

        Cells at the solid bound count as solid and cells at the void bound as
        void. Used by the benchmark and by reference optimizers so that they
        optimize exactly what the agent is rewarded for.
        """
        density = np.asarray(density, dtype=np.float64).reshape(self.height, self.width)
        status = np.full(density.shape, UNDECIDED, dtype=np.int8)
        status[density >= self.maximum_density - 1e-6] = SOLID
        status[density <= self.minimum_density + 1e-6] = VOID
        compliance = self.solver.solve(density)
        contacts = count_corner_contacts(status)
        return {
            "objective_cost": self._objective(compliance, contacts, density),
            "compliance": float(compliance),
            "compliance_ratio": float(compliance / self.base_compliance_value),
            "contacts": contacts,
        }

    def _solve_current(self):
        compliance, energy = self.solver.solve(self._density, return_energy=True)
        self.current_compliance = float(compliance)
        self.current_contacts = count_corner_contacts(self._status)
        self.current_objective = self._objective(
            self.current_compliance, self.current_contacts, self._density
        )
        return energy

    def _record_completed_design(self):
        """Track the best finished, full-budget design. Returns True if new."""
        if self._best_density is not None and self.current_objective >= self.best_objective:
            return False
        self.best_objective = self.current_objective
        self.best_compliance = self.current_compliance
        self.best_compliance_ratio = self.current_compliance / self.base_compliance_value
        self.best_contacts = self.current_contacts
        self._best_density = self._density.copy()
        return True

    def _info(self, **extra):
        info = {
            "compliance": self.current_compliance,
            "compliance_ratio": self.current_compliance / self.base_compliance_value,
            "objective_cost": self.current_objective,
            "contacts": self.current_contacts,
            "volume_fraction": float(np.mean(self._density)),
            "solid_cells": self._solid_count,
            "undecided_cells": self._undecided_count,
            "completed_refinement_swaps": self.completed_refinement_swaps,
            "phase_id": self._phase_id,
            "best_objective": self.best_objective,
            "best_compliance_ratio": self.best_compliance_ratio,
            "best_contacts": self.best_contacts,
            "step": self.current_step,
        }
        info.update(extra)
        return info

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if self.use_random_beam and options.get("regenerate_beam", True):
            normals, forces = self._sample_problem()
            self._set_problem(normals, forces)

        self._status = np.zeros((self.height, self.width), dtype=np.int8)
        self._density = np.full((self.height, self.width), self.target_density)
        self._solid_count = 0
        self._undecided_count = self.total_cells
        self.current_step = 0
        self.pending_removal_cell = None
        self.completed_refinement_swaps = 0
        self._swap_objective_before = None
        self._terminated = False
        self.reward = 0.0

        # The uniform field was already solved when the problem was set.
        self.current_compliance = self.base_compliance_value
        self.current_contacts = 0
        self.current_objective = self._objective(self.current_compliance, 0, self._density)
        self.previous_objective = self.current_objective
        self.initial_objective = self.current_objective
        self.best_objective = float("inf")
        self.best_compliance = float("inf")
        self.best_compliance_ratio = float("inf")
        self.best_contacts = 0
        self._best_density = None

        self._update_phase()
        self._update_mask()
        self._refresh_observation(self._base_energy)
        return self._obs.copy(), self._info()

    def step(self, action):
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action}; expected [0, {self.action_space.n})")
        if self._terminated:
            raise RuntimeError("step() called on a finished episode; call reset() first")

        self.current_step += 1
        phase_before = self._phase_id
        truncated = False
        swap_completed = False
        is_new_best = False

        if not self._mask[action]:
            # The policy masks these out. The penalty is a guard for callers
            # that step the environment directly.
            reward = -0.05
            truncated = self.current_step >= 2 * self.max_steps + 2
            self.reward = reward
            return (
                self._obs.copy(),
                reward,
                False,
                truncated,
                self._info(action_was_valid=False, swap_completed=False, is_new_best=False),
            )

        if action == self.stop_action:
            self._terminated = True
            self._update_phase()
            self._update_mask()
            self.reward = 0.0
            return (
                self._obs.copy(),
                0.0,
                True,
                False,
                self._info(action_was_valid=True, swap_completed=False, is_new_best=False),
            )

        cell = action % self.total_cells
        row, col = divmod(cell, self.width)
        make_solid = action >= self.total_cells
        old_status = int(self._status[row, col])
        new_status = SOLID if make_solid else VOID
        self._status[row, col] = new_status
        self._density[row, col] = self.maximum_density if make_solid else self.minimum_density
        self._solid_count += int(new_status == SOLID) - int(old_status == SOLID)
        self._undecided_count -= int(old_status == UNDECIDED)

        if phase_before == 1:  # removal half of a swap
            self.pending_removal_cell = cell
            self._swap_objective_before = self.previous_objective
        elif phase_before == 2:  # addition half of a swap
            self.pending_removal_cell = None
            self.completed_refinement_swaps += 1
            swap_completed = True

        energy = self._solve_current()

        if phase_before == 1:
            # A removal is only a proposal: the under-filled design is not
            # scored, and the pair is paid when it is completed.
            reward = 0.0
        elif swap_completed:
            reward = self._swap_objective_before - self.current_objective
            self.previous_objective = self.current_objective
            self._swap_objective_before = None
        else:
            reward = self.previous_objective - self.current_objective
            self.previous_objective = self.current_objective

        design_complete = self._undecided_count == 0 and self.pending_removal_cell is None
        if design_complete:
            is_new_best = self._record_completed_design()
            if not self._swap_possible():
                self._terminated = True

        self._update_phase()
        self._update_mask()
        self._refresh_observation(energy)
        reward = float(np.clip(reward, -10.0, 10.0))
        self.reward = reward
        info = self._info(
            action_was_valid=True, swap_completed=swap_completed, is_new_best=is_new_best
        )
        return self._obs.copy(), reward, self._terminated, truncated, info

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def check_connectivity(self, densities=None):
        """Fraction of solid cells outside the largest edge-connected component.

        A diagnostic only. Compliance already captures whether loads reach
        supports, so this is not part of the reward.
        """
        grid = self._density if densities is None else np.asarray(densities)
        solid = grid.reshape(self.height, self.width) > 0.2
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

    def render(self, best=False):
        """Draw the current design, or the best completed one with ``best=True``."""
        import matplotlib.pyplot as plt

        if best and self._best_density is not None:
            grid = self._best_density
            compliance, contacts, label = self.best_compliance, self.best_contacts, "best"
        else:
            grid = self._density
            compliance, contacts, label = self.current_compliance, self.current_contacts, "current"
        plt.figure(figsize=(6, 6))
        image = plt.imshow(grid, cmap="viridis", interpolation="nearest", vmin=0, vmax=1)
        colorbar = plt.colorbar(image)
        colorbar.set_label("Density (0 = void, 1 = solid)")
        support_x, support_y = np.where(self.normals.any(axis=2))
        plt.scatter(support_x - 0.5, support_y - 0.5, marker="s", s=40, c="cyan", label="support")
        load_x, load_y = np.where(np.linalg.norm(self.forces, axis=2) > 1e-12)
        if len(load_x):
            # Arrows show direction only, drawn about one cell long.
            vectors = self.forces[load_x, load_y]
            vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
            plt.quiver(
                load_x - 0.5,
                load_y - 0.5,
                vectors[:, 0],
                vectors[:, 1],
                color="red",
                angles="xy",
                scale_units="xy",
                scale=1.0,
            )
        plt.title(
            f"{label} | step {self.current_step} | "
            f"C/C0 {compliance / self.base_compliance_value:.3f} | "
            f"corner contacts {contacts} | "
            f"volume {float(np.mean(grid)):.3f}/{self.target_density:.3f}"
        )
        plt.xlabel("X / width")
        plt.ylabel("Y / height")
        plt.show()

    def close(self):
        try:
            import matplotlib.pyplot as plt

            plt.close("all")
        except Exception:
            pass
