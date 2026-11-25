import sys
import unittest
from pathlib import Path

import numpy as np
from gymnasium.utils.env_checker import check_env


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from env.mbb_cnn import (  # noqa: E402
    LOG_RATIO_KNEE,
    SOLID,
    UNDECIDED,
    VOID,
    BeamOptimizationEnv,
    count_corner_contacts,
    soft_log_ratio,
)
from utils import fem, layout  # noqa: E402


def construct(env, solid_cells):
    """Finish construction with exactly ``solid_cells`` solid."""
    cells = env.total_cells
    solid_cells = set(solid_cells)
    info = None
    for cell in sorted(solid_cells):
        _, _, _, _, info = env.step(cells + cell)
    for cell in range(cells):
        if cell not in solid_cells:
            _, _, _, _, info = env.step(cell)
    return info


def random_episode(env, rng, stop_probability=0.2, seed=None):
    obs, info = env.reset(seed=seed)
    rewards = []
    while True:
        valid = np.flatnonzero(env.valid_action_mask())
        if env.stop_action in valid and len(valid) > 1 and rng.random() > stop_probability:
            valid = valid[valid != env.stop_action]
        obs, reward, terminated, truncated, info = env.step(int(rng.choice(valid)))
        rewards.append(reward)
        if terminated or truncated:
            return obs, rewards, info


class BeamOptimizationEnvironmentTests(unittest.TestCase):
    def test_gymnasium_contract_and_seeded_random_reset(self):
        env = BeamOptimizationEnv(width=3, height=2, use_random_beam=True)
        check_env(env, skip_render_check=True)

        first_state, _ = env.reset(seed=1234)
        first_normals = env.normals.copy()
        first_forces = env.forces.copy()
        second_state, _ = env.reset(seed=1234)
        np.testing.assert_array_equal(first_state, second_state)
        np.testing.assert_array_equal(first_normals, env.normals)
        np.testing.assert_array_equal(first_forces, env.forces)

    def test_action_space_is_two_per_cell_plus_stop(self):
        env = BeamOptimizationEnv(width=5, height=3)
        self.assertEqual(env.action_space.n, 2 * 15 + 1)
        self.assertEqual(env.stop_action, 30)
        self.assertEqual(env.observation_space.shape, (layout.NUM_CHANNELS, 4, 6))

    def test_rectangular_actions_address_row_major_cells(self):
        for width, height in ((2, 3), (3, 2)):
            env = BeamOptimizationEnv(width=width, height=height, density=0.5)
            cells = width * height
            env.step(cells + (cells - 1))        # last cell solid
            env.step(1)                          # second cell void
            field = env.density_field()
            self.assertEqual(field.shape, (height, width))
            self.assertEqual(field[height - 1, width - 1], 1.0)
            self.assertEqual(field[0, 1], 0.001)
            self.assertEqual(field[0, 0], 0.5)

    def test_boundary_conditions_are_transposed_once_into_image_coordinates(self):
        width, height = 4, 2
        normals = np.zeros((width + 1, height + 1, 2))
        forces = np.zeros((width + 1, height + 1, 2))
        normals[0, :, 0] = 1
        normals[0, 0, 1] = 1
        normals[4, 2, 1] = 1
        forces[3, 1, 1] = -1
        env = BeamOptimizationEnv(
            width=width, height=height, custom_normals=normals, custom_forces=forces
        )
        obs, _ = env.reset()
        self.assertEqual(obs[layout.SUPPORT_Y, 2, 4], 1.0)   # [y, x]
        self.assertEqual(obs[layout.FORCE_Y, 1, 3], -1.0)
        np.testing.assert_array_equal(obs[layout.SUPPORT_X, :, 0], np.ones(height + 1))
        self.assertEqual(float(np.abs(obs[layout.FORCE_X]).sum()), 0.0)

    def test_observation_layout(self):
        env = BeamOptimizationEnv(width=4, height=3, density=0.5, max_refinement_swaps=2)
        obs, _ = env.reset()
        cells = obs[:, :3, :4]
        np.testing.assert_array_equal(cells[layout.UNDECIDED], np.ones((3, 4)))
        self.assertEqual(float(obs[layout.BUDGET_LEFT, 0, 0]), env.solid_cell_budget / 12)
        self.assertEqual(float(obs[layout.SWAPS_LEFT, 0, 0]), 1.0)
        self.assertEqual(float(obs[layout.TARGET, 0, 0]), env.solid_cell_budget / 12)
        self.assertTrue(np.all(np.abs(obs[layout.ENERGY]) <= 1.0))
        self.assertGreater(float(np.abs(obs[layout.ENERGY]).sum()), 0.0)
        # Cell channels are zero in the padded last row and column.
        for channel in (layout.UNDECIDED, layout.SOLID, layout.VOID, layout.ENERGY):
            self.assertEqual(float(np.abs(obs[channel, 3, :]).sum()), 0.0)
            self.assertEqual(float(np.abs(obs[channel, :, 4]).sum()), 0.0)

        construct(env, range(env.solid_cell_budget))
        obs, *_ = env.step(0)                    # remove solid cell 0
        cells = obs[:, :3, :4]
        one_hot = cells[layout.UNDECIDED] + cells[layout.SOLID] + cells[layout.VOID]
        np.testing.assert_array_equal(one_hot, np.ones((3, 4)))
        self.assertEqual(float(obs[layout.PENDING, 0, 0]), 1.0)
        self.assertEqual(float(obs[layout.PENDING].sum()), 1.0)
        self.assertAlmostEqual(float(obs[layout.BUDGET_LEFT, 0, 0]), 1 / 12, places=6)

    def test_step_compliance_matches_the_reference_solver(self):
        env = BeamOptimizationEnv(width=4, height=3, density=0.5, beam_type=3)
        _, _, _, _, info = env.step(env.total_cells + 5)
        _, _, _, _, info = env.step(2)
        expected, _ = fem.optim(env.args, x=env.density_field(), verbose=False)
        self.assertAlmostEqual(info["compliance"] / float(expected), 1.0, delta=1e-9)

    def test_reset_objective_is_zero_and_decade_scaled(self):
        env = BeamOptimizationEnv(width=4, height=4, density=0.4)
        _, info = env.reset()
        self.assertEqual(info["objective_cost"], 0.0)
        self.assertEqual(info["compliance_ratio"], 1.0)
        self.assertAlmostEqual(soft_log_ratio(1000.0, 10.0), LOG_RATIO_KNEE, places=12)
        self.assertAlmostEqual(soft_log_ratio(1.0, 10.0), -1.0, places=12)
        # Above the knee the slope is shallow but never flat.
        self.assertAlmostEqual(soft_log_ratio(1e7, 10.0), LOG_RATIO_KNEE + 0.4, places=12)
        self.assertLess(soft_log_ratio(1e6, 10.0), soft_log_ratio(1e7, 10.0))

    def test_episode_reward_equals_the_drop_in_log_compliance(self):
        rng = np.random.default_rng(0)
        env = BeamOptimizationEnv(
            width=5, height=4, density=0.6, use_random_beam=True,
            reward_weights={"w_contact": 0.0},
        )
        for seed in range(10):
            _, rewards, info = random_episode(env, rng, seed=seed)
            total = sum(rewards)
            self.assertAlmostEqual(total, -info["objective_cost"], places=9)
            if info["compliance_ratio"] < 10 ** LOG_RATIO_KNEE:
                self.assertAlmostEqual(total, -np.log10(info["compliance_ratio"]), places=9)

    def test_construction_enforces_the_material_budget(self):
        rng = np.random.default_rng(1)
        for width, height, density in ((6, 6, 0.4), (5, 3, 0.25), (3, 4, 0.8)):
            env = BeamOptimizationEnv(width=width, height=height, density=density)
            for _ in range(5):
                env.reset()
                for _ in range(env.total_cells):
                    valid = np.flatnonzero(env.valid_action_mask())
                    self.assertNotIn(env.stop_action, valid)
                    _, _, _, _, info = env.step(int(rng.choice(valid)))
                self.assertEqual(info["undecided_cells"], 0)
                self.assertEqual(info["solid_cells"], env.solid_cell_budget)
                self.assertEqual(env.phase, "refinement_remove")

    def test_swap_pays_on_completion_and_preserves_material(self):
        env = BeamOptimizationEnv(width=4, height=3, density=0.5, max_refinement_swaps=2)
        budget = env.solid_cell_budget
        construct(env, range(budget))
        objective_before = env.current_objective
        mask = env.valid_action_mask()
        np.testing.assert_array_equal(np.flatnonzero(mask[:12]), np.arange(budget))
        self.assertFalse(mask[12:24].any())
        self.assertTrue(mask[env.stop_action])

        _, reward, terminated, _, info = env.step(0)             # propose removing cell 0
        self.assertEqual(reward, 0.0)
        self.assertFalse(terminated)
        self.assertEqual(env.phase, "refinement_add")
        self.assertEqual(info["solid_cells"], budget - 1)
        mask = env.valid_action_mask()
        self.assertFalse(mask[:12].any())
        self.assertFalse(mask[12 + 0])                           # cannot refill the removed cell
        self.assertFalse(mask[env.stop_action])                  # cannot stop mid-swap
        self.assertTrue(mask[12 + budget])

        _, reward, terminated, _, info = env.step(12 + budget)   # complete the swap
        self.assertTrue(info["swap_completed"])
        self.assertEqual(info["solid_cells"], budget)
        self.assertAlmostEqual(reward, objective_before - env.current_objective, places=12)
        self.assertFalse(terminated)

        env.step(1)
        _, _, terminated, _, info = env.step(12 + 0)             # second swap hits the cap
        self.assertTrue(terminated)
        self.assertEqual(info["completed_refinement_swaps"], 2)
        self.assertEqual(env.current_step, env.max_steps)
        self.assertEqual(env.phase, "complete")
        with self.assertRaises(RuntimeError):
            env.step(env.stop_action)

    def test_stop_action_ends_the_episode_without_reward(self):
        env = BeamOptimizationEnv(width=3, height=3, density=0.5)
        self.assertFalse(env.valid_action_mask()[env.stop_action])
        construct(env, range(env.solid_cell_budget))
        design = env.density_field()
        _, reward, terminated, truncated, info = env.step(env.stop_action)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(reward, 0.0)
        np.testing.assert_array_equal(design, env.density_field())
        self.assertFalse(env.valid_action_mask().any())

    def test_invalid_actions_are_penalized_and_change_nothing(self):
        env = BeamOptimizationEnv(width=3, height=3, density=0.5)
        env.step(0)
        before = env.density_field()
        _, reward, terminated, _, info = env.step(0)             # cell 0 is already decided
        self.assertEqual(reward, -0.05)
        self.assertFalse(info["action_was_valid"])
        self.assertFalse(terminated)
        np.testing.assert_array_equal(before, env.density_field())
        _, reward, _, _, _ = env.step(env.stop_action)           # stop during construction
        self.assertEqual(reward, -0.05)

    def test_corner_contacts_are_counted_and_cost_the_objective(self):
        status = np.full((3, 3), VOID, dtype=np.int8)
        status[0, 0] = status[1, 1] = status[2, 2] = SOLID       # diagonal chain: two contacts
        self.assertEqual(count_corner_contacts(status), 2)
        status[0, 1] = SOLID                                     # an edge neighbour removes one
        self.assertEqual(count_corner_contacts(status), 1)
        status[1, 2] = UNDECIDED                                 # undecided is not yet a hinge
        self.assertEqual(count_corner_contacts(status), 0)
        anti = np.full((2, 2), VOID, dtype=np.int8)
        anti[0, 1] = anti[1, 0] = SOLID
        self.assertEqual(count_corner_contacts(anti), 1)

        field = np.where(np.eye(4, dtype=bool), 1.0, 0.001)
        with_cost = BeamOptimizationEnv(width=4, height=4, reward_weights={"w_contact": 0.02})
        without = BeamOptimizationEnv(width=4, height=4, reward_weights={"w_contact": 0.0})
        scored = with_cost.evaluate_design(field)
        self.assertEqual(scored["contacts"], 3)
        self.assertAlmostEqual(
            scored["objective_cost"] - without.evaluate_design(field)["objective_cost"],
            0.06,
            places=12,
        )
        with self.assertRaises(ValueError):
            BeamOptimizationEnv(width=4, height=4, reward_weights={"w_typo": 1.0})

    def test_best_completed_design_is_tracked(self):
        rng = np.random.default_rng(4)
        env = BeamOptimizationEnv(width=5, height=4, density=0.5, use_random_beam=True)
        for seed in range(8):
            env.reset(seed=seed)
            self.assertIsNone(env.best_density_field())
            completed = []
            while True:
                valid = np.flatnonzero(env.valid_action_mask())
                if len(valid) > 1:
                    valid = valid[valid != env.stop_action]
                _, _, terminated, _, info = env.step(int(rng.choice(valid)))
                if info["undecided_cells"] == 0 and env.pending_removal_cell is None:
                    completed.append(info["objective_cost"])
                if terminated:
                    break
            self.assertAlmostEqual(info["best_objective"], min(completed), places=12)
            best = env.best_density_field()
            self.assertAlmostEqual(
                env.evaluate_design(best)["objective_cost"], info["best_objective"], places=9
            )
            self.assertEqual(int(np.count_nonzero(best == 1.0)), env.solid_cell_budget)

    def test_evaluate_design_matches_the_episode_objective(self):
        rng = np.random.default_rng(5)
        env = BeamOptimizationEnv(width=6, height=4, density=0.4, beam_type=5)
        _, _, info = random_episode(env, rng)
        scored = env.evaluate_design(env.density_field())
        self.assertAlmostEqual(scored["objective_cost"], info["objective_cost"], places=9)
        self.assertEqual(scored["contacts"], info["contacts"])

    def test_density_range_resamples_the_budget(self):
        env = BeamOptimizationEnv(
            width=6, height=5, use_random_beam=True, density_range=(0.3, 0.6)
        )
        budgets = set()
        for seed in range(12):
            obs, _ = env.reset(seed=seed)
            self.assertTrue(0.3 <= env.target_density <= 0.6)
            self.assertAlmostEqual(
                float(obs[layout.TARGET, 0, 0]), env.solid_cell_budget / 30, places=6
            )
            budgets.add(env.solid_cell_budget)
        self.assertGreater(len(budgets), 1)
        with self.assertRaises(ValueError):
            BeamOptimizationEnv(width=4, height=4, density_range=(0.3, 0.6))

    def test_reset_can_keep_or_regenerate_the_random_problem(self):
        env = BeamOptimizationEnv(width=6, height=6, use_random_beam=True, random_domain="rich")
        env.reset(seed=1)
        normals = env.normals.copy()
        env.reset(options={"regenerate_beam": False})
        np.testing.assert_array_equal(normals, env.normals)
        changed = False
        for _ in range(5):
            env.reset()
            changed = changed or not np.array_equal(normals, env.normals)
        self.assertTrue(changed)

    def test_every_builtin_beam_supports_rectangular_grids(self):
        for beam_type in range(1, 10):
            for width, height in ((6, 4), (4, 6)):
                env = BeamOptimizationEnv(width=width, height=height, beam_type=beam_type)
                self.assertTrue(np.isfinite(env.base_compliance_value))
                self.assertGreater(env.base_compliance_value, 0.0)

    def test_constructor_validation(self):
        with self.assertRaises(ValueError):
            BeamOptimizationEnv(width=0, height=3)
        with self.assertRaises(ValueError):
            BeamOptimizationEnv(width=3, height=3, density=1.0)
        with self.assertRaises(ValueError):
            BeamOptimizationEnv(width=3, height=3, max_steps=8)      # fewer steps than cells
        with self.assertRaises(ValueError):
            BeamOptimizationEnv(width=3, height=3, max_steps=10)     # odd refinement steps
        env = BeamOptimizationEnv(width=3, height=3, max_steps=13)
        self.assertEqual(env.max_refinement_swaps, 2)


if __name__ == "__main__":
    unittest.main()
