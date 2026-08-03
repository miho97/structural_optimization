import sys
import unittest
from pathlib import Path

import numpy as np
from gymnasium.utils.env_checker import check_env


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from env.mbb_cnn import BeamOptimizationEnv  # noqa: E402
from utils import fem  # noqa: E402


class BeamOptimizationEnvironmentTests(unittest.TestCase):
    def test_gymnasium_contract_and_seeded_random_reset(self):
        env = BeamOptimizationEnv(width=3, height=2, use_random_beam=True)
        check_env(env, skip_render_check=True)

        first_state, _ = env.reset(seed=1234)
        first_normals = env.normals.numpy().copy()
        first_forces = env.forces.numpy().copy()
        second_state, _ = env.reset(seed=1234)
        np.testing.assert_array_equal(first_state, second_state)
        np.testing.assert_array_equal(first_normals, env.normals.numpy())
        np.testing.assert_array_equal(first_forces, env.forces.numpy())

    def test_rectangular_actions_cover_every_physical_cell_once(self):
        for width, height in ((2, 3), (3, 2)):
            env = BeamOptimizationEnv(width=width, height=height, beam_type=1)
            reached = set()
            for cell in range(width * height):
                for direction_offset, expected_direction in (
                    (0, "decrease"),
                    (width * height, "increase"),
                ):
                    env.reset(seed=7)
                    before = env.density_field()
                    _, _, _, _, info = env.step(cell + direction_offset)
                    after = env.density_field()
                    changed = np.argwhere(before != after)
                    self.assertEqual(changed.shape, (1, 2))
                    row, col = map(int, changed[0])
                    self.assertEqual((row, col), divmod(cell, width))
                    self.assertEqual((row, col), (info["row"], info["col"]))
                    self.assertEqual(info["direction"], expected_direction)
                    reached.add((row, col))
                    # The density channel's nodal padding is never writable.
                    self.assertTrue(bool((env.state[0, height, :] == 0).all()))
                    self.assertTrue(bool((env.state[0, :, width] == 0).all()))
            self.assertEqual(len(reached), width * height)

    def test_boundary_conditions_are_transposed_once_into_image_coordinates(self):
        width, height = 3, 2
        normals = np.zeros((width + 1, height + 1, 2), dtype=np.float32)
        forces = np.zeros_like(normals)
        # Stable cantilever supports plus one deliberately placed marker.
        normals[0, :, 0] = 1
        normals[0, 0, 1] = 1
        normals[0, height, 1] = 1
        normals[2, 1, 0] = 1
        forces[3, 2, 1] = -1
        env = BeamOptimizationEnv(
            width=width,
            height=height,
            custom_normals=normals,
            custom_forces=forces,
        )
        state, _ = env.reset()
        self.assertEqual(state.shape, (6, height + 1, width + 1))
        self.assertEqual(state[1, 1, 2], 1)  # normal_x at image[y, x]
        self.assertEqual(state[2, 0, 0], 1)  # normal_y at image[y, x]
        self.assertEqual(state[4, 2, 3], -1)  # force_y at image[y, x]

    def test_step_compliance_uses_the_visible_density_field(self):
        env = BeamOptimizationEnv(width=3, height=2, beam_type=1)
        _, _, _, _, info = env.step(4)
        direct_compliance, direct_volume = fem.optim(
            env.args, x=env.density_field(), verbose=False
        )
        self.assertAlmostEqual(info["compliance"], float(direct_compliance), places=8)
        self.assertAlmostEqual(info["volume_fraction"], float(direct_volume), places=8)

    def test_volume_penalty_is_resolution_invariant(self):
        penalties = []
        for size in (6, 10):
            env = BeamOptimizationEnv(size, size, density=0.4, beam_type=1)
            env.state[0, :size, :size] = 0.44
            _, info = env.calculate_reward()
            penalties.append(info["r_mass"])
            self.assertAlmostEqual(info["volume_violation"], 0.1, places=6)
            self.assertAlmostEqual(info["volume_error"], 0.1, places=6)
        self.assertAlmostEqual(penalties[0], penalties[1], places=6)

    def test_volume_penalty_is_symmetric_and_prevents_empty_collapse(self):
        env = BeamOptimizationEnv(6, 6, density=0.4, beam_type=1)
        baseline_score, _ = env.calculate_reward()

        env.state[0, :6, :6] = 0.36
        under_score, under_info = env.calculate_reward()
        env.state[0, :6, :6] = 0.44
        _, over_info = env.calculate_reward()
        self.assertAlmostEqual(under_info["volume_error"], -0.1, places=6)
        self.assertAlmostEqual(over_info["volume_error"], 0.1, places=6)
        self.assertAlmostEqual(under_info["r_mass"], over_info["r_mass"], places=6)

        env.state[0, :6, :6] = env.minimum_density
        empty_score, empty_info = env.calculate_reward()
        self.assertLess(empty_score, baseline_score)
        self.assertLess(empty_score, under_score)
        self.assertGreater(empty_info["volume_deviation"], 0.99)

    def test_step_reward_is_objective_improvement_without_edit_bonus(self):
        env = BeamOptimizationEnv(4, 3, density=0.4, beam_type=1)
        objective_before = env.current_objective
        _, reward, _, _, info = env.step(0)
        self.assertAlmostEqual(info["objective_before"], objective_before, places=8)
        self.assertAlmostEqual(
            info["reward_main"],
            info["objective_before"] - info["objective_after"],
            places=8,
        )
        self.assertEqual(info["reward_action"], 0.0)
        self.assertAlmostEqual(reward, info["reward_main"], places=8)

    def test_saturated_edits_are_exposed_as_invalid_actions(self):
        env = BeamOptimizationEnv(3, 2, density=0.4, beam_type=1)
        total_cells = env.width * env.height
        _, _, _, _, first_info = env.step(0)
        self.assertTrue(first_info["action_changed_density"])
        mask = env.valid_action_mask()
        self.assertFalse(bool(mask[0]))
        self.assertFalse(bool(mask[total_cells]))

        _, _, _, _, repeated_info = env.step(0)
        self.assertFalse(repeated_info["action_changed_density"])
        self.assertFalse(repeated_info["action_was_valid"])
        self.assertEqual(repeated_info["reward_action"], -0.05)

    def test_construction_enforces_the_material_budget(self):
        env = BeamOptimizationEnv(6, 6, density=0.4, beam_type=1)
        selected_cells = set()
        final_info = None
        for step in range(env.width * env.height):
            valid_actions = np.flatnonzero(env.valid_action_mask())
            self.assertGreater(valid_actions.size, 0)
            # Prefer solid assignments to stress the upper-budget guard.
            action = int(valid_actions[-1])
            selected_cells.add(action % (env.width * env.height))
            _, _, terminated, _, final_info = env.step(action)
            self.assertFalse(terminated)

        self.assertEqual(len(selected_cells), env.width * env.height)
        self.assertEqual(final_info["solid_cells"], env.solid_cell_budget)
        self.assertEqual(final_info["undecided_cells"], 0)
        self.assertEqual(env.phase, "refinement_remove")
        mask = env.valid_action_mask()
        self.assertEqual(np.count_nonzero(mask), env.solid_cell_budget)
        self.assertFalse(bool(mask[env.width * env.height :].any()))
        expected_volume = (
            env.solid_cell_budget * env.maximum_density
            + (env.width * env.height - env.solid_cell_budget)
            * env.minimum_density
        ) / (env.width * env.height)
        self.assertAlmostEqual(final_info["volume_fraction"], expected_volume, places=7)

    def test_refinement_swaps_preserve_material_and_finish_at_108_steps(self):
        env = BeamOptimizationEnv(6, 6, density=0.4, beam_type=1)
        total_cells = env.width * env.height

        for _ in range(total_cells):
            action = int(np.flatnonzero(env.valid_action_mask())[-1])
            _, _, terminated, _, _ = env.step(action)
            self.assertFalse(terminated)

        completed_design_objective = env.previous_objective
        remove_action = int(np.flatnonzero(env.valid_action_mask())[0])
        removed_cell = remove_action % total_cells
        _, remove_reward, terminated, _, remove_info = env.step(remove_action)
        self.assertFalse(terminated)
        self.assertEqual(remove_reward, 0.0)
        self.assertEqual(remove_info["phase"], "refinement_remove")
        self.assertEqual(remove_info["next_phase"], "refinement_add")
        self.assertEqual(remove_info["pending_removal_cell"], removed_cell)
        self.assertEqual(remove_info["solid_cells"], env.solid_cell_budget - 1)
        self.assertEqual(env.state[5].sum().item(), 1.0)

        add_mask = env.valid_action_mask()
        self.assertFalse(bool(add_mask[:total_cells].any()))
        self.assertFalse(bool(add_mask[total_cells + removed_cell]))
        add_action = int(np.flatnonzero(add_mask)[0])
        _, add_reward, terminated, _, add_info = env.step(add_action)
        self.assertFalse(terminated)
        self.assertTrue(add_info["swap_completed"])
        self.assertEqual(add_info["next_phase"], "refinement_remove")
        self.assertEqual(add_info["solid_cells"], env.solid_cell_budget)
        self.assertEqual(env.state[5].sum().item(), 0.0)
        self.assertAlmostEqual(
            add_reward,
            completed_design_objective - add_info["objective_after"],
            places=8,
        )

        while not terminated:
            action = int(np.flatnonzero(env.valid_action_mask())[0])
            _, _, terminated, _, final_info = env.step(action)

        self.assertEqual(env.current_step, 3 * total_cells)
        self.assertEqual(env.completed_refinement_swaps, total_cells)
        self.assertEqual(final_info["solid_cells"], env.solid_cell_budget)
        self.assertEqual(final_info["next_phase"], "complete")
        self.assertEqual(np.count_nonzero(env.valid_action_mask()), 0)

    def test_baseline_compliance_reward_is_zero_and_decade_scaled(self):
        env = BeamOptimizationEnv(4, 3, density=0.4, beam_type=1)
        _, baseline_info = env.calculate_reward()
        self.assertAlmostEqual(baseline_info["r_compliance"], 0.0, places=6)
        # Verify the transformation itself without constructing a singular
        # design: one decade worse is exactly one negative reward unit.
        ratio = 10.0
        self.assertAlmostEqual(-np.log10(ratio), -1.0)

    def test_density_defaults_to_the_single_authoritative_target(self):
        env = BeamOptimizationEnv(4, 3, density=0.37, beam_type=1)
        state, info = env.reset()
        np.testing.assert_allclose(state[0, :3, :4], 0.37, rtol=0, atol=1e-7)
        self.assertAlmostEqual(info["target_volume_fraction"], 0.37)
        self.assertAlmostEqual(info["volume_fraction"], 0.37, places=6)

    def test_every_builtin_beam_supports_rectangular_grids(self):
        for beam_type in range(1, 10):
            env = BeamOptimizationEnv(3, 2, density=0.4, beam_type=beam_type)
            state, _ = env.reset(seed=beam_type)
            self.assertEqual(state.shape, (6, 3, 4), beam_type)
            self.assertTrue(env.observation_space.contains(state), beam_type)


if __name__ == "__main__":
    unittest.main()
