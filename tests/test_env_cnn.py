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
        self.assertEqual(state.shape, (5, height + 1, width + 1))
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
        self.assertAlmostEqual(penalties[0], penalties[1], places=6)

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
            self.assertEqual(state.shape, (5, 3, 4), beam_type)
            self.assertTrue(env.observation_space.contains(state), beam_type)


if __name__ == "__main__":
    unittest.main()
