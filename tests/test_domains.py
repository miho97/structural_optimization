import sys
import unittest
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from env import domains  # noqa: E402
from utils.fem_fast import FastFEM  # noqa: E402


class DomainSamplerTests(unittest.TestCase):
    def test_rich_domains_are_stable_normalized_and_solvable(self):
        for width, height, density in ((6, 6, 0.4), (12, 8, 0.45), (4, 9, 0.3)):
            rng = np.random.default_rng(7)
            budget = domains.solid_cell_budget(width, height, density)
            support_families = set()
            load_families = set()
            for _ in range(120):
                normals, forces, meta = domains.sample_rich_domain(width, height, budget, rng)
                support_families.add(meta["support_family"])
                load_families.add(meta["load_family"])
                self.assertEqual(normals.shape, (width + 1, height + 1, 2))
                self.assertEqual(domains.rigid_mode_rank(normals), 3)
                self.assertAlmostEqual(float(np.linalg.norm(forces, axis=2).sum()), 1.0, places=9)
                # No load is applied to a fixed degree of freedom.
                self.assertFalse(np.any(forces[normals.astype(bool)]))
                self.assertLessEqual(domains.material_path_size(normals, forces), budget)
                compliance = FastFEM(normals, forces).solve(np.full((height, width), density))
                self.assertTrue(np.isfinite(compliance) and compliance > 0)
            self.assertEqual(support_families, set(domains.SUPPORT_FAMILIES))
            self.assertEqual(load_families, set(domains.LOAD_FAMILIES))

    def test_sampling_is_reproducible(self):
        first = domains.sample_domain(8, 5, 0.4, np.random.default_rng(3))
        second = domains.sample_domain(8, 5, 0.4, np.random.default_rng(3))
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])

    def test_kinds(self):
        rng = np.random.default_rng(0)
        _, _, meta = domains.sample_domain(6, 6, 0.4, rng, kind="legacy")
        self.assertEqual(meta["support_family"], "legacy template")
        _, _, meta = domains.sample_domain(6, 6, 0.4, rng, kind="rich")
        self.assertIn(meta["support_family"], domains.SUPPORT_FAMILIES)
        with self.assertRaises(ValueError):
            domains.sample_domain(6, 6, 0.4, rng, kind="other")

    def test_rigid_mode_rank_detects_mechanisms(self):
        normals = np.zeros((5, 4, 2))
        self.assertEqual(domains.rigid_mode_rank(normals), 0)
        normals[0, 0, :] = 1          # one pin: free to rotate
        self.assertEqual(domains.rigid_mode_rank(normals), 2)
        normals[4, 0, 1] = 1          # plus a roller: stable
        self.assertEqual(domains.rigid_mode_rank(normals), 3)

    def test_solid_cell_budget_matches_target_volume(self):
        budget = domains.solid_cell_budget(12, 8, 0.45)
        self.assertEqual(budget, 43)
        volume = (budget + (96 - budget) * 0.001) / 96
        self.assertAlmostEqual(volume, 0.45, delta=0.006)


if __name__ == "__main__":
    unittest.main()
