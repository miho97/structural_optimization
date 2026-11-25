import sys
import unittest
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from utils import fem  # noqa: E402
from utils.fem_fast import FastFEM  # noqa: E402


def connected_binary_field(rng, width, height, fraction=0.45):
    """Random solid/void field with a solid frame, so loads reach supports."""
    field = np.full(width * height, 0.001)
    field[rng.choice(width * height, int(fraction * width * height), replace=False)] = 1.0
    field = field.reshape(height, width)
    field[0, :] = field[-1, :] = 1.0
    field[:, 0] = field[:, -1] = 1.0
    return field


class FastSolverTests(unittest.TestCase):
    def test_matches_reference_solver_on_every_builtin_beam(self):
        rng = np.random.default_rng(0)
        for width, height in ((6, 6), (12, 8), (5, 9)):
            for beam_type in range(1, 10):
                normals, forces, _ = getattr(fem, f"mbb_beam_{beam_type}")(width, height, 0.45)
                args = fem.get_args(normals, forces, 0.45)
                solver = FastFEM(normals, forces)
                fields = [
                    np.full((height, width), 0.45),
                    rng.uniform(0.2, 1.0, size=(height, width)),
                    connected_binary_field(rng, width, height),
                ]
                for field in fields:
                    expected, _ = fem.optim(args, x=field, verbose=False)
                    actual = solver.solve(field)
                    self.assertAlmostEqual(actual / float(expected), 1.0, delta=1e-9)

    def test_disconnected_designs_agree_to_solver_conditioning(self):
        # A design whose load is carried only by void material has a condition
        # number near 1e9, so two exact solvers agree to fewer digits.
        rng = np.random.default_rng(1)
        normals, forces, _ = fem.mbb_beam_1(12, 8, 0.45)
        args = fem.get_args(normals, forces, 0.45)
        solver = FastFEM(normals, forces)
        for _ in range(10):
            field = np.full(96, 0.001)
            field[rng.choice(96, 43, replace=False)] = 1.0
            field = field.reshape(8, 12)
            expected, _ = fem.optim(args, x=field, verbose=False)
            self.assertAlmostEqual(solver.solve(field) / float(expected), 1.0, delta=1e-3)

    def test_strain_energy_sums_to_compliance(self):
        rng = np.random.default_rng(2)
        normals, forces, _ = fem.mbb_beam_3(7, 4, 0.4)
        solver = FastFEM(normals, forces)
        field = rng.uniform(0.2, 1.0, size=(4, 7))
        compliance, energy = solver.solve(field, return_energy=True)
        self.assertEqual(energy.shape, (4, 7))
        self.assertTrue(np.all(energy >= 0))
        modulus = 1e-9 + field**3 * (1 - 1e-9)
        self.assertAlmostEqual(float(np.sum(modulus * energy)) / compliance, 1.0, delta=1e-9)

    def test_strain_energy_is_the_compliance_sensitivity(self):
        normals, forces, _ = fem.mbb_beam_1(5, 4, 0.5)
        solver = FastFEM(normals, forces)
        field = np.full((4, 5), 0.5)
        compliance, energy = solver.solve(field, return_energy=True)
        step = 1e-6
        for row, col in ((0, 0), (2, 3), (3, 4)):
            bumped = field.copy()
            bumped[row, col] += step
            numeric = (solver.solve(bumped) - compliance) / step
            analytic = -3 * field[row, col] ** 2 * (1 - 1e-9) * energy[row, col]
            self.assertAlmostEqual(numeric / analytic, 1.0, delta=1e-4)

    def test_unsupported_structure_reports_infinite_compliance(self):
        normals = np.zeros((4, 4, 2))
        forces = np.zeros((4, 4, 2))
        forces[3, 0, 1] = -1
        self.assertEqual(FastFEM(normals, forces).solve(np.full((3, 3), 0.5)), float("inf"))

    def test_rejects_mismatched_arrays(self):
        with self.assertRaises(ValueError):
            FastFEM(np.zeros((4, 4, 2)), np.zeros((4, 5, 2)))


if __name__ == "__main__":
    unittest.main()
