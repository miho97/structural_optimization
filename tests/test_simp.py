import contextlib
import io
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import benchmark  # noqa: E402
from utils import simp  # noqa: E402


def fixed_beam(width, height, beam_type, density=0.45):
    with contextlib.redirect_stdout(io.StringIO()):
        return benchmark.build_problems(width, height, density, fixed_beams=[beam_type])[0][1]


class SimpTest(unittest.TestCase):
    def test_gradient_matches_finite_differences(self):
        env = fixed_beam(12, 6, 1)
        solver = env.solver
        width, height = solver.width, solver.height
        matrix, sums = simp.density_filter(width, height, 2.0)
        rng = np.random.default_rng(0)
        x = rng.uniform(0.2, 0.8, width * height)
        beta = 4.0

        def compliance(values):
            projected = simp._project(matrix @ values / sums, beta)
            return solver.solve(projected.reshape(height, width))

        x_tilde = matrix @ x / sums
        x_phys = simp._project(x_tilde, beta)
        _, energy = solver.solve(x_phys.reshape(height, width), return_energy=True)
        dc_phys = -3 * x_phys**2 * (solver.young - solver.young_min) * energy.reshape(-1)
        gradient = matrix.T @ (dc_phys * simp._project_derivative(x_tilde, beta) / sums)
        for cell in rng.choice(width * height, 5, replace=False):
            step = np.zeros_like(x)
            step[cell] = 1e-6
            numeric = (compliance(x + step) - compliance(x - step)) / 2e-6
            self.assertAlmostEqual(gradient[cell] / numeric, 1.0, places=4)

    def test_binary_design_has_exact_budget_and_matches_reference(self):
        env = fixed_beam(40, 20, 4)
        best, runs = simp.best_of_radii(env, radii=(1.5,))
        self.assertEqual(len(best["solid_cells"]), env.solid_cell_budget)
        solid = best["density"] >= env.maximum_density - 1e-9
        self.assertEqual(int(solid.sum()), env.solid_cell_budget)
        # Classical MMA + local search reaches 24.44 on this beam.
        self.assertLess(best["binary"]["compliance"], 24.44 * 1.05)


if __name__ == "__main__":
    unittest.main()
