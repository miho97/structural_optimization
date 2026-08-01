import sys
import unittest
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from utils import fem  # noqa: E402


class FemCorrectnessTests(unittest.TestCase):
    def test_compliance_matches_external_work_on_rectangular_meshes(self):
        random = np.random.default_rng(42)
        for width, height in ((6, 6), (4, 7), (7, 4)):
            args = fem.get_args(*fem.mbb_beam_1(width, height, 0.4))
            density = random.uniform(0.2, 1.0, size=(height, width))
            stiffness = fem.get_stiffness_matrix(args.young, args.poisson)
            displacement = fem.displace(
                density,
                stiffness,
                args.forces,
                args.freedofs,
                args.fixdofs,
                penal=args.penal,
                e_min=args.young_min,
                e_0=args.young,
            )
            strain_energy = fem.compliance_calc(
                density,
                displacement,
                stiffness,
                penal=args.penal,
                e_min=args.young_min,
                e_0=args.young,
            )
            external_work = np.dot(args.forces, displacement)
            self.assertTrue(
                np.isclose(strain_energy, external_work, rtol=1e-10, atol=1e-10),
                (width, height, strain_energy, external_work),
            )

    def test_autograd_compliance_gradient_survives_cached_topology(self):
        import autograd
        import autograd.numpy as anp

        args = fem.get_args(*fem.mbb_beam_1(3, 2, 0.4))
        density = anp.full((args.nely, args.nelx), 0.4)
        value, gradient = autograd.value_and_grad(
            lambda design: fem.objective_calc(design, args)
        )(density)
        self.assertTrue(np.isfinite(value))
        self.assertEqual(gradient.shape, density.shape)
        self.assertTrue(np.isfinite(gradient).all())

    def test_cached_and_uncached_fem_paths_are_identical(self):
        args = fem.get_args(*fem.mbb_beam_1(5, 3, 0.4))
        density = np.random.default_rng(9).uniform(0.2, 1.0, (3, 5))
        cached_compliance = fem.objective_calc(density, args)
        stiffness = fem.get_stiffness_matrix(args.young, args.poisson)
        displacement = fem.displace(
            density,
            stiffness,
            args.forces,
            args.freedofs,
            args.fixdofs,
            penal=args.penal,
            e_min=args.young_min,
            e_0=args.young,
        )
        uncached_compliance = fem.compliance_calc(
            density,
            displacement,
            stiffness,
            penal=args.penal,
            e_min=args.young_min,
            e_0=args.young,
        )
        self.assertTrue(
            np.isclose(cached_compliance, uncached_compliance, rtol=1e-12, atol=1e-12)
        )


if __name__ == "__main__":
    unittest.main()
