"""Fast compliance solver for the reinforcement-learning environment.

``utils.fem`` stays the reference implementation: it supports automatic
differentiation and drives the MMA optimizer. This module solves the same
linear system for the one thing the environment needs on every step, the
compliance of a given density field, and does it several times faster:

* the assembly operator is built once per grid size and reused,
* the stiffness matrix is symmetric, positive definite and banded, so it is
  factorized with a banded Cholesky instead of a general sparse LU,
* compliance is taken as ``f . u``.

Conventions match ``utils.fem``: nodal arrays are ``(width + 1, height + 1,
2)``, density fields are ``(height, width)``, and a non-zero entry in
``normals`` fixes that degree of freedom.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import scipy.linalg
import scipy.sparse


def element_stiffness(young=1.0, poisson=0.3):
    """Stiffness matrix of a unit square four-node plane-stress element."""
    nu = poisson
    k = np.array(
        [
            1 / 2 - nu / 6,
            1 / 8 + nu / 8,
            -1 / 4 - nu / 12,
            -1 / 8 + 3 * nu / 8,
            -1 / 4 + nu / 12,
            -1 / 8 - nu / 8,
            nu / 6,
            1 / 8 - 3 * nu / 8,
        ]
    )
    return young / (1 - nu**2) * np.array(
        [
            [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
            [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
            [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
            [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
            [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
            [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
            [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
            [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]],
        ]
    )


class _GridTopology:
    """Everything that depends only on the grid size and material constants."""

    def __init__(self, width, height, young, poisson):
        self.width = width
        self.height = height
        self.ndof = 2 * (width + 1) * (height + 1)
        self.ke = element_stiffness(young, poisson)

        # Elements are ordered column by column (x-major), matching the
        # ``stiffness.T.reshape`` ordering in ``utils.fem``.
        elx, ely = np.meshgrid(np.arange(width), np.arange(height), indexing="ij")
        elx = elx.ravel()
        ely = ely.ravel()
        n1 = (height + 1) * elx + ely
        n2 = (height + 1) * (elx + 1) + ely
        n3 = (height + 1) * (elx + 1) + ely + 1
        n4 = (height + 1) * elx + ely + 1
        self.edof = np.stack(
            [2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1, 2 * n3, 2 * n3 + 1, 2 * n4, 2 * n4 + 1],
            axis=1,
        )

        # Lower-triangular band storage: entry (i, j) with i >= j lives at
        # ab[i - j, j]. One sparse product maps element moduli to that band.
        num_elements = width * height
        rows = np.repeat(self.edof, 8, axis=1).ravel()
        cols = np.tile(self.edof, (1, 8)).ravel()
        elements = np.repeat(np.arange(num_elements), 64)
        values = np.tile(self.ke.ravel(), num_elements)
        lower = rows >= cols
        rows, cols, elements, values = rows[lower], cols[lower], elements[lower], values[lower]
        self.bandwidth = int((rows - cols).max())
        flat = (rows - cols) * self.ndof + cols
        self.assembly = scipy.sparse.csr_matrix(
            (values, (flat, elements)),
            shape=((self.bandwidth + 1) * self.ndof, num_elements),
        )


@lru_cache(maxsize=32)
def _grid_topology(width, height, young, poisson):
    return _GridTopology(width, height, young, poisson)


class FastFEM:
    """Compliance solver for one support/load problem on a fixed grid."""

    def __init__(self, normals, forces, penal=3.0, young=1.0, young_min=1e-9, poisson=0.3):
        normals = np.asarray(normals)
        forces = np.asarray(forces, dtype=np.float64)
        if normals.ndim != 3 or normals.shape[2] != 2 or normals.shape != forces.shape:
            raise ValueError("normals and forces must both have shape (width + 1, height + 1, 2)")
        self.width = normals.shape[0] - 1
        self.height = normals.shape[1] - 1
        if self.width < 1 or self.height < 1:
            raise ValueError("the grid needs at least one element in each direction")
        self.penal = float(penal)
        self.young = float(young)
        self.young_min = float(young_min)
        self.topology = _grid_topology(self.width, self.height, float(young), float(poisson))

        fixed = normals.reshape(-1) != 0
        free = ~fixed
        ndof = self.topology.ndof
        bandwidth = self.topology.bandwidth
        # A band entry survives only if both of its degrees of freedom are
        # free. Fixed degrees of freedom get a unit diagonal and zero load,
        # which pins their displacement to zero.
        keep = np.zeros((bandwidth + 1, ndof))
        for offset in range(bandwidth + 1):
            keep[offset, : ndof - offset] = free[: ndof - offset] & free[offset:]
        self._keep = keep
        self._fixed = np.flatnonzero(fixed)
        self._load = np.where(free, forces.reshape(-1), 0.0)

    def solve(self, density, return_energy=False):
        """Return compliance, and optionally the element strain energy.

        ``density`` has shape ``(height, width)``. The strain energy is
        ``u_e^T k_e u_e`` per element at unit modulus, shape ``(height,
        width)``. Its negative is the sensitivity of compliance to that
        element's modulus. A singular system returns ``inf`` compliance.
        """
        topology = self.topology
        density = np.asarray(density, dtype=np.float64)
        if density.shape != (self.height, self.width):
            density = density.reshape(self.height, self.width)
        modulus = self.young_min + density.T.ravel() ** self.penal * (self.young - self.young_min)
        band = (topology.assembly @ modulus).reshape(topology.bandwidth + 1, topology.ndof)
        band *= self._keep
        band[0, self._fixed] = 1.0
        try:
            displacement = scipy.linalg.solveh_banded(
                band, self._load, lower=True, overwrite_ab=True, check_finite=False
            )
        except np.linalg.LinAlgError:
            displacement = None
        if displacement is None or not np.all(np.isfinite(displacement)):
            if return_energy:
                return float("inf"), np.zeros((self.height, self.width))
            return float("inf")

        compliance = float(self._load @ displacement)
        if not return_energy:
            return compliance
        element_displacement = displacement[topology.edof]
        energy = np.einsum(
            "ei,ei->e", element_displacement @ topology.ke, element_displacement
        )
        return compliance, energy.reshape(self.width, self.height).T
