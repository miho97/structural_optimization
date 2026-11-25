"""Random support/load problems for training and benchmarking.

A domain is a pair of nodal arrays ``(normals, forces)`` with shape
``(width + 1, height + 1, 2)`` in the FEM convention ``[x, y, dof]``.

Two generators live here:

* ``legacy_random_beam`` is the original three-template generator
  (cantilever, simply supported, corner supported) with one point load.
* ``sample_rich_domain`` ports the sampler prototyped in
  ``random_domain_comparison.ipynb``: clamped edge segments, two pins, or a
  pin plus a roller, with single, multiple or distributed loads in 16
  directions. Two things differ from the notebook. Supports are sampled
  directly instead of from an enumerated catalogue, so large grids stay
  cheap. A fourth support family, a roller edge plus one support, is added
  because every built-in benchmark beam uses a roller (symmetry) edge.

``sample_domain`` mixes the two.
"""

from __future__ import annotations

import numpy as np

MINIMUM_DENSITY = 0.001
MAXIMUM_DENSITY = 1.0
ANGLE_BINS = 16
_ANGLES = np.linspace(0.0, 2.0 * np.pi, ANGLE_BINS, endpoint=False)
DIRECTIONS = np.column_stack((np.cos(_ANGLES), np.sin(_ANGLES)))

SUPPORT_FAMILIES = ("clamped segment", "two pins", "pin + roller", "roller edge + support")
LOAD_FAMILIES = ("single point", "multiple points", "distributed segment")
LOAD_FAMILY_PROBABILITIES = (0.40, 0.30, 0.30)


def solid_cell_budget(width, height, density):
    """Nearest solid-cell count that realizes the target volume fraction."""
    total_cells = width * height
    continuous = (density * total_cells - MINIMUM_DENSITY * total_cells) / (
        MAXIMUM_DENSITY - MINIMUM_DENSITY
    )
    return int(np.clip(np.rint(continuous), 0, total_cells))


def legacy_random_beam(width, height, density=0.4, rng=None):
    """Original generator: one of three templates with a single point load.

    Returns ``(normals, forces, density)``. Pass a NumPy ``Generator`` to make
    the result reproducible without touching global random state.
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


def rigid_mode_rank(normals):
    """Rank of the constrained planar rigid-body modes. Rank 3 is stable."""
    normals = np.asarray(normals)
    width = normals.shape[0] - 1
    height = normals.shape[1] - 1
    x_coords, y_coords = np.meshgrid(
        np.arange(width + 1), np.arange(height + 1), indexing="ij"
    )
    modes = np.zeros((width + 1, height + 1, 2, 3), dtype=float)
    modes[:, :, 0, 0] = 1.0                        # X translation
    modes[:, :, 1, 1] = 1.0                        # Y translation
    modes[:, :, 0, 2] = -(y_coords - height / 2)   # rotation: ux
    modes[:, :, 1, 2] = x_coords - width / 2       # rotation: uy
    constrained = normals.astype(bool).reshape(-1)
    if not constrained.any():
        return 0
    return int(np.linalg.matrix_rank(modes.reshape(-1, 3)[constrained]))


def _edge_nodes(width, height):
    return {
        "bottom": [(x, 0) for x in range(width + 1)],
        "top": [(x, height) for x in range(width + 1)],
        "left": [(0, y) for y in range(height + 1)],
        "right": [(width, y) for y in range(height + 1)],
    }


def _boundary_nodes(width, height):
    return [
        (x, y)
        for x in range(width + 1)
        for y in range(height + 1)
        if x in (0, width) or y in (0, height)
    ]


def _sample_supports(width, height, rng):
    """Draw one support map. The caller checks rigid-body stability."""
    edges = _edge_nodes(width, height)
    boundary = _boundary_nodes(width, height)
    normals = np.zeros((width + 1, height + 1, 2), dtype=np.float64)
    family = SUPPORT_FAMILIES[int(rng.integers(len(SUPPORT_FAMILIES)))]

    if family == "clamped segment":
        nodes = edges[("bottom", "top", "left", "right")[int(rng.integers(4))]]
        length = int(rng.integers(2, len(nodes) + 1))
        start = int(rng.integers(0, len(nodes) - length + 1))
        for x, y in nodes[start : start + length]:
            normals[x, y, :] = 1

    elif family == "two pins":
        first, second = rng.choice(len(boundary), size=2, replace=False)
        for index in (first, second):
            x, y = boundary[int(index)]
            normals[x, y, :] = 1

    elif family == "pin + roller":
        pin, roller = rng.choice(len(boundary), size=2, replace=False)
        x, y = boundary[int(pin)]
        normals[x, y, :] = 1
        x, y = boundary[int(roller)]
        normals[x, y, int(rng.integers(2))] = 1

    else:  # roller edge + support
        edge = ("bottom", "top", "left", "right")[int(rng.integers(4))]
        # A roller edge fixes the displacement normal to that edge.
        normal_dof = 1 if edge in ("bottom", "top") else 0
        for x, y in edges[edge]:
            normals[x, y, normal_dof] = 1
        x, y = boundary[int(rng.integers(len(boundary)))]
        normals[x, y, 1 - normal_dof] = 1

    return normals, family


def _adjacent_elements(node, width, height):
    x, y = node
    return {
        (row, col)
        for row in (y - 1, y)
        for col in (x - 1, x)
        if 0 <= row < height and 0 <= col < width
    }


def _axis_path(start, finish, column_first):
    row, col = start
    target_row, target_col = finish
    path = {(row, col)}
    for axis in ("col", "row") if column_first else ("row", "col"):
        if axis == "col":
            while col != target_col:
                col += 1 if target_col > col else -1
                path.add((row, col))
        else:
            while row != target_row:
                row += 1 if target_row > row else -1
                path.add((row, col))
    return path


def material_path_size(normals, forces):
    """Cells needed to join every loaded node to a support along grid paths.

    This is a conservative feasibility test: a domain whose loads cannot be
    connected to a support within the solid-cell budget is rejected.
    """
    width = normals.shape[0] - 1
    height = normals.shape[1] - 1
    support_xy = np.argwhere(normals.any(axis=2))
    load_xy = np.argwhere(np.linalg.norm(forces, axis=2) > 1e-12)
    support_cells = set()
    for x, y in support_xy:
        support_cells |= _adjacent_elements((int(x), int(y)), width, height)
    if not support_cells:
        return width * height + 1
    support_array = np.array(sorted(support_cells))
    used_cells = set()
    for x, y in load_xy:
        load_cells = _adjacent_elements((int(x), int(y)), width, height)
        if not load_cells:
            return width * height + 1
        best = None
        for cell in load_cells:
            distances = np.abs(support_array - np.array(cell)).sum(axis=1)
            index = int(np.argmin(distances))
            candidate = (int(distances[index]), cell, tuple(support_array[index]))
            if best is None or candidate[0] < best[0]:
                best = candidate
        _, start, finish = best
        paths = [_axis_path(start, finish, False), _axis_path(start, finish, True)]
        used_cells |= min(paths, key=lambda path: len(path - used_cells))
    return len(used_cells)


def _free_direction(normals, node, rng):
    """A unit load direction restricted to the node's free degrees of freedom."""
    direction = DIRECTIONS[int(rng.integers(ANGLE_BINS))].copy()
    direction[normals[node[0], node[1]].astype(bool)] = 0.0
    norm = np.linalg.norm(direction)
    if norm < 1e-9:
        return None
    return direction / norm


def sample_rich_domain(width, height, solid_budget, rng=None, max_attempts=200):
    """Sample a mechanically stable domain with a feasible material budget.

    Returns ``(normals, forces, meta)``. The total applied load magnitude is
    normalized to one.
    """
    if width < 1 or height < 1:
        raise ValueError("width and height must both be positive")
    rng = np.random.default_rng() if rng is None else rng
    edges = _edge_nodes(width, height)
    boundary = _boundary_nodes(width, height)

    for attempt in range(1, max_attempts + 1):
        normals, support_family = _sample_supports(width, height, rng)
        if rigid_mode_rank(normals) != 3:
            continue
        loadable = [node for node in boundary if not normals[node[0], node[1]].all()]
        if not loadable:
            continue

        load_family = LOAD_FAMILIES[
            int(rng.choice(len(LOAD_FAMILIES), p=LOAD_FAMILY_PROBABILITIES))
        ]
        forces = np.zeros_like(normals)

        if load_family == "single point":
            node = loadable[int(rng.integers(len(loadable)))]
            direction = _free_direction(normals, node, rng)
            if direction is None:
                continue
            forces[node[0], node[1]] = direction

        elif load_family == "multiple points":
            if len(loadable) < 2:
                continue
            count = int(rng.integers(2, min(3, len(loadable)) + 1))
            chosen = rng.choice(len(loadable), size=count, replace=False)
            valid = True
            for index in np.atleast_1d(chosen):
                node = loadable[int(index)]
                direction = _free_direction(normals, node, rng)
                if direction is None:
                    valid = False
                    break
                forces[node[0], node[1]] = direction
            if not valid:
                continue

        else:  # distributed segment
            nodes = edges[("bottom", "top", "left", "right")[int(rng.integers(4))]]
            upper = min(6, len(nodes))
            length = int(rng.integers(2, upper + 1))
            start = int(rng.integers(0, len(nodes) - length + 1))
            segment = nodes[start : start + length]
            direction = DIRECTIONS[int(rng.integers(ANGLE_BINS))]
            valid = True
            for x, y in segment:
                component = direction.copy()
                component[normals[x, y].astype(bool)] = 0.0
                if np.linalg.norm(component) < 1e-9:
                    valid = False
                    break
                forces[x, y] = component
            if not valid:
                continue

        total = np.linalg.norm(forces, axis=2).sum()
        if not np.isfinite(total) or total <= 1e-12:
            continue
        forces = forces / total
        if material_path_size(normals, forces) > solid_budget:
            continue
        meta = {
            "support_family": support_family,
            "load_family": load_family,
            "attempts": attempt,
        }
        return normals, forces, meta

    raise RuntimeError(f"no valid domain found after {max_attempts} attempts")


def sample_domain(width, height, density, rng=None, kind="mixed", legacy_fraction=0.25):
    """Sample a training domain.

    ``kind`` is ``"legacy"``, ``"rich"`` or ``"mixed"``. Mixed draws the
    legacy templates with probability ``legacy_fraction`` and the rich
    sampler otherwise. Returns ``(normals, forces, meta)``.
    """
    rng = np.random.default_rng() if rng is None else rng
    if kind not in ("legacy", "rich", "mixed"):
        raise ValueError("kind must be 'legacy', 'rich' or 'mixed'")
    use_legacy = kind == "legacy" or (kind == "mixed" and rng.random() < legacy_fraction)
    if not use_legacy:
        budget = solid_cell_budget(width, height, density)
        try:
            return sample_rich_domain(width, height, budget, rng)
        except RuntimeError:
            # Tiny grids or budgets can leave the rich sampler without a
            # feasible draw. The templates always produce a valid problem.
            pass
    normals, forces, _ = legacy_random_beam(width, height, density, rng)
    return normals, forces, {"support_family": "legacy template", "load_family": "single point"}
