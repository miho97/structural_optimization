"""Classical SIMP topology optimization, used as a reference for the agent.

The standard "top88" recipe (Andreassen et al. 2011) with a density filter
and Heaviside projection (``ft = 3`` in that paper's notation):

1. design variables ``x`` in [0, 1],
2. density filter ``x_tilde = H x / sum(H)`` with cone weights of radius ``rmin``,
3. Heaviside projection ``x_phys`` with threshold 0.5 and a steepness ``beta``
   that doubles on convergence (1, 2, 4, ... ``beta_max``),
4. optimality-criteria update with the volume constraint on ``x_phys``.

Physics come from ``utils.fem_fast.FastFEM``, the same solver that scores
the agent, with the same SIMP penalty (3) and minimum modulus (1e-9).

The continuous result is not comparable to the agent's black-and-white
designs, so ``optimize`` also returns the design thresholded to exactly the
environment's solid-cell budget. Score that one with ``env.evaluate_design``.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import scipy.sparse


@lru_cache(maxsize=16)
def density_filter(width, height, rmin):
    """Row-normalized cone filter on a ``(height, width)`` field, row-major."""
    reach = int(np.ceil(rmin)) - 1
    rows, cols, values = [], [], []
    for y in range(height):
        for x in range(width):
            row = y * width + x
            for dy in range(-reach, reach + 1):
                for dx in range(-reach, reach + 1):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < height and 0 <= xx < width:
                        weight = rmin - np.hypot(dx, dy)
                        if weight > 0:
                            rows.append(row)
                            cols.append(yy * width + xx)
                            values.append(weight)
    matrix = scipy.sparse.csr_matrix((values, (rows, cols)), shape=(width * height,) * 2)
    return matrix, np.asarray(matrix.sum(axis=1)).ravel()


def _project(x_tilde, beta, eta=0.5):
    denominator = np.tanh(beta * eta) + np.tanh(beta * (1 - eta))
    return (np.tanh(beta * eta) + np.tanh(beta * (x_tilde - eta))) / denominator


def _project_derivative(x_tilde, beta, eta=0.5):
    denominator = np.tanh(beta * eta) + np.tanh(beta * (1 - eta))
    return beta * (1 - np.tanh(beta * (x_tilde - eta)) ** 2) / denominator


def optimize(
    solver,
    solid_budget,
    rmin=2.0,
    beta_max=32.0,
    max_iterations=400,
    iterations_per_beta=50,
    tolerance=0.01,
    move=0.2,
):
    """Run SIMP on ``solver`` (a ``FastFEM``) and return a result dict.

    ``solid_budget`` is the number of solid cells of the final binary design;
    the continuous volume fraction is ``solid_budget / cells``.
    """
    width, height = solver.width, solver.height
    cells = width * height
    volume_fraction = solid_budget / cells
    filter_matrix, filter_sums = density_filter(width, height, float(rmin))
    penal, young, young_min = solver.penal, solver.young, solver.young_min

    def physical(x, beta):
        x_tilde = filter_matrix @ x / filter_sums
        return x_tilde, _project(x_tilde, beta)

    def binary(x_phys):
        order = np.argsort(-x_phys, kind="stable")
        solid = np.zeros(cells, dtype=bool)
        solid[order[:solid_budget]] = True
        density = np.where(solid, 1.0, 0.0).reshape(height, width)
        return np.flatnonzero(solid), solver.solve(density)

    x = np.full(cells, volume_fraction)
    beta = 1.0
    iterations_at_beta = 0
    history = []
    best_cells, best_binary, best_iteration = None, np.inf, 0
    for iteration in range(1, max_iterations + 1):
        x_tilde, x_phys = physical(x, beta)
        compliance, energy = solver.solve(x_phys.reshape(height, width), return_energy=True)
        energy = energy.reshape(-1)
        history.append(compliance)
        # Keep the best black-and-white design seen so far (one extra solve).
        cells_now, binary_now = binary(x_phys)
        if binary_now < best_binary:
            best_cells, best_binary, best_iteration = cells_now, binary_now, iteration

        # Chain rule: compliance -> x_phys -> x_tilde -> x.
        dc_phys = -penal * x_phys ** (penal - 1) * (young - young_min) * energy
        dproj = _project_derivative(x_tilde, beta)
        dc = filter_matrix.T @ (dc_phys * dproj / filter_sums)
        dv = filter_matrix.T @ (dproj / filter_sums) / cells
        dc = np.minimum(dc, -1e-30)  # compliance never decreases with less material

        low, high = 0.0, 1e9
        while (high - low) / (high + low + 1e-30) > 1e-4:
            middle = 0.5 * (low + high)
            candidate = np.clip(
                x * np.sqrt(-dc / (dv * middle)), np.maximum(0.0, x - move), np.minimum(1.0, x + move)
            )
            if physical(candidate, beta)[1].mean() > volume_fraction:
                low = middle
            else:
                high = middle
        change = float(np.abs(candidate - x).max())
        x = candidate
        iterations_at_beta += 1

        if beta < beta_max and (change < tolerance or iterations_at_beta >= iterations_per_beta):
            beta *= 2
            iterations_at_beta = 0
        elif beta >= beta_max and change < tolerance:
            break

    _, x_phys = physical(x, beta)
    gray_compliance = solver.solve(x_phys.reshape(height, width))
    final_cells, final_binary = binary(x_phys)
    if final_binary < best_binary:
        best_cells, best_binary, best_iteration = final_cells, final_binary, iteration
    return {
        "solid_cells": np.sort(best_cells),
        "best_iteration": best_iteration,
        "final_binary_compliance": float(final_binary),
        "x_phys": x_phys.reshape(height, width),
        "gray_compliance": float(gray_compliance),
        "grayness": float(np.mean(4 * x_phys * (1 - x_phys))),
        "iterations": iteration,
        "beta": beta,
        "rmin": float(rmin),
        "history": np.asarray(history),
    }


def best_of_radii(env, radii=(1.5, 2.0, 3.0), **kwargs):
    """Run ``optimize`` for each filter radius; keep the stiffest binary design.

    Returns ``(best, all_runs)``; each run dict gains ``binary`` scores from
    ``env.evaluate_design`` and the ``density`` field it scored.
    """
    runs = []
    for rmin in radii:
        result = optimize(env.solver, env.solid_cell_budget, rmin=rmin, **kwargs)
        density = np.full(env.total_cells, env.minimum_density)
        density[result["solid_cells"]] = env.maximum_density
        density = density.reshape(env.height, env.width)
        result["density"] = density
        result["binary"] = env.evaluate_design(density)
        runs.append(result)
    best = min(runs, key=lambda run: run["binary"]["compliance"])
    return best, runs
