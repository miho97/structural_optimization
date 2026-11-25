"""Figures for the README, in the viridis (purple = void, yellow = solid) style.

usage: plot_readme.py CHECKPOINT OUTDIR

1. ``zeroshot_40x20.png``: the 12x6 checkpoint run on the nine fixed beams at
   40x20 (never trained on that size), each title giving the gap to SIMP.
2. ``zeroshot_80x40_vs_simp.png``: SIMP vs agent at 80x40 for three beams,
   drawn from ``benchmarks/compare_80x40_seed0`` (run ``infer_compare.py`` first).

Run from the project folder.
"""
import contextlib
import io
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path.cwd()
CHECKPOINT, OUTDIR = sys.argv[1], Path(sys.argv[2])
sys.path.insert(0, str(ROOT / "src"))
import benchmark  # noqa: E402
from utils import simp  # noqa: E402

OUTDIR.mkdir(parents=True, exist_ok=True)


def draw(ax, grid, normals, forces, title):
    height, width = grid.shape
    image = ax.imshow(grid, cmap="viridis", interpolation="nearest", vmin=0, vmax=1)
    sx, sy = np.where(normals.any(axis=2))
    ax.scatter(sx - 0.5, sy - 0.5, marker="^", s=10, c="tab:red", zorder=3, clip_on=False, linewidths=0)
    lx, ly = np.where(np.linalg.norm(forces, axis=2) > 1e-12)
    if len(lx):
        v = forces[lx, ly] / np.linalg.norm(forces[lx, ly], axis=1, keepdims=True)
        if len(lx) > 3:
            keep = np.arange(0, len(lx), max(1, len(lx) // 10))
            lx, ly, v = lx[keep], ly[keep], v[keep]
        length = max(width, height) / 10
        # Force y is physical (negative = down); rows grow downward, so flip y to draw it.
        ax.quiver(lx - 0.5 - length * v[:, 0], ly - 0.5 + length * v[:, 1], v[:, 0], -v[:, 1],
                  color="tab:red", angles="xy", scale_units="xy", scale=1 / length, width=0.012, zorder=4)
    pad = max(width, height) / 9 + 1
    ax.set_xlim(-pad, width + pad - 1)
    ax.set_ylim(height + pad - 1, -pad)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=10)
    return image


# 1. 40x20, nine fixed beams, agent only, gap to SIMP in the title.
width, height, density = 40, 20, 0.45
agent = benchmark.load_agent(CHECKPOINT, width, height)
with contextlib.redirect_stdout(io.StringIO()):
    problems = benchmark.build_problems(width, height, density)
fig, axes = plt.subplots(3, 3, figsize=(15, 6.6))
for ax, (name, env) in zip(axes.flat, problems):
    reference, _ = simp.best_of_radii(env)
    result = benchmark.run_policy(agent, env)
    gap = result["final_compliance"] / reference["binary"]["compliance"] - 1
    label = "DISCONNECTED (fails)" if gap > 9 else f"{gap:+.1%} compliance vs SIMP"
    image = draw(ax, result["final_density"], env.normals, env.forces, f"{name.replace('_', ' ')}: {label}")
    print(f"{name}: agent C={result['final_compliance']:.2f}, SIMP C={reference['binary']['compliance']:.2f}, {gap:+.1%}")
fig.colorbar(image, ax=axes, fraction=0.015, pad=0.01, label="density")
fig.suptitle("Agent trained only on 12x6 grids, run on 40x20 without retraining (45% material)", fontsize=13)
fig.savefig(OUTDIR / "zeroshot_40x20.png", dpi=110, bbox_inches="tight")
plt.close(fig)

# 2. 80x40: SIMP vs agent for three beams, from saved comparison results.
compare = ROOT / "benchmarks" / "compare_80x40_seed0"
beams = ["beam_1", "beam_4", "beam_9"]
fig, axes = plt.subplots(len(beams), 2, figsize=(12, 3.3 * len(beams)))
for row, name in enumerate(beams):
    meta = json.loads((compare / f"{name}.json").read_text())
    data = np.load(compare / f"{name}.npz")
    reference, agent_c = meta["simp"]["compliance"], meta["final"]["compliance"]
    draw(axes[row, 0], data["simp"], data["normals"], data["forces"],
         f"{name.replace('_', ' ')}  SIMP: C = {reference:.1f}")
    image = draw(axes[row, 1], data["final"], data["normals"], data["forces"],
                 f"Agent (12x6 model): C = {agent_c:.1f} ({agent_c / reference - 1:+.1%})")
fig.colorbar(image, ax=axes, fraction=0.02, pad=0.01, label="density")
fig.suptitle("80x40, 45% material: SIMP (density filter + projection) vs the 12x6 agent", fontsize=13)
fig.savefig(OUTDIR / "zeroshot_80x40_vs_simp.png", dpi=110, bbox_inches="tight")
print("saved", OUTDIR / "zeroshot_40x20.png", OUTDIR / "zeroshot_80x40_vs_simp.png")
