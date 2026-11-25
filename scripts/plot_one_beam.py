"""Large view of one fixed beam: reference, agent best, agent final."""
import contextlib
import io
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(sys.argv[1])
CHECKPOINT = ROOT / sys.argv[2]
OUT = ROOT / sys.argv[3]
BEAM = int(sys.argv[4])
WIDTH, HEIGHT = int(os.environ["WIDTH"]), int(os.environ["HEIGHT"])
DENSITY = float(os.environ["DENSITY"])
sys.path.insert(0, str(ROOT / "src"))
import benchmark  # noqa: E402

agent = benchmark.load_agent(str(CHECKPOINT), WIDTH, HEIGHT)
with contextlib.redirect_stdout(io.StringIO()):
    (name, env), = benchmark.build_problems(WIDTH, HEIGHT, DENSITY, fixed_beams=[BEAM])
ref = (benchmark.compute_reference(env) if os.environ.get("CLASSICAL") else benchmark.ReferenceCache().get(env, name))
result = benchmark.run_policy(agent, env)
ref_grid = benchmark.design_from_solid(env, ref["solid_cells"])


def gap(obj):
    return 10 ** (obj - ref["objective"]) - 1


panels = [
    ("Reference (MMA / greedy + local search)", ref_grid, ref["compliance"], None, ref["contacts"]),
    ("Agent: best design during the episode", result["best_density"], result["best_compliance"],
     gap(result["best_objective"]), result["best_contacts"]),
    (f"Agent: final design (after {result['steps']} steps, "
     f"{'stopped itself' if result['stopped'] else 'used every swap'})",
     result["final_density"], result["final_compliance"], gap(result["final_objective"]), result["final_contacts"]),
]

aspect = WIDTH / HEIGHT
fig, axes = plt.subplots(len(panels), 1, figsize=(9, len(panels) * (9 / aspect + 0.9)))
for ax, (title, grid, c, g, k) in zip(axes, panels):
    ax.imshow(grid, cmap="Greys", interpolation="nearest", vmin=0, vmax=1)
    # cell grid lines
    for x in np.arange(-0.5, WIDTH, 1):
        ax.axvline(x, color="0.85", lw=0.6, zorder=1)
    for y in np.arange(-0.5, HEIGHT, 1):
        ax.axhline(y, color="0.85", lw=0.6, zorder=1)
    sx, sy = np.where(env.normals.any(axis=2))
    kinds = env.normals[sx, sy]
    for (x, y), (nx, ny) in zip(zip(sx, sy), kinds):
        marker = "s" if (nx and ny) else ("^" if ny else ">")
        ax.scatter(x - 0.5, y - 0.5, marker=marker, s=90, c="tab:blue", zorder=3, clip_on=False)
    lx, ly = np.where(np.linalg.norm(env.forces, axis=2) > 1e-12)
    v = env.forces[lx, ly]
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    # Arrow ends at the loaded node; force y is physical (negative = down), rows grow downward.
    ax.quiver(lx - 0.5 - 0.9 * v[:, 0], ly - 0.5 - 0.9 * (-v[:, 1]), v[:, 0], -v[:, 1], color="tab:red", angles="xy",
              scale_units="xy", scale=1 / 0.9, width=0.012, zorder=4)
    ax.set_xlim(-1.2, WIDTH + 0.2)
    ax.set_ylim(HEIGHT + 0.2, -1.6)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    extra = "" if g is None else f"  ({c / ref['compliance'] - 1:+.0%} vs reference)"
    score = "" if g is None else f"  |  score incl. contact cost {g:+.0%}"
    ax.set_title(f"{title}\ncompliance C = {c:.1f}{extra}\n{k} corner-only contacts{score}", fontsize=11)

fig.suptitle(f"{name} on a {WIDTH}x{HEIGHT} grid, {DENSITY:.0%} material\n"
             "black = solid  |  blue: ▲ blocks vertical, ▶ blocks horizontal, ■ both  |  red = load",
             fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(OUT, dpi=130)
print(name, "ref", ref["compliance"], "best", result["best_compliance"], "final", result["final_compliance"])
print("supports (x, y, normals):", [(int(x), int(y), tuple(n)) for x, y, n in zip(sx, sy, kinds)])
print("load nodes:", list(zip(lx.tolist(), ly.tolist())), "dirs", env.forces[lx, ly].tolist())
print("saved", OUT)
