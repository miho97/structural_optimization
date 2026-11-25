"""Plot reference vs agent (best, final) designs on the nine fixed beams."""
import contextlib
import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(sys.argv[1])
CHECKPOINT = ROOT / sys.argv[2]
OUT = ROOT / sys.argv[3]
import os
WIDTH, HEIGHT = int(os.environ.get("WIDTH", "6")), int(os.environ.get("HEIGHT", "6"))
DENSITY = float(os.environ.get("DENSITY", "0.4"))
sys.path.insert(0, str(ROOT / "src"))

import benchmark  # noqa: E402

agent = benchmark.load_agent(str(CHECKPOINT), WIDTH, HEIGHT)
with contextlib.redirect_stdout(io.StringIO()):
    problems = benchmark.build_problems(WIDTH, HEIGHT, DENSITY)
cache = benchmark.ReferenceCache()

aspect = WIDTH / HEIGHT
# Wide grids: one beam per row (reference | best | final); square grids: one beam per column.
TRANSPOSE = aspect > 1.5
if TRANSPOSE:
    fig, axes = plt.subplots(len(problems), 3, figsize=(3 * 4.2, len(problems) * (4.2 / aspect + 0.45)), squeeze=False)
else:
    fig, axes = plt.subplots(3, len(problems), figsize=(2.4 * aspect * len(problems), 8.4), squeeze=False)


def ax_for(kind, beam):
    return axes[beam, kind] if TRANSPOSE else axes[kind, beam]
row_labels = ["Reference\n(MMA / greedy + local search)", "Agent: best design\nduring episode", "Agent: final design"]


def draw(ax, env, grid, title, failed=False):
    ax.imshow(grid, cmap="Greys", interpolation="nearest", vmin=0, vmax=1)
    sx, sy = np.where(env.normals.any(axis=2))
    ax.scatter(sx - 0.5, sy - 0.5, marker="^", s=55, c="tab:blue", zorder=3, clip_on=False)
    lx, ly = np.where(np.linalg.norm(env.forces, axis=2) > 1e-12)
    if len(lx):
        v = env.forces[lx, ly]
        v = v / np.linalg.norm(v, axis=1, keepdims=True)
        # Force y is physical (negative = down) but row index grows downward, so flip y to draw it.
        ax.quiver(lx - 0.5, ly - 0.5, v[:, 0], -v[:, 1], color="tab:red", angles="xy",
                  scale_units="xy", scale=1.0, width=0.03, zorder=4)
    ax.set_xlim(-1, env.width)
    ax.set_ylim(env.height, -1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9, color="tab:red" if failed else "black")
    for spine in ax.spines.values():
        spine.set_visible(False)


for col, (name, env) in enumerate(problems):
    ref = (benchmark.compute_reference(env) if os.environ.get("CLASSICAL") else cache.get(env, name))
    ref_obj = ref["objective"]
    result = benchmark.run_policy(agent, env)
    ref_grid = benchmark.design_from_solid(env, ref["solid_cells"])
    best_gap = 10 ** (result["best_objective"] - ref_obj) - 1
    final_gap = 10 ** (result["final_objective"] - ref_obj) - 1
    best_failed = result["best_compliance"] > benchmark.FAILURE_FACTOR * ref["compliance"]
    final_failed = result["final_compliance"] > benchmark.FAILURE_FACTOR * ref["compliance"]

    def label(c, gap, failed, contacts):
        if failed:
            return "DISCONNECTED" if c > 1e4 else f"C={c:.1f}  FAILED ({gap:+.0%})"
        c_gap = c / ref["compliance"] - 1
        return f"C={c:.1f} ({c_gap:+.0%}), {contacts} corner contacts, score {gap:+.0%}"

    prefix = ("Reference: ", "Best: ", "Final: ") if TRANSPOSE else ("", "", "")
    draw(ax_for(0, col), env, ref_grid, f"{name}\n{prefix[0]}C={ref['compliance']:.1f}" if not TRANSPOSE
         else f"{name}  {prefix[0]}C={ref['compliance']:.1f}, {ref['contacts']} corner contacts")
    draw(ax_for(1, col), env, result["best_density"],
         prefix[1] + label(result["best_compliance"], best_gap, best_failed, result["best_contacts"]), best_failed)
    draw(ax_for(2, col), env, result["final_density"],
         prefix[2] + label(result["final_compliance"], final_gap, final_failed, result["final_contacts"]), final_failed)
    print(f"{name}: ref {ref['compliance']:.2f}  best {result['best_compliance']:.2f} ({best_gap:+.1%})  "
          f"final {result['final_compliance']:.4g} ({final_gap:+.1%})  steps {result['steps']} stopped {result['stopped']}")

if not TRANSPOSE:
    for row, text in enumerate(row_labels):
        axes[row, 0].set_ylabel(text, fontsize=10)

fig.suptitle(
    f"PPO agent vs reference on the {len(problems)} fixed beams ({WIDTH}x{HEIGHT}, {DENSITY:.0%} material)"
    + ("\n" if TRANSPOSE else "  |  ")
    + "black = solid, blue = supports, red = loads  |  C = compliance (lower is stiffer), (%) = compliance vs reference\nscore = training objective incl. 0.02 per corner-only contact",
    fontsize=11,
)
fig.tight_layout(rect=(0, 0, 1, 0.97 if TRANSPOSE else 0.95))
fig.savefig(OUT, dpi=130)
print("saved", OUT)
