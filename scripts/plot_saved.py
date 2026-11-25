"""Plot saved inference results (infer_large.py output): reference | best | final per beam."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

IN = Path(sys.argv[1])
OUT = Path(sys.argv[2])
metas = sorted((json.loads(p.read_text()) for p in IN.glob("beam_*.json")), key=lambda m: int(m["name"].split("_")[1]))
W, H, D = metas[0]["width"], metas[0]["height"], metas[0]["density"]
aspect = W / H
fig, axes = plt.subplots(len(metas), 3, figsize=(3 * 4.6, len(metas) * (4.6 / aspect + 0.55)), squeeze=False)


def draw(ax, grid, normals, forces, title, failed=False):
    ax.imshow(grid, cmap="Greys", interpolation="nearest", vmin=0, vmax=1)
    sx, sy = np.where(normals.any(axis=2))
    ax.scatter(sx - 0.5, sy - 0.5, marker="^", s=12, c="tab:blue", zorder=3, clip_on=False, linewidths=0)
    lx, ly = np.where(np.linalg.norm(forces, axis=2) > 1e-12)
    if len(lx):
        v = forces[lx, ly]
        v = v / np.linalg.norm(v, axis=1, keepdims=True)
        L = max(W, H) / 12  # arrow length in cells
        if len(lx) > 3:  # distributed load: draw every few arrows
            keep = np.arange(0, len(lx), max(1, len(lx) // 12))
            lx, ly, v = lx[keep], ly[keep], v[keep]
        # force y is physical (negative = down); rows grow downward, so flip y
        ax.quiver(lx - 0.5 - L * v[:, 0], ly - 0.5 + L * v[:, 1], v[:, 0], -v[:, 1], color="tab:red",
                  angles="xy", scale_units="xy", scale=1 / L, width=0.008, zorder=4)
    ax.set_xlim(-1.5, W + 0.5)
    ax.set_ylim(H + 0.5, -max(W, H) / 10 - 1)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=9, color="tab:red" if failed else "black")


for row, m in enumerate(metas):
    data = np.load(IN / f"{m['name']}.npz")
    ref = m["reference"]
    ref_broken = ref["compliance"] > 1e4
    draw(axes[row, 0], data["ref"], data["normals"], data["forces"],
         f"{m['name']}  Reference ({m['reference_source']}): "
         + ("DISCONNECTED (thresholding broke it)" if ref_broken else f"C={ref['compliance']:.1f}")
         + f", {int(ref['contacts'])} corner contacts", ref_broken)
    for col, key in ((1, "best"), (2, "final")):
        s = m[key]
        failed = s["compliance"] > 10 * ref["compliance"] and not ref_broken
        if failed:
            text = "DISCONNECTED" if s["compliance"] > 1e4 else f"C={s['compliance']:.1f} FAILED"
        elif ref_broken:
            text = f"C={s['compliance']:.1f} (no valid reference), {int(s['contacts'])} corner contacts"
        else:
            text = f"C={s['compliance']:.1f} ({s['compliance'] / ref['compliance'] - 1:+.0%}), {int(s['contacts'])} corner contacts"
        if key == "final":
            text += f"\n{m['steps']} steps, {'stopped itself' if m['stopped'] else 'used every swap'}"
        draw(axes[row, col], data[key], data["normals"], data["forces"], f"{key.capitalize()}: {text}", failed)

fig.suptitle(
    f"12x6-trained model (no retraining) on {W}x{H}, {D:.0%} material  |  columns: reference, agent best, agent final\n"
    "Reference = better of MMA-thresholded and greedy strain-energy design, NO local search (too slow at this size)\n"
    "black = solid, blue = supports, red = loads  |  C = compliance (lower is stiffer), % vs reference",
    fontsize=11,
)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(OUT, dpi=140)
print("saved", OUT)
