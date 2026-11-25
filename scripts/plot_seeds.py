"""Plot two compare runs side by side: SIMP | agent seed 0 | agent seed 1, one problem per row.

usage: plot_seeds.py SEED0_DIR SEED1_DIR OUT.png beam|random [LIMIT] [TITLE]
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

IN0, IN, OUT, WHICH = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]  # seed0 dir, seed1 dir
LIMIT = int(sys.argv[5]) if len(sys.argv) > 5 else 99
TITLE = sys.argv[6] if len(sys.argv) > 6 else ""
metas = [json.loads(p.read_text()) for p in IN.glob(f"{WHICH}*.json")]
metas = sorted(metas, key=lambda m: int(m["name"].split("_")[-1]))[:LIMIT]
W, H, D = metas[0]["width"], metas[0]["height"], metas[0]["density"]
aspect = W / H
fig, axes = plt.subplots(len(metas), 3, figsize=(3 * 4.6, len(metas) * (4.6 / aspect + 0.55)), squeeze=False)


def draw(ax, grid, normals, forces, title, color="black"):
    ax.imshow(grid, cmap="Greys", interpolation="nearest", vmin=0, vmax=1)
    sx, sy = np.where(normals.any(axis=2))
    ax.scatter(sx - 0.5, sy - 0.5, marker="^", s=12, c="tab:blue", zorder=3, clip_on=False, linewidths=0)
    lx, ly = np.where(np.linalg.norm(forces, axis=2) > 1e-12)
    if len(lx):
        v = forces[lx, ly] / np.linalg.norm(forces[lx, ly], axis=1, keepdims=True)
        if len(lx) > 3:
            keep = np.arange(0, len(lx), max(1, len(lx) // 12))
            lx, ly, v = lx[keep], ly[keep], v[keep]
        L = max(W, H) / 12
        ax.quiver(lx - 0.5 - L * v[:, 0], ly - 0.5 + L * v[:, 1], v[:, 0], -v[:, 1], color="tab:red",
                  angles="xy", scale_units="xy", scale=1 / L, width=0.008, zorder=4)
    pad = max(W, H) / 10 + 1
    ax.set_xlim(-pad, W + pad - 1)
    ax.set_ylim(H + pad - 1, -pad)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=9, color=color)


for row, m in enumerate(metas):
    data = np.load(IN / f"{m['name']}.npz")
    s = m["simp"]
    draw(axes[row, 0], data["simp"], data["normals"], data["forces"],
         f"{m['name']}   SIMP (filter r={s['rmin']}): C={s['compliance']:.2f}, {int(s['contacts'])} corner contacts")
    m0 = json.loads((IN0 / f"{m['name']}.json").read_text())
    d0 = np.load(IN0 / f"{m['name']}.npz")
    for col, key, label in ((1, "final", "Agent seed 0"), (2, "final", "Agent seed 1")):
        src_m, src_d = (m0, d0) if col == 1 else (m, data)
        c = src_m[key]["compliance"]
        failed = c > 10 * s["compliance"]
        text = (f"{label}: DISCONNECTED" if failed else
                f"{label}: C={c:.2f} ({c / s['compliance'] - 1:+.1%}), {int(src_m[key]['contacts'])} contacts")
        draw(axes[row, col], src_d[key], data["normals"], data["forces"], text,
             "tab:red" if failed else ("tab:green" if c < s["compliance"] else "black"))

fig.suptitle(
    f"{TITLE}{W}x{H}, {D:.0%} material  |  columns: SIMP (density filter + projection, best of r=1.5/2/3) | "
    "agent seed 0 | agent seed 1 (both trained only on 12x6, no retraining)\nall designs have exactly the same number of solid cells and are scored "
    "with the same solver  |  % = compliance vs SIMP (lower is stiffer)",
    fontsize=11,
)
fig.tight_layout(rect=(0, 0, 1, 0.975 if len(metas) > 6 else 0.95))
fig.savefig(OUT, dpi=130)
print("saved", OUT)
