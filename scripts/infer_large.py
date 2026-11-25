"""Run a checkpoint on fixed beams at a large grid and save designs + references.

usage: infer_large.py ROOT CHECKPOINT OUTDIR WIDTH HEIGHT DENSITY BEAM [BEAM ...]

References are the classical starting designs WITHOUT local search (too slow
at this size): MMA thresholded to the solid budget, and greedy strain-energy
construction. The one with lower compliance is kept.
"""
import contextlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1])
CKPT = ROOT / sys.argv[2]
OUTDIR = Path(sys.argv[3])
W, H, D = int(sys.argv[4]), int(sys.argv[5]), float(sys.argv[6])
BEAMS = [int(b) for b in sys.argv[7:]]
sys.path.insert(0, str(ROOT / "src"))
import benchmark  # noqa: E402

OUTDIR.mkdir(parents=True, exist_ok=True)
agent = benchmark.load_agent(str(CKPT), W, H)
with contextlib.redirect_stdout(io.StringIO()):
    problems = benchmark.build_problems(W, H, D, fixed_beams=BEAMS)

for name, env in problems:
    t0 = time.perf_counter()
    starts = {}
    for label, fn in (("mma", benchmark.mma_start), ("greedy", benchmark.energy_start)):
        cells = fn(env)
        grid = benchmark.design_from_solid(env, cells)
        starts[label] = (grid, env.evaluate_design(grid))
    ref_label = min(starts, key=lambda k: starts[k][1]["compliance"])
    ref_grid, ref_scores = starts[ref_label]
    t_ref = time.perf_counter() - t0

    t0 = time.perf_counter()
    result = benchmark.run_policy(agent, env)
    t_pol = time.perf_counter() - t0
    best_scores = env.evaluate_design(result["best_density"])
    final_scores = env.evaluate_design(result["final_density"])
    np.savez_compressed(
        OUTDIR / f"{name}.npz",
        ref=ref_grid, best=result["best_density"], final=result["final_density"],
        normals=env.normals, forces=env.forces,
    )
    meta = {
        "name": name, "width": W, "height": H, "density": D,
        "reference_source": ref_label,
        "reference": {k: float(v) for k, v in ref_scores.items()},
        "other_start": {k: float(v) for k, v in starts["greedy" if ref_label == "mma" else "mma"][1].items()},
        "best": {k: float(v) for k, v in best_scores.items()},
        "final": {k: float(v) for k, v in final_scores.items()},
        "steps": result["steps"], "stopped": bool(result["stopped"]),
        "seconds_reference": t_ref, "seconds_policy": t_pol,
        "checkpoint": str(CKPT.relative_to(ROOT)),
    }
    (OUTDIR / f"{name}.json").write_text(json.dumps(meta, indent=1))
    print(f"{name}: ref {ref_label} C={ref_scores['compliance']:.2f} ({ref_scores['contacts']} contacts) | "
          f"agent best C={best_scores['compliance']:.2f} final C={final_scores['compliance']:.2f} "
          f"({final_scores['contacts']} contacts) steps {result['steps']} | ref {t_ref:.0f}s policy {t_pol:.0f}s",
          flush=True)
