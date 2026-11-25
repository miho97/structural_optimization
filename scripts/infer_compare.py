"""Agent vs SIMP vs thresholded MMA on fixed + random problems at a large grid.

usage: infer_compare.py ROOT CHECKPOINT OUTDIR WIDTH HEIGHT DENSITY WORKER NUM_WORKERS [RANDOM_COUNT]

Problems: the 9 fixed beams plus RANDOM_COUNT (default 50) random 'mixed'
domains with the benchmark's seeds (10000, 10001, ...). Worker k of n takes
every n-th problem. All designs have exactly the environment's solid budget
and are scored with env.evaluate_design (same solver as the agent's reward).
"""
import contextlib
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1])
CKPT = ROOT / sys.argv[2]
OUTDIR = Path(sys.argv[3])
W, H, D = int(sys.argv[4]), int(sys.argv[5]), float(sys.argv[6])
WORKER, WORKERS = int(sys.argv[7]), int(sys.argv[8])
RANDOM = int(sys.argv[9]) if len(sys.argv) > 9 else 50
sys.path.insert(0, str(ROOT / "src"))
import benchmark  # noqa: E402
from utils import simp  # noqa: E402

OUTDIR.mkdir(parents=True, exist_ok=True)
agent = benchmark.load_agent(str(CKPT), W, H)
with contextlib.redirect_stdout(io.StringIO()):
    problems = benchmark.build_problems(W, H, D, random_count=RANDOM)
problems = problems[WORKER::WORKERS]


def scores(env, density):
    return {k: float(v) for k, v in env.evaluate_design(density).items()}


for name, env in problems:
    if (OUTDIR / f"{name}.json").exists():
        continue
    reuse = os.environ.get("REUSE")  # directory of an earlier run: take SIMP and MMA from there
    if reuse:
        old = json.loads((Path(reuse) / f"{name}.json").read_text())
        old_data = np.load(Path(reuse) / f"{name}.npz")
        best_simp = {"density": old_data["simp"], "rmin": old["simp"]["rmin"],
                     "gray_compliance": old["simp"]["gray_compliance"], "best_iteration": old["simp"]["best_iteration"]}
        runs = [{"rmin": float(r), "binary": {"compliance": c}} for r, c in old["simp"]["by_radius"].items()]
        mma_grid = old_data["mma"]
        t_simp, t_mma = old["seconds"]["simp"], old["seconds"]["mma"]
    else:
        t0 = time.perf_counter()
        best_simp, runs = simp.best_of_radii(env, radii=(1.5, 2.0, 3.0))
        t_simp = time.perf_counter() - t0

        t0 = time.perf_counter()
        mma_grid = benchmark.design_from_solid(env, benchmark.mma_start(env))
        t_mma = time.perf_counter() - t0

    t0 = time.perf_counter()
    result = benchmark.run_policy(agent, env)
    t_pol = time.perf_counter() - t0

    np.savez_compressed(
        OUTDIR / f"{name}.npz",
        simp=best_simp["density"], mma=mma_grid, best=result["best_density"], final=result["final_density"],
        normals=env.normals, forces=env.forces,
    )
    meta = {
        "name": name, "width": W, "height": H, "density": D, "checkpoint": str(CKPT.relative_to(ROOT)),
        "simp": scores(env, best_simp["density"]) | {
            "rmin": best_simp["rmin"], "gray_compliance": best_simp["gray_compliance"],
            "best_iteration": best_simp["best_iteration"],
            "by_radius": {str(r["rmin"]): r["binary"]["compliance"] for r in runs},
        },
        "mma": scores(env, mma_grid),
        "best": scores(env, result["best_density"]),
        "final": scores(env, result["final_density"]),
        "steps": result["steps"], "stopped": bool(result["stopped"]),
        "seconds": {"simp": t_simp, "mma": t_mma, "agent": t_pol},
        "domain": getattr(env, "domain_meta", None) if not name.startswith("beam") else None,
    }
    (OUTDIR / f"{name}.json").write_text(json.dumps(meta, indent=1, default=str))
    s, f = meta["simp"]["compliance"], meta["final"]["compliance"]
    print(f"{name}: SIMP C={s:.2f} (r={best_simp['rmin']}) | MMA C={meta['mma']['compliance']:.4g} | "
          f"agent final C={f:.4g} ({f / s - 1:+.1%} vs SIMP), best C={meta['best']['compliance']:.4g} | "
          f"steps {result['steps']} | {t_simp:.0f}s/{t_mma:.0f}s/{t_pol:.0f}s", flush=True)
print(f"worker {WORKER} done", flush=True)
