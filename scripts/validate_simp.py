"""Compare SIMP (best of radii, thresholded) with the classical MMA+local-search reference."""
import contextlib
import io
import sys
import time
from pathlib import Path

ROOT = Path(sys.argv[1])
W, H = int(sys.argv[2]), int(sys.argv[3])
sys.path.insert(0, str(ROOT / "src"))
import benchmark  # noqa: E402
from utils import simp  # noqa: E402

with contextlib.redirect_stdout(io.StringIO()):
    problems = benchmark.build_problems(W, H, 0.45)
cache = benchmark.ReferenceCache()
print(f"{W}x{H}  {'beam':<7} {'MMA+LS C':>9} {'src':>6} {'SIMP C':>8} {'vs ref':>7} {'rmin':>4} "
      f"{'gray C':>8} {'gray':>5} {'best@':>5} {'contacts S/R':>12} {'sec':>5}")
for name, env in problems:
    key = benchmark.problem_key(env)
    entry = cache.entries.get(key)
    if entry is None or entry["source"] == "policy":
        entry = benchmark.compute_reference(env)  # classical only
    t = time.perf_counter()
    best, runs = simp.best_of_radii(env, radii=(1.5, 2.0, 3.0) if W > 12 else (1.2, 1.5, 2.0))
    sec = time.perf_counter() - t
    b = best["binary"]
    print(f"{'':6} {name:<7} {entry['compliance']:>9.2f} {entry['source']:>6} {b['compliance']:>8.2f} "
          f"{b['compliance'] / entry['compliance'] - 1:>+7.1%} {best['rmin']:>4} {best['gray_compliance']:>8.2f} "
          f"{best['grayness']:>5.2f} {best["best_iteration"]:>5} {b['contacts']:>5}/{entry['contacts']:<6} {sec:>5.1f}",
          flush=True)
