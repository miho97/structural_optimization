"""Both seeds on the 59 12x6 benchmark problems vs classical references (MMA+greedy+LS), compliance only.
Does not write to the reference cache."""
import contextlib, io, sys, statistics as st
from pathlib import Path
import numpy as np
ROOT = Path(sys.argv[1]); sys.path.insert(0, str(ROOT / "src"))
import benchmark
ckpts = sys.argv[2:]
agents = [benchmark.load_agent(str(ROOT / c), 12, 6) for c in ckpts]
with contextlib.redirect_stdout(io.StringIO()):
    problems = benchmark.build_problems(12, 6, 0.45, random_count=50)
cache = benchmark.ReferenceCache()
rows = []
for name, env in problems:
    entry = cache.entries.get(benchmark.problem_key(env))
    if entry is None or entry["source"] == "policy":
        with contextlib.redirect_stdout(io.StringIO()):
            entry = benchmark.compute_reference(env)
    row = [name, entry["compliance"], entry["contacts"]]
    for a in agents:
        r = benchmark.run_policy(a, env)
        row += [r["final_compliance"], r["final_contacts"], r["stopped"]]
    rows.append(row)
    print(f"{name:<11} ref {row[1]:7.2f}/{row[2]:<2} " + " ".join(
        f"| C {row[3+3*i]:7.2f} ({row[3+3*i]/row[1]-1:+6.1%}) cont {row[4+3*i]:<2} stop {int(row[5+3*i])}" for i in range(len(agents))), flush=True)
for i, c in enumerate(ckpts):
    g = np.array([r[3 + 3 * i] / r[1] - 1 for r in rows])
    ok = g[g < 9]
    print(f"\n{c}: median {np.median(g):+.1%}, mean(non-failed) {ok.mean():+.1%}, stiffer than ref {(g < 0).sum()}, "
          f"within 5% {(g < .05).sum()}, within 10% {(g < .10).sum()}, failed(>10x) {(g >= 9).sum()}, "
          f"stopped {sum(r[5 + 3 * i] for r in rows)}/{len(rows)}, contacts mean {st.mean(r[4 + 3 * i] for r in rows):.2f}")
print(f"reference contacts mean {st.mean(r[2] for r in rows):.2f}")
g0 = np.array([r[3] for r in rows]); g1 = np.array([r[6] for r in rows])
print(f"seed1 vs seed0 per problem: seed1 stiffer on {(g1 < g0 * 0.999).sum()}, equal {(abs(g1 / g0 - 1) <= 0.001).sum()}, seed0 stiffer on {(g0 < g1 * 0.999).sum()}")
