"""Print agent vs SIMP vs MMA statistics for an infer_compare.py output directory.

usage: summarize_compare.py COMPARE_DIR
"""
import json, sys, glob, statistics as st
import numpy as np
D = sys.argv[1]
ms = [json.load(open(p)) for p in glob.glob(f"{D}/*.json")]
def gap(a, b): return a / b - 1
BROKEN = 1e4
for label, sel in (("fixed beams", lambda m: m["name"].startswith("beam")), ("random", lambda m: m["name"].startswith("random")), ("all", lambda m: True)):
    g = [m for m in ms if sel(m)]
    valid = [m for m in g if m["simp"]["compliance"] < BROKEN]
    fin = np.array([gap(m["final"]["compliance"], m["simp"]["compliance"]) for m in valid])
    best = np.array([gap(m["best"]["compliance"], m["simp"]["compliance"]) for m in valid])
    mma = np.array([gap(m["mma"]["compliance"], m["simp"]["compliance"]) for m in valid])
    agent_fail = sum(m["final"]["compliance"] > 10 * m["simp"]["compliance"] for m in valid)
    mma_fail = sum(m["mma"]["compliance"] > 10 * m["simp"]["compliance"] for m in valid)
    ok = fin[fin < 9]
    print(f"== {label}: {len(g)} problems ({len(g)-len(valid)} with broken SIMP)")
    print(f"  agent final vs SIMP: median {np.median(fin):+.1%} | mean of non-failed {ok.mean():+.1%} | agent better on {(fin<0).sum()} | within 5% on {(fin<0.05).sum()} | within 10% on {(fin<0.10).sum()} | failed (>10x) {agent_fail}")
    print(f"  agent best  vs SIMP: median {np.median(best):+.1%}")
    print(f"  MMA-thresh  vs SIMP: median {np.median(mma):+.1%} | MMA better on {(mma<0).sum()} | failed {mma_fail}")
    print(f"  agent vs MMA-thresh: agent better on {sum(m['final']['compliance'] < m['mma']['compliance'] for m in valid)}/{len(valid)}")
    print(f"  corner contacts median: SIMP {st.median(m['simp']['contacts'] for m in g):.0f}, agent {st.median(m['final']['contacts'] for m in g):.0f}, MMA {st.median(m['mma']['contacts'] for m in g):.0f}")
    print(f"  SIMP radius chosen: " + ", ".join(f"{r}:{sum(m['simp']['rmin']==r for m in g)}" for r in (1.5, 2.0, 3.0)))
    print(f"  agent stopped itself: {sum(m['stopped'] for m in g)}/{len(g)}")
worst = sorted(ms, key=lambda m: -gap(m["final"]["compliance"], m["simp"]["compliance"]))[:6]
print("worst:", [(m["name"], f"{gap(m['final']['compliance'], m['simp']['compliance']):+.0%}") for m in worst])
bestl = sorted(ms, key=lambda m: gap(m["final"]["compliance"], m["simp"]["compliance"]))[:6]
print("best:", [(m["name"], f"{gap(m['final']['compliance'], m['simp']['compliance']):+.1%}") for m in bestl])
print("SIMP time median %.0fs, MMA %.0fs, agent %.0fs" % (st.median(m['seconds']['simp'] for m in ms), st.median(m['seconds']['mma'] for m in ms), st.median(m['seconds']['agent'] for m in ms)))
