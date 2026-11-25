# Benchmark results

All runs use 45% material unless the name says otherwise. Compliance is
scored with `env.evaluate_design` (the solver behind the agent's reward), and
every design has exactly the environment's solid-cell budget. Scripts are in
`../scripts`; run them from the project folder with `../rl_env/bin/python`.
Checkpoints are in `src/PPO_preTrained/scalable/` (not in git).

## Figures

| File | What it shows | Made with |
|---|---|---|
| `fixed_beams_6x6.png`, `fixed_beams_6x6_10k_vs_20k.png` | 6x6, 40% material, 10k and 20k steps | `plot_fixed_beams.py` |
| `fixed_beams_6x6_45pct.png` | 6x6, 45% material, 20k steps | `plot_fixed_beams.py` |
| `fixed_beams_12x6_45pct.png`, `beam_1_12x6_45pct.png` | 12x6, 10k steps | `plot_fixed_beams.py`, `plot_one_beam.py` |
| `fixed_beams_12x6_45pct_500k.png`, `beam_1_12x6_45pct_500k.png` | 12x6, 500k steps (`ppo_12x6_seed0_1.pth`, the main model) | same |
| `fixed_beams_40x20_45pct_10k.png` | 40x20 trained from scratch, 10k steps (too short) | `plot_fixed_beams.py` |
| `fixed_beams_40x20_45pct_zeroshot_from_12x6.png` | 12x6 model run on 40x20, no retraining | `plot_fixed_beams.py` |
| `fixed_beams_80x40_45pct_zeroshot_from_12x6.png` | 12x6 model on 80x40 vs thresholded MMA / greedy | `infer_large.py`, `plot_saved.py` |
| `compare_80x40_fixed_beams.png`, `compare_80x40_random_first10.png` | 80x40: SIMP, thresholded MMA, agent (seed 0) | `infer_compare.py`, `plot_compare.py` |
| `compare_80x40_seeds_fixed_beams.png` | 80x40: SIMP, agent seed 0, agent seed 1 | `plot_seeds.py` |

## Data

- `compare_80x40_seed0/`, `compare_80x40_seed1/`: one `.json` (scores, timings)
  and one `.npz` (designs) per problem; 9 fixed beams and 50 random problems
  (seeds 10000-10049). Seed 1 reuses seed 0's SIMP and MMA designs
  (`REUSE=benchmarks/compare_80x40_seed0`). Summarize with `summarize_compare.py`.
- `infer_80x40_zeroshot_from_12x6/`: output of `infer_large.py`.
- `references.json`: reference cache used by `src/benchmark.py`. 14 of the 59
  12x6 entries were replaced by designs from the seed-0 policy; for
  classical-only comparisons see `validate_simp.py` and `seeds_12x6.py`.
- `results_ppo_*.csv`: `src/benchmark.py` output per checkpoint.

## Headline numbers (80x40, models trained only on 12x6)

Agent final design vs SIMP (density filter + projection, best of r = 1.5/2/3):

| | Seed 0 | Seed 1 |
|---|---|---|
| Median gap, all 59 | +6.4% | +7.3% |
| Random 50: median / within 5% / within 10% | +5.1% / 25 / 45 | +6.0% / 21 / 40 |
| Fixed 9: median | +10.4% | +11.3% |
| Failures (>10x SIMP) | 0 | 0 |
| Corner contacts, median (SIMP 0, MMA 110) | 4 | 16 |

At 12x6 against classical references (all 59): seed 0 median +2.0% (50 within
10%), seed 1 median +4.1% (41 within 10%). Median time per 80x40 design: SIMP
20 s, thresholded MMA 8 s, agent about 100-115 s.
