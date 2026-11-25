# Structural optimization with PPO

A PPO agent places material on a 2D grid to minimize compliance at a fixed
volume; a finite-element solver scores each design. See `README.md` for the
episode, reward, observation and benchmark.

## Commands

Run from this folder. The virtual environment is `../rl_env` (Python 3.12,
PyTorch 2.5.1 with CUDA, Gymnasium 1.0.0).

```bash
../rl_env/bin/python -m unittest discover -s tests
../rl_env/bin/python src/train_cnn.py --width 12 --height 8 --target-density 0.45 --timesteps 300000
../rl_env/bin/python src/benchmark.py --checkpoint <path.pth> --width 12 --height 8 --target-density 0.45
```

## Active code

- `src/env/mbb_cnn.py`: environment. `src/env/domains.py`: random problems.
- `src/models/ppo_cnn.py`: networks and PPO.
- `src/utils/fem_fast.py`: solver used by the environment. `src/utils/fem.py`:
  reference solver and MMA; keep it unchanged, tests compare against it.
- `src/utils/layout.py`: observation channels, shared by environment and policy.
- `src/utils/simp.py`: classical SIMP reference (density filter + projection).
- `src/train_cnn.py`, `src/benchmark.py`, `src/ppo_main_cnn.ipynb`.
- `scripts/`: large-grid inference, SIMP comparison and plots, run from this
  folder. Outputs go to `benchmarks/` and `docs/images/`.

The older MLP agent, ensemble, surrogate and their notebooks were removed;
they are in the git history.

## Rules that are easy to break

- Width is X / columns, height is Y / rows. Fields are `(height, width)`;
  FEM nodal arrays are `(width + 1, height + 1, 2)`.
- The policy rebuilds the action mask from the observation. Any change to the
  environment's mask or to `layout.py` must be mirrored in
  `_SpatialActorCritic.valid_action_mask`; a test checks they agree exactly.
- Vector environments reset a finished episode on the next step. That step is
  marked invalid in the rollout buffer and must stay excluded from training.
- Bump `LAYOUT_VERSION` when the observation or action layout changes.

## State as of 2026-11-26

- Trained on CUDA here (RTX 4070 laptop). On this laptop use
  `--num-envs 16 --vector sync`; the default 64 async environments crashed it.
- Main model: `src/PPO_preTrained/scalable/ppo_12x6_seed0_1.pth` (12x6, 45%,
  500k vector steps). Second seed: `ppo_12x6_seed1_0.pth`. Checkpoints are not
  in git.
- Run unchanged on 80x40, it is a median 6-7% less stiff than SIMP
  (`src/utils/simp.py`) on 59 problems, with no failures, for both seeds.
  It never beats SIMP there and takes about 100 s per design against 20 s.
  Results and figures: `benchmarks/README.md`; scripts: `scripts/`.
- Set `OMP_NUM_THREADS=1` per worker when running several inference processes;
  thread contention made each solve about 10x slower.
- `benchmarks/references.json` holds 14 policy designs among the 12x6
  references; compare compliance against classical references for fair gaps.

## Next steps

1. Smaller networks (`--base-channels 32` or `16`): the network is about 85%
   of training time and half of the inference time per step.
2. Fine-tune on 40x20 so member thickness adapts to the grid.
3. The stop action competes with every cell in one softmax, so the agent never
   stops on large grids.
4. Not implemented: multi-cell actions for large grids, mixed grid sizes in one
   run. `env.render` still draws load arrows with y flipped.
