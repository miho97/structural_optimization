# Structural Optimization with Fully Convolutional PPO

This project trains a PPO agent to edit element densities while structural
compliance is evaluated by a sparse finite-element solver. The active scalable
path is:

- `src/env/mbb_cnn.py` — Gymnasium environment and reward
- `src/models/ppo_cnn.py` — fully convolutional actor/critic and PPO
- `src/utils/fem.py` — sparse FEM implementation
- `src/train_cnn.py` — reproducible command-line trainer

## First 10k validation run

From this repository directory:

```bash
../rl_env/bin/python src/train_cnn.py \
  --timesteps 10000 \
  --width 6 \
  --height 6
```

`timesteps` counts vector-environment steps, as it did in the original
notebook. With the default 64 environments, 10,000 vector steps contain up to
640,000 transitions. Logs and checkpoints receive non-overwriting run numbers
under `src/PPO_logs/scalable/` and `src/PPO_preTrained/scalable/`. Each run also
writes its complete configuration and deterministic fixed-beam evaluation.

Use a smaller smoke run before a long job:

```bash
../rl_env/bin/python src/train_cnn.py \
  --timesteps 128 \
  --num-envs 4 \
  --skip-evaluation
```

Run `--help` for fixed-beam training, density, device, rollout, and optimizer
options. `src/ppo_main_cnn.ipynb` remains available and uses the same corrected
environment/model APIs, but the script is the recommended reproducible entry
point.

## Coordinate and physics contract

- Width is X / columns; height is Y / rows.
- Density fields are `(height, width)`.
- CNN observations are `(channels, height + 1, width + 1)`.
- FEM nodal inputs remain `(width + 1, height + 1, 2)` and are transposed once
  when converted to CNN channels.
- The density argument is the authoritative target volume fraction unless the
  legacy `optimal_density` override is explicitly supplied.

Volume violation is measured as a relative fraction and receives a default
weight of five, so feasibility remains important without scaling with the
number of cells.

The actor emits two spatial logits per physical cell (decrease and increase),
so padded nodal pixels can never be selected. A dilated residual context stack
provides a receptive field larger than 50x50, allowing actions to depend on
distant loads and supports. The critic uses global pooling; model parameter
count is independent of grid resolution.

## Tests

The tests use Python's standard-library runner and require no pytest install:

```bash
../rl_env/bin/python -m unittest discover -s tests -v
```

They cover rectangular action mapping, node/image alignment, seeded resets,
volume-penalty invariance, FEM energy consistency and gradients, constant model
size through 50x50, bootstrapped GAE, minibatched updates, and checkpoints.

## Scaling notes

The rollout horizon is independent of episode length, GAE bootstraps partial
rollouts, and PPO updates use minibatches. Minibatch size is automatically
bounded by a spatial-pixel budget on large grids. Sparse FEM topology and
boundary maps are cached, although numerical factorization is still performed
after each density edit.

Old CNN checkpoints used flattened fully connected heads and are intentionally
not architecture-compatible with the scalable policy.
