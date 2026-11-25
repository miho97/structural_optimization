"""Train the fully convolutional PPO agent.

Example:

    python src/train_cnn.py --timesteps 10000 --width 6 --height 6

``timesteps`` counts vector-environment steps. With 64 environments, 10,000
vector steps are up to 640,000 transitions.

Environments reset independently, so episodes of different lengths drift out
of phase and every rollout mixes construction and refinement. One row per PPO
update is written to the log, including KL divergence, clip fraction,
explained variance and where the wall time went.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import torch
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv

from benchmark import evaluate_fixed_beams
from env.mbb_cnn import DEFAULT_REWARD_WEIGHTS, BeamOptimizationEnv
from models.ppo_cnn import PPO
from utils import layout


def build_environment(
    width,
    height,
    target_density,
    beam_type,
    random_domain="mixed",
    contact_weight=DEFAULT_REWARD_WEIGHTS["w_contact"],
    max_refinement_swaps=None,
    density_range=None,
):
    return BeamOptimizationEnv(
        width=width,
        height=height,
        density=target_density,
        beam_type=1 if beam_type is None else beam_type,
        use_random_beam=beam_type is None,
        random_domain=random_domain,
        reward_weights={"w_contact": contact_weight},
        max_refinement_swaps=max_refinement_swaps,
        density_range=density_range if beam_type is None else None,
    )


def make_vector_env(env_fns, kind="async"):
    """Vector environment that resets a finished episode on the next step."""
    kwargs = {}
    try:  # Gymnasium >= 1.1 makes the mode explicit; 1.0 always uses it.
        from gymnasium.vector import AutoresetMode

        kwargs["autoreset_mode"] = AutoresetMode.NEXT_STEP
    except ImportError:
        pass
    if kind == "sync":
        return SyncVectorEnv(env_fns, **kwargs)
    return AsyncVectorEnv(env_fns, context="spawn", **kwargs)


def next_available_path(directory, stem, suffix):
    directory.mkdir(parents=True, exist_ok=True)
    run = 0
    while True:
        candidate = directory / f"{stem}_{run}{suffix}"
        if not candidate.exists():
            return candidate
        run += 1


LOG_FIELDS = (
    "update",
    "vector_steps",
    "transitions",
    "episodes",
    "mean_episode_return",
    "mean_episode_length",
    "mean_final_log_ratio",
    "mean_best_log_ratio",
    "mean_final_contacts",
    "stop_fraction",
    "construction_share",
    "actor_loss",
    "critic_loss",
    "entropy",
    "approx_kl",
    "clip_fraction",
    "explained_variance",
    "value_offset",
    "grad_norm",
    "gradient_steps",
    "learning_rate",
    "entropy_coef",
    "transitions_per_second",
    "env_seconds",
    "policy_seconds",
    "update_seconds",
    "elapsed_seconds",
)


def train(config):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = torch.device(
        config.device if config.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    beam_type = None if config.random_beams else config.beam_type
    density_range = tuple(config.density_range) if config.density_range else None
    env_fn = partial(
        build_environment,
        config.width,
        config.height,
        config.target_density,
        beam_type,
        config.domain,
        config.contact_weight,
        config.max_refinement_swaps,
        density_range,
    )
    vector_env = make_vector_env([env_fn for _ in range(config.num_envs)], config.vector)
    observation_shape = tuple(int(size) for size in vector_env.single_observation_space.shape)
    action_dim = int(vector_env.single_action_space.n)
    stop_action = action_dim - 1
    total_updates = math.ceil(config.timesteps / config.rollout_steps)
    agent = PPO(
        state_channels=observation_shape[0],
        width=observation_shape[1],
        height=observation_shape[2],
        action_dim=action_dim,
        lr_actor=config.lr_actor,
        lr_critic=config.lr_critic,
        gamma=config.gamma,
        K_epochs=config.update_epochs,
        eps_clip=config.clip,
        gae_lambda=config.gae_lambda,
        num_envs=config.num_envs,
        device=device,
        use_two_path=config.two_path,
        rollout_steps=config.rollout_steps,
        minibatch_size=config.minibatch_size,
        value_coef=config.value_coef,
        entropy_coef=config.entropy_coef,
        entropy_coef_final=config.entropy_coef_final,
        total_updates=total_updates if config.anneal else None,
        target_kl=config.target_kl if config.target_kl > 0 else None,
        base_channels=config.base_channels,
    )
    if config.init_checkpoint:
        agent.load(config.init_checkpoint, weights_only=True)
        print(f"Initialized weights from {config.init_checkpoint}")

    stem = f"ppo_{config.width}x{config.height}_seed{config.seed}"
    log_path = next_available_path(config.log_dir, stem, ".csv")
    eval_path = log_path.with_name(f"{log_path.stem}_eval.csv")
    checkpoint_path = next_available_path(config.checkpoint_dir, stem, ".pth")
    config_path = checkpoint_path.with_suffix(".json")
    serialized_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(config).items()
    }
    serialized_config.update(
        {
            "resolved_device": str(device),
            "observation_shape": observation_shape,
            "action_dim": action_dim,
            "total_updates": total_updates,
            "parameter_count": sum(parameter.numel() for parameter in agent.policy.parameters()),
        }
    )
    with config_path.open("w") as config_file:
        json.dump(serialized_config, config_file, indent=2, sort_keys=True)
    print(f"Device: {device}")
    print(f"Observation: {observation_shape}; actions: {action_dim}")
    print(
        f"Training for {config.timesteps:,} vector steps / "
        f"up to {config.timesteps * config.num_envs:,} transitions / {total_updates:,} updates"
    )
    print(f"Log: {log_path}")
    print(f"Checkpoint: {checkpoint_path}")

    started = time.perf_counter()
    env_seconds = policy_seconds = update_seconds = 0.0
    vector_steps = transitions = episodes = updates = 0
    next_save = config.save_every
    episode_return = np.zeros(config.num_envs)
    episode_length = np.zeros(config.num_envs, dtype=np.int64)
    finished = []  # (return, length, final log ratio, best log ratio, contacts, stopped)
    construction_steps = rollout_valid_steps = 0
    eval_writer = None
    eval_file = None

    def run_evaluation():
        nonlocal eval_writer, eval_file
        rows = evaluate_fixed_beams(
            agent,
            config.width,
            config.height,
            config.target_density,
            reward_weights={"w_contact": config.contact_weight},
            max_refinement_swaps=config.max_refinement_swaps,
        )
        summary = {
            "update": updates,
            "vector_steps": vector_steps,
            "mean_final_log_ratio": float(np.mean([row["final_log_ratio"] for row in rows])),
            "mean_best_log_ratio": float(np.mean([row["best_log_ratio"] for row in rows])),
            "disconnected": int(sum(row["final_log_ratio"] > 1.0 for row in rows)),
            "mean_final_contacts": float(np.mean([row["final_contacts"] for row in rows])),
            "mean_steps": float(np.mean([row["steps"] for row in rows])),
        }
        if eval_writer is None:
            eval_file = eval_path.open("w", newline="")
            eval_writer = csv.DictWriter(eval_file, fieldnames=summary.keys())
            eval_writer.writeheader()
        eval_writer.writerow(summary)
        eval_file.flush()
        print(
            f"  eval on fixed beams: log10(C/C0) final {summary['mean_final_log_ratio']:+.3f}, "
            f"best {summary['mean_best_log_ratio']:+.3f}, "
            f"disconnected {summary['disconnected']}/{len(rows)}, "
            f"contacts {summary['mean_final_contacts']:.1f}, steps {summary['mean_steps']:.0f}"
        )
        return rows

    with log_path.open("w", newline="") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=LOG_FIELDS)
        writer.writeheader()
        try:
            states_np, _ = vector_env.reset(seed=config.seed)
            states = torch.as_tensor(states_np, dtype=torch.float32, device=device)
            # After an episode ends, the vector environment spends one step
            # resetting it and ignores the action. That step is not a real
            # transition and is excluded from training.
            previous_done = np.zeros(config.num_envs, dtype=bool)

            while vector_steps < config.timesteps:
                tick = time.perf_counter()
                actions, logprobs, state_values = agent.select_action(states)
                actions_np = actions.cpu().numpy()
                policy_seconds += time.perf_counter() - tick

                tick = time.perf_counter()
                next_states_np, rewards, terminated, truncated, infos = vector_env.step(actions_np)
                env_seconds += time.perf_counter() - tick

                done = np.logical_or(terminated, truncated)
                valid = ~previous_done
                agent.buffer.store(
                    states,
                    actions,
                    logprobs,
                    torch.as_tensor(rewards, dtype=torch.float32),
                    state_values,
                    torch.as_tensor(done, dtype=torch.bool),
                    torch.as_tensor(valid, dtype=torch.bool),
                )

                episode_return[valid] += rewards[valid]
                episode_length[valid] += 1
                rollout_valid_steps += int(valid.sum())
                undecided = states_np[:, layout.UNDECIDED].reshape(config.num_envs, -1)
                construction_steps += int(np.count_nonzero(valid & (undecided.max(axis=1) > 0.5)))
                for index in np.flatnonzero(done & valid):
                    final_ratio = float(infos["compliance_ratio"][index])
                    best_ratio = float(infos["best_compliance_ratio"][index])
                    finished.append(
                        (
                            episode_return[index],
                            episode_length[index],
                            math.log10(final_ratio) if final_ratio > 0 else float("nan"),
                            math.log10(best_ratio) if 0 < best_ratio < float("inf") else float("nan"),
                            float(infos["contacts"][index]),
                            float(actions_np[index] == stop_action),
                        )
                    )
                    episode_return[index] = 0.0
                    episode_length[index] = 0
                episodes += int(np.count_nonzero(done & valid))
                transitions += int(valid.sum())
                previous_done = done
                states_np = next_states_np
                states = torch.as_tensor(states_np, dtype=torch.float32, device=device)
                vector_steps += 1

                if agent.buffer.is_full or vector_steps >= config.timesteps:
                    tick = time.perf_counter()
                    metrics = agent.update(agent.estimate_values(states))
                    update_seconds += time.perf_counter() - tick
                    updates += 1
                    elapsed = time.perf_counter() - started
                    stats = np.array(finished, dtype=float) if finished else np.full((1, 6), np.nan)
                    row = {
                        "update": updates,
                        "vector_steps": vector_steps,
                        "transitions": transitions,
                        "episodes": episodes,
                        "mean_episode_return": float(np.nanmean(stats[:, 0])) if finished else "",
                        "mean_episode_length": float(np.nanmean(stats[:, 1])) if finished else "",
                        "mean_final_log_ratio": float(np.nanmean(stats[:, 2])) if finished else "",
                        "mean_best_log_ratio": float(np.nanmean(stats[:, 3])) if finished else "",
                        "mean_final_contacts": float(np.nanmean(stats[:, 4])) if finished else "",
                        "stop_fraction": float(np.nanmean(stats[:, 5])) if finished else "",
                        "construction_share": construction_steps / max(rollout_valid_steps, 1),
                        "transitions_per_second": transitions / max(elapsed, 1e-9),
                        "env_seconds": round(env_seconds, 3),
                        "policy_seconds": round(policy_seconds, 3),
                        "update_seconds": round(update_seconds, 3),
                        "elapsed_seconds": round(elapsed, 3),
                    }
                    row.update({key: metrics[key] for key in LOG_FIELDS if key in metrics})
                    writer.writerow(row)
                    log_file.flush()
                    if updates % config.print_every_updates == 0 or vector_steps >= config.timesteps:
                        returned = row["mean_episode_return"]
                        print(
                            f"update {updates:,}/{total_updates:,} step {vector_steps:,} "
                            f"episodes {episodes:,} "
                            f"return {returned if returned == '' else format(returned, '.3f')} "
                            f"kl {metrics['approx_kl']:.4f} ev {metrics['explained_variance']:.2f} "
                            f"entropy {metrics['entropy']:.2f} "
                            f"{row['transitions_per_second']:.0f} tr/s"
                        )
                    finished.clear()
                    construction_steps = rollout_valid_steps = 0
                    if config.eval_every and updates % config.eval_every == 0:
                        run_evaluation()

                if vector_steps >= next_save:
                    agent.save(checkpoint_path)
                    next_save += config.save_every
        finally:
            vector_env.close()

    agent.save(checkpoint_path)
    elapsed = time.perf_counter() - started
    print(
        f"Training finished in {elapsed:.1f}s with {updates} PPO updates. "
        f"Time split: environments {env_seconds:.0f}s, policy {policy_seconds:.0f}s, "
        f"updates {update_seconds:.0f}s."
    )
    if not config.skip_evaluation:
        run_evaluation()
    if eval_file is not None:
        eval_file.close()
    print(f"Score it against references with: python src/benchmark.py --checkpoint {checkpoint_path} "
          f"--width {config.width} --height {config.height} --target-density {config.target_density}")
    return checkpoint_path, log_path


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--width", type=int, default=6)
    parser.add_argument("--height", type=int, default=6)
    parser.add_argument("--timesteps", type=int, default=10_000)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--vector", choices=("async", "sync"), default="async",
                        help="async uses one process per environment; sync steps them in-process")
    parser.add_argument("--target-density", type=float, default=0.4)
    parser.add_argument("--density-range", type=float, nargs=2, default=None, metavar=("LOW", "HIGH"),
                        help="sample a target density per episode (random beams only)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--random-beams", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--domain", choices=("legacy", "rich", "mixed"), default="mixed")
    parser.add_argument("--beam-type", type=int, choices=range(1, 10), default=1)
    parser.add_argument("--contact-weight", type=float, default=DEFAULT_REWARD_WEIGHTS["w_contact"],
                        help="cost per corner-only contact; 0 disables it")
    parser.add_argument("--max-refinement-swaps", type=int, default=None,
                        help="default: a quarter of the cell count")
    parser.add_argument("--two-path", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--init-checkpoint", type=Path, default=None,
                        help="start from these weights, for example from a smaller grid")
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--minibatch-size", type=int, default=1024)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--lr-actor", type=float, default=3e-4)
    parser.add_argument("--lr-critic", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--entropy-coef-final", type=float, default=0.001)
    parser.add_argument("--target-kl", type=float, default=0.02, help="0 disables early stopping")
    parser.add_argument("--anneal", action=argparse.BooleanOptionalAction, default=True,
                        help="linear learning-rate and entropy schedules")
    parser.add_argument("--save-every", type=int, default=1_000, help="in vector steps")
    parser.add_argument("--eval-every", type=int, default=100, help="in updates; 0 disables")
    parser.add_argument("--print-every-updates", type=int, default=10)
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--log-dir", type=Path, default=Path("src/PPO_logs/scalable"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("src/PPO_preTrained/scalable"))
    return parser


def parse_args(argv=None):
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.timesteps < 1 or arguments.num_envs < 1:
        parser.error("timesteps and num-envs must be positive")
    if arguments.save_every < 1 or arguments.print_every_updates < 1:
        parser.error("save-every and print-every-updates must be positive")
    return arguments


def make_config(**overrides):
    """Default configuration with keyword overrides, for notebooks.

    Example: ``train(make_config(width=12, height=8, timesteps=300_000))``.
    """
    config = build_parser().parse_args([])
    unknown = set(overrides) - set(vars(config))
    if unknown:
        raise ValueError(f"unknown options: {sorted(unknown)}")
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


if __name__ == "__main__":
    configuration = parse_args()
    print(f"Started: {datetime.now().isoformat(timespec='seconds')}")
    train(configuration)
