"""Train the scalable CNN PPO agent.

Example (the requested first validation run):

    python src/train_cnn.py --timesteps 10000 --width 6 --height 6

``timesteps`` counts vector-environment steps, matching the original notebook.
With 64 environments, 10,000 vector steps contain 640,000 transitions.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import torch
from gymnasium.vector import AsyncVectorEnv

from env.mbb_cnn import BeamOptimizationEnv
from models.ppo_cnn import PPO


def build_environment(width, height, target_density, beam_type):
    return BeamOptimizationEnv(
        width=width,
        height=height,
        density=target_density,
        beam_type=1 if beam_type is None else beam_type,
        use_random_beam=beam_type is None,
    )


def make_environment_factories(width, height, target_density, beam_type, num_envs):
    factory = partial(build_environment, width, height, target_density, beam_type)
    return [factory for _ in range(num_envs)]


def next_available_path(directory, stem, suffix):
    directory.mkdir(parents=True, exist_ok=True)
    run = 0
    while True:
        candidate = directory / f"{stem}_{run}{suffix}"
        if not candidate.exists():
            return candidate
        run += 1


def evaluate_fixed_beams(agent, width, height, target_density, device):
    """Run one deterministic episode for each built-in benchmark."""
    results = []
    for beam_type in range(1, 10):
        env = BeamOptimizationEnv(
            width=width,
            height=height,
            density=target_density,
            beam_type=beam_type,
        )
        state, _ = env.reset(seed=10_000 + beam_type)
        total_reward = 0.0
        final_info = {}
        for _ in range(env.max_steps):
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            action, _, _ = agent.select_action(state_tensor, deterministic=True)
            state, reward, terminated, truncated, final_info = env.step(int(action.item()))
            total_reward += reward
            if terminated or truncated:
                break
        results.append(
            {
                "beam_type": beam_type,
                "reward": total_reward,
                "compliance": final_info["compliance"],
                "compliance_ratio": final_info["compliance_ratio"],
                "volume_fraction": final_info["volume_fraction"],
                "volume_violation": final_info["volume_violation"],
            }
        )
        env.close()
    return results


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
    env_fns = make_environment_factories(
        config.width,
        config.height,
        config.target_density,
        beam_type,
        config.num_envs,
    )
    vector_env = AsyncVectorEnv(env_fns, context="spawn")
    observation_shape = tuple(
        int(dimension) for dimension in vector_env.single_observation_space.shape
    )
    action_dim = int(vector_env.single_action_space.n)
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
        has_continuous_action_space=False,
        gae_lambda=config.gae_lambda,
        num_envs=config.num_envs,
        device=device,
        use_two_path=config.two_path,
        rollout_steps=config.rollout_steps,
        minibatch_size=config.minibatch_size,
        value_coef=config.value_coef,
        entropy_coef=config.entropy_coef,
    )

    log_path = next_available_path(
        config.log_dir,
        f"ppo_{config.width}x{config.height}_seed{config.seed}",
        ".csv",
    )
    checkpoint_path = next_available_path(
        config.checkpoint_dir,
        f"ppo_{config.width}x{config.height}_seed{config.seed}",
        ".pth",
    )
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
            "parameter_count": sum(
                parameter.numel() for parameter in agent.policy.parameters()
            ),
        }
    )
    with config_path.open("w") as config_file:
        json.dump(serialized_config, config_file, indent=2, sort_keys=True)
    print(f"Device: {device}")
    print(f"Observation: {observation_shape}; actions: {action_dim}")
    print(
        f"Training for {config.timesteps:,} vector steps / "
        f"up to {config.timesteps * config.num_envs:,} transitions"
    )
    print(f"Log: {log_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Configuration: {config_path}")

    started = time.perf_counter()
    vector_steps = 0
    transitions = 0
    episodes = 0
    updates = 0
    last_states = None
    next_save = config.save_every

    with log_path.open("w", newline="") as log_file:
        writer = csv.DictWriter(
            log_file,
            fieldnames=(
                "episodes",
                "vector_steps",
                "transitions",
                "mean_episode_reward",
                "mean_final_compliance",
                "mean_final_compliance_ratio",
                "mean_final_volume_fraction",
                "updates",
                "elapsed_seconds",
            ),
        )
        writer.writeheader()
        try:
            first_reset = True
            while vector_steps < config.timesteps:
                reset_seed = config.seed if first_reset else None
                states_np, _ = vector_env.reset(seed=reset_seed)
                first_reset = False
                states = torch.as_tensor(states_np, dtype=torch.float32, device=device)
                episode_rewards = np.zeros(config.num_envs, dtype=np.float64)
                final_infos = None

                for _ in range(config.width * config.height * 3):
                    actions, logprobs, state_values = agent.select_action(states)
                    next_states_np, rewards, terminated, truncated, infos = vector_env.step(
                        actions.cpu().numpy()
                    )
                    done = np.logical_or(terminated, truncated)
                    next_states = torch.as_tensor(
                        next_states_np, dtype=torch.float32, device=device
                    )
                    agent.buffer.store(
                        states,
                        actions,
                        logprobs,
                        torch.as_tensor(rewards, dtype=torch.float32),
                        state_values,
                        torch.as_tensor(done, dtype=torch.bool),
                    )
                    episode_rewards += rewards
                    states = next_states
                    last_states = next_states
                    final_infos = infos
                    vector_steps += 1
                    transitions += config.num_envs

                    if agent.buffer.is_full:
                        last_values = agent.estimate_values(next_states)
                        metrics = agent.update(last_values)
                        updates += 1
                        if updates % config.print_every_updates == 0:
                            print(
                                f"step={vector_steps:,} update={updates:,} "
                                f"actor={metrics['actor_loss']:.4f} "
                                f"critic={metrics['critic_loss']:.4f} "
                                f"entropy={metrics['entropy']:.4f}"
                            )

                    if vector_steps >= next_save:
                        agent.save(checkpoint_path)
                        next_save += config.save_every
                    if done.all() or vector_steps >= config.timesteps:
                        break

                completed = int(np.count_nonzero(done)) if final_infos is not None else 0
                episodes += completed
                elapsed = time.perf_counter() - started
                final_compliance = np.asarray(final_infos.get("compliance", np.nan))
                final_ratio = np.asarray(final_infos.get("compliance_ratio", np.nan))
                final_volume = np.asarray(final_infos.get("volume_fraction", np.nan))
                row = {
                    "episodes": episodes,
                    "vector_steps": vector_steps,
                    "transitions": transitions,
                    "mean_episode_reward": float(episode_rewards.mean()),
                    "mean_final_compliance": float(np.nanmean(final_compliance)),
                    "mean_final_compliance_ratio": float(np.nanmean(final_ratio)),
                    "mean_final_volume_fraction": float(np.nanmean(final_volume)),
                    "updates": updates,
                    "elapsed_seconds": round(elapsed, 3),
                }
                writer.writerow(row)
                log_file.flush()
                print(
                    f"episodes={episodes:,} step={vector_steps:,}/{config.timesteps:,} "
                    f"reward={row['mean_episode_reward']:.3f} "
                    f"C/C0={row['mean_final_compliance_ratio']:.3f} "
                    f"volume={row['mean_final_volume_fraction']:.3f}"
                )
        finally:
            vector_env.close()

    if agent.buffer.ptr:
        last_values = agent.estimate_values(last_states)
        agent.update(last_values)
        updates += 1
    agent.save(checkpoint_path)
    elapsed = time.perf_counter() - started
    print(f"Training finished in {elapsed:.1f}s with {updates} PPO updates")

    if not config.skip_evaluation:
        results = evaluate_fixed_beams(
            agent, config.width, config.height, config.target_density, device
        )
        evaluation_path = log_path.with_name(f"{log_path.stem}_evaluation.csv")
        with evaluation_path.open("w", newline="") as evaluation_file:
            writer = csv.DictWriter(evaluation_file, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print("Deterministic fixed-beam evaluation:")
        for result in results:
            print(
                f"  beam={result['beam_type']} C={result['compliance']:.4f} "
                f"C/C0={result['compliance_ratio']:.4f} "
                f"volume={result['volume_fraction']:.4f}"
            )
        print(f"Evaluation: {evaluation_path}")
    return checkpoint_path, log_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=6)
    parser.add_argument("--height", type=int, default=6)
    parser.add_argument("--timesteps", type=int, default=10_000)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--target-density", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--random-beams", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--beam-type", type=int, choices=range(1, 10), default=1)
    parser.add_argument("--two-path", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--minibatch-size", type=int, default=1024)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--lr-actor", type=float, default=3e-4)
    parser.add_argument("--lr-critic", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--save-every", type=int, default=1_000)
    parser.add_argument("--print-every-updates", type=int, default=10)
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--log-dir", type=Path, default=Path("src/PPO_logs/scalable"))
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=Path("src/PPO_preTrained/scalable")
    )
    arguments = parser.parse_args()
    if arguments.timesteps < 1 or arguments.num_envs < 1:
        parser.error("timesteps and num-envs must be positive")
    if arguments.save_every < 1 or arguments.print_every_updates < 1:
        parser.error("save-every and print-every-updates must be positive")
    return arguments


if __name__ == "__main__":
    configuration = parse_args()
    print(f"Started: {datetime.now().isoformat(timespec='seconds')}")
    train(configuration)
