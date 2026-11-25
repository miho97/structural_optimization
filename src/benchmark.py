"""Score a trained policy against classical reference designs.

Example:

    python src/benchmark.py --checkpoint src/PPO_preTrained/scalable/ppo_12x8_seed0_0.pth \
        --width 12 --height 8 --target-density 0.45 --random 50

Problems are the nine built-in beams plus seeded random domains. For each
one the script runs the deterministic policy and compares it with a
reference design built without learning:

1. MMA on the continuous problem, thresholded to the solid budget.
2. A greedy construction that always fills the most strained undecided cell.
3. Single-swap local search from each start, on the same objective the
   agent is rewarded for (compliance plus the corner-contact cost).

A reference is the best design found, not a proven optimum. References are
cached in ``benchmarks/references.json`` and improve over time: when a policy
finds a better design, that design becomes the reference.

Use ``--references-only`` to build the cache without a checkpoint.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np
import torch

from env.mbb_cnn import (
    DEFAULT_REWARD_WEIGHTS,
    MAXIMUM_DENSITY,
    MINIMUM_DENSITY,
    BeamOptimizationEnv,
)
from models.ppo_cnn import PPO
from utils import fem

REFERENCE_VERSION = 1
# A policy design counts as a failure when its compliance exceeds the
# reference by this factor, which in practice means it is disconnected.
FAILURE_FACTOR = 10.0
DEFAULT_CACHE = Path(__file__).resolve().parents[1] / "benchmarks" / "references.json"


# ----------------------------------------------------------------------
# Problems
# ----------------------------------------------------------------------
def build_problems(
    width,
    height,
    density,
    fixed_beams=range(1, 10),
    random_count=0,
    random_domain="mixed",
    seed=10_000,
    reward_weights=None,
    max_refinement_swaps=None,
):
    """Return ``[(name, env), ...]`` with one fixed problem per environment."""
    common = dict(
        width=width,
        height=height,
        density=density,
        reward_weights=reward_weights,
        max_refinement_swaps=max_refinement_swaps,
    )
    problems = []
    for beam_type in fixed_beams:
        try:
            problems.append((f"beam_{beam_type}", BeamOptimizationEnv(beam_type=beam_type, **common)))
        except ValueError as error:
            print(f"Skipping beam {beam_type} on {width}x{height}: {error}")
    for index in range(random_count):
        sampler = BeamOptimizationEnv(use_random_beam=True, random_domain=random_domain, **common)
        sampler.reset(seed=seed + index)
        env = BeamOptimizationEnv(
            custom_normals=sampler.normals, custom_forces=sampler.forces, **common
        )
        problems.append((f"random_{random_domain}_{seed + index}", env))
    return problems


# ----------------------------------------------------------------------
# Policy rollout
# ----------------------------------------------------------------------
def run_policy(agent, env, deterministic=True):
    """Run one episode and return the final and best designs with metrics."""
    obs, _ = env.reset(options={"regenerate_beam": False})
    total_reward = 0.0
    steps = 0
    stopped = False
    while True:
        state = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        action, _, _ = agent.select_action(state, deterministic=deterministic)
        action = int(action.item())
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        if terminated or truncated:
            stopped = action == env.stop_action
            break
    return {
        "steps": steps,
        "stopped": stopped,
        "total_reward": total_reward,
        "final_objective": env.current_objective,
        "final_compliance": env.current_compliance,
        "final_contacts": env.current_contacts,
        "final_density": env.density_field(),
        "best_objective": env.best_objective,
        "best_compliance": env.best_compliance,
        "best_contacts": env.best_contacts,
        "best_density": env.best_density_field(),
        "base_compliance": env.base_compliance_value,
    }


def evaluate_fixed_beams(agent, width, height, density, reward_weights=None,
                         max_refinement_swaps=None):
    """Quick deterministic check on the built-in beams, without references."""
    rows = []
    with contextlib.redirect_stdout(io.StringIO()):
        problems = build_problems(
            width, height, density, reward_weights=reward_weights,
            max_refinement_swaps=max_refinement_swaps,
        )
    for name, env in problems:
        result = run_policy(agent, env)
        rows.append(
            {
                "problem": name,
                "final_log_ratio": float(np.log10(result["final_compliance"] / result["base_compliance"])),
                "best_log_ratio": float(np.log10(result["best_compliance"] / result["base_compliance"])),
                "final_contacts": result["final_contacts"],
                "steps": result["steps"],
                "stopped": result["stopped"],
            }
        )
    return rows


# ----------------------------------------------------------------------
# Reference designs
# ----------------------------------------------------------------------
def design_from_solid(env, solid_cells):
    density = np.full(env.total_cells, MINIMUM_DENSITY)
    density[np.asarray(solid_cells, dtype=int)] = MAXIMUM_DENSITY
    return density.reshape(env.height, env.width)


def _cost(env, solid_mask):
    density = np.where(solid_mask, MAXIMUM_DENSITY, MINIMUM_DENSITY).reshape(env.height, env.width)
    return env.evaluate_design(density)["objective_cost"]


def mma_start(env):
    """Solid cells from thresholding the MMA optimum of the continuous problem."""
    with contextlib.redirect_stdout(io.StringIO()):
        _, final_field, _, _ = fem.fast_stopt(env.args, verbose=False)
    order = np.argsort(-np.asarray(final_field).reshape(-1), kind="stable")
    return np.sort(order[: env.solid_cell_budget])


def energy_start(env):
    """Greedy construction: fill the most strained undecided cell each step."""
    density = np.full(env.total_cells, env.target_density)
    undecided = np.ones(env.total_cells, dtype=bool)
    for _ in range(env.solid_cell_budget):
        _, energy = env.solver.solve(density.reshape(env.height, env.width), return_energy=True)
        candidates = np.flatnonzero(undecided)
        cell = candidates[np.argmax(energy.reshape(-1)[candidates])]
        density[cell] = MAXIMUM_DENSITY
        undecided[cell] = False
    return np.flatnonzero(~undecided)


def local_search(env, solid_cells, exhaustive_limit=5000, candidates=16, max_iterations=2000):
    """Single-swap descent on the environment's objective.

    Each iteration first tries swaps between the least strained solid cells
    and the most strained void cells. When the full swap neighbourhood has
    at most ``exhaustive_limit`` members it is searched completely before
    stopping, so the result is a true single-swap local optimum.
    """
    solid = np.zeros(env.total_cells, dtype=bool)
    solid[np.asarray(solid_cells, dtype=int)] = True
    current = _cost(env, solid)
    neighbourhood = int(solid.sum()) * int((~solid).sum())

    def best_swap(remove_cells, add_cells):
        best_cost, best_pair = current, None
        for remove in remove_cells:
            solid[remove] = False
            for add in add_cells:
                solid[add] = True
                cost = _cost(env, solid)
                solid[add] = False
                if cost < best_cost - 1e-10:
                    best_cost, best_pair = cost, (remove, add)
            solid[remove] = True
        return best_cost, best_pair

    for _ in range(max_iterations):
        density = np.where(solid, MAXIMUM_DENSITY, MINIMUM_DENSITY).reshape(env.height, env.width)
        _, energy = env.solver.solve(density, return_energy=True)
        energy = energy.reshape(-1)
        solid_cells_now = np.flatnonzero(solid)
        void_cells_now = np.flatnonzero(~solid)
        remove_cells = solid_cells_now[np.argsort(energy[solid_cells_now])[:candidates]]
        add_cells = void_cells_now[np.argsort(-energy[void_cells_now])[:candidates]]
        cost, pair = best_swap(remove_cells, add_cells)
        if pair is None and neighbourhood <= exhaustive_limit:
            cost, pair = best_swap(solid_cells_now, void_cells_now)
        if pair is None:
            break
        solid[pair[0]] = False
        solid[pair[1]] = True
        current = cost
    return current, np.flatnonzero(solid)


def compute_reference(env, extra_starts=None, exhaustive_limit=5000):
    """Best design over local searches from the MMA and energy starts."""
    starts = {"mma": mma_start(env), "energy": energy_start(env)}
    for name, cells in (extra_starts or {}).items():
        starts[name] = np.asarray(cells, dtype=int)
    best = None
    for name, cells in starts.items():
        cost, solid_cells = local_search(env, cells, exhaustive_limit=exhaustive_limit)
        if best is None or cost < best[0]:
            best = (cost, solid_cells, name)
    cost, solid_cells, source = best
    scores = env.evaluate_design(design_from_solid(env, solid_cells))
    return {
        "objective": scores["objective_cost"],
        "compliance": scores["compliance"],
        "contacts": scores["contacts"],
        "solid_cells": [int(cell) for cell in solid_cells],
        "source": source,
    }


def problem_key(env):
    digest = hashlib.sha1()
    digest.update(np.asarray([env.width, env.height, env.solid_cell_budget], dtype=np.int64).tobytes())
    digest.update(
        np.asarray(
            [env.target_density, env.w_compliance, env.w_contact, env.w_total_mass, env.w_binary],
            dtype=np.float64,
        ).tobytes()
    )
    digest.update(np.ascontiguousarray(env.normals, dtype=np.float64).tobytes())
    digest.update(np.round(np.ascontiguousarray(env.forces, dtype=np.float64), 9).tobytes())
    return f"v{REFERENCE_VERSION}-{digest.hexdigest()}"


class ReferenceCache:
    def __init__(self, path=DEFAULT_CACHE):
        self.path = Path(path)
        self.entries = {}
        if self.path.exists():
            self.entries = json.loads(self.path.read_text())

    def get(self, env, name=None, exhaustive_limit=5000):
        key = problem_key(env)
        if key not in self.entries:
            entry = compute_reference(env, exhaustive_limit=exhaustive_limit)
            entry["problem"] = name
            entry["grid"] = [env.width, env.height]
            self.entries[key] = entry
            self.save()
        return self.entries[key]

    def offer(self, env, design, source):
        """Adopt ``design`` as the reference if it beats the cached one."""
        key = problem_key(env)
        scores = env.evaluate_design(design)
        if key in self.entries and scores["objective_cost"] >= self.entries[key]["objective"] - 1e-9:
            return False
        solid_cells = np.flatnonzero(np.asarray(design).reshape(-1) >= MAXIMUM_DENSITY - 1e-6)
        previous = self.entries.get(key, {})
        self.entries[key] = {
            "objective": scores["objective_cost"],
            "compliance": scores["compliance"],
            "contacts": scores["contacts"],
            "solid_cells": [int(cell) for cell in solid_cells],
            "source": source,
            "problem": previous.get("problem"),
            "grid": [env.width, env.height],
        }
        self.save()
        return True

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.entries, indent=1, sort_keys=True))


# ----------------------------------------------------------------------
# Benchmark
# ----------------------------------------------------------------------
def benchmark(agent, problems, cache=None, exhaustive_limit=5000, verbose=True):
    """Score ``agent`` on ``problems``. Returns ``(rows, summary)``."""
    cache = ReferenceCache() if cache is None else cache
    rows = []
    for name, env in problems:
        started = time.perf_counter()
        reference = cache.get(env, name, exhaustive_limit=exhaustive_limit)
        result = run_policy(agent, env)
        if result["best_density"] is not None and cache.offer(env, result["best_density"], "policy"):
            reference = cache.entries[problem_key(env)]
        final_gap = 10 ** (result["final_objective"] - reference["objective"]) - 1
        best_gap = 10 ** (result["best_objective"] - reference["objective"]) - 1
        rows.append(
            {
                "problem": name,
                "final_compliance": result["final_compliance"],
                "final_contacts": result["final_contacts"],
                "best_compliance": result["best_compliance"],
                "best_contacts": result["best_contacts"],
                "reference_compliance": reference["compliance"],
                "reference_contacts": reference["contacts"],
                "reference_source": reference["source"],
                "final_gap": final_gap,
                "best_gap": best_gap,
                "failed": bool(
                    result["final_compliance"] > FAILURE_FACTOR * reference["compliance"]
                ),
                "steps": result["steps"],
                "stopped": result["stopped"],
                "seconds": time.perf_counter() - started,
            }
        )
        if verbose:
            row = rows[-1]
            print(
                f"{name:<24} final C={row['final_compliance']:>12.4f} ({row['final_contacts']} contacts) "
                f"best C={row['best_compliance']:>12.4f} ref C={row['reference_compliance']:>10.4f} "
                f"({row['reference_contacts']} contacts, {row['reference_source']}) "
                f"gap final {row['final_gap']:+8.1%} best {row['best_gap']:+8.1%} "
                f"steps={row['steps']}{' FAILED' if row['failed'] else ''}"
            )
    summary = summarize(rows)
    if verbose:
        print(
            f"\n{summary['problems']} problems | failures {summary['failures']} | "
            f"median gap: final {summary['median_final_gap']:+.1%}, best {summary['median_best_gap']:+.1%} | "
            f"worst non-failing final gap {summary['worst_final_gap']:+.1%} | "
            f"mean steps {summary['mean_steps']:.0f}"
        )
    return rows, summary


def summarize(rows):
    working = [row for row in rows if not row["failed"]]
    return {
        "problems": len(rows),
        "failures": len(rows) - len(working),
        "median_final_gap": float(np.median([row["final_gap"] for row in rows])) if rows else float("nan"),
        "median_best_gap": float(np.median([row["best_gap"] for row in rows])) if rows else float("nan"),
        "worst_final_gap": max((row["final_gap"] for row in working), default=float("nan")),
        "mean_steps": float(np.mean([row["steps"] for row in rows])) if rows else float("nan"),
    }


def load_agent(checkpoint_path, width, height, device=None):
    """Build an agent for ``width x height`` and load checkpoint weights."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    meta = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    channels = meta["observation_shape"][0]
    with contextlib.redirect_stdout(io.StringIO()):
        agent = PPO(
            state_channels=channels,
            width=height + 1,
            height=width + 1,
            action_dim=2 * width * height + 1,
            lr_actor=3e-4,
            lr_critic=1e-3,
            gamma=0.995,
            K_epochs=1,
            eps_clip=0.2,
            num_envs=1,
            device=device,
            use_two_path=meta.get("architecture") == "two_path",
            base_channels=meta.get("base_channels", 64),
            rollout_steps=1,
        )
    agent.load(checkpoint_path, weights_only=True)
    return agent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--references-only", action="store_true")
    parser.add_argument("--width", type=int, default=12)
    parser.add_argument("--height", type=int, default=8)
    parser.add_argument("--target-density", type=float, default=0.45)
    parser.add_argument("--random", type=int, default=50, help="number of seeded random problems")
    parser.add_argument("--domain", choices=("legacy", "rich", "mixed"), default="mixed")
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--contact-weight", type=float, default=DEFAULT_REWARD_WEIGHTS["w_contact"])
    parser.add_argument("--max-refinement-swaps", type=int, default=None)
    parser.add_argument("--exhaustive-limit", type=int, default=5000,
                        help="largest swap neighbourhood searched completely for references")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=None, help="CSV path for the result rows")
    arguments = parser.parse_args()
    if arguments.checkpoint is None and not arguments.references_only:
        parser.error("give --checkpoint, or --references-only to build the reference cache")
    return arguments


def main():
    config = parse_args()
    problems = build_problems(
        config.width,
        config.height,
        config.target_density,
        random_count=config.random,
        random_domain=config.domain,
        seed=config.seed,
        reward_weights={"w_contact": config.contact_weight},
        max_refinement_swaps=config.max_refinement_swaps,
    )
    cache = ReferenceCache(config.cache)
    if config.references_only:
        for name, env in problems:
            started = time.perf_counter()
            reference = cache.get(env, name, exhaustive_limit=config.exhaustive_limit)
            print(
                f"{name:<24} C={reference['compliance']:.4f} contacts={reference['contacts']} "
                f"source={reference['source']} ({time.perf_counter() - started:.1f}s)"
            )
        print(f"References cached in {cache.path}")
        return

    agent = load_agent(config.checkpoint, config.width, config.height, config.device)
    rows, _ = benchmark(agent, problems, cache, exhaustive_limit=config.exhaustive_limit)
    output = config.output
    if output is None:
        output = cache.path.parent / f"results_{config.checkpoint.stem}_{config.width}x{config.height}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
