import contextlib
import csv
import io
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import benchmark  # noqa: E402
import train_cnn  # noqa: E402
from env.mbb_cnn import BeamOptimizationEnv  # noqa: E402


class ReferenceDesignTests(unittest.TestCase):
    def test_starts_respect_the_solid_budget(self):
        env = BeamOptimizationEnv(width=6, height=4, density=0.4, beam_type=1)
        for start in (benchmark.mma_start(env), benchmark.energy_start(env)):
            self.assertEqual(len(start), env.solid_cell_budget)
            self.assertEqual(len(set(int(cell) for cell in start)), env.solid_cell_budget)

    def test_local_search_reaches_a_single_swap_optimum(self):
        env = BeamOptimizationEnv(width=5, height=4, density=0.4, beam_type=1)
        rng = np.random.default_rng(0)
        start = rng.choice(env.total_cells, env.solid_cell_budget, replace=False)
        start_cost = env.evaluate_design(benchmark.design_from_solid(env, start))["objective_cost"]
        cost, solid_cells = benchmark.local_search(env, start)
        self.assertLessEqual(cost, start_cost)
        self.assertEqual(len(solid_cells), env.solid_cell_budget)
        solid = np.zeros(env.total_cells, dtype=bool)
        solid[solid_cells] = True
        for remove in np.flatnonzero(solid):
            for add in np.flatnonzero(~solid):
                trial = solid.copy()
                trial[remove], trial[add] = False, True
                self.assertGreaterEqual(benchmark._cost(env, trial), cost - 1e-9)

    def test_reference_cache_persists_and_adopts_better_designs(self):
        env = BeamOptimizationEnv(width=4, height=3, density=0.5, beam_type=1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "references.json"
            cache = benchmark.ReferenceCache(path)
            reference = cache.get(env, "beam_1")
            self.assertTrue(path.exists())
            self.assertEqual(benchmark.ReferenceCache(path).get(env)["objective"], reference["objective"])
            design = benchmark.design_from_solid(env, reference["solid_cells"])
            self.assertFalse(cache.offer(env, design, "policy"))       # not better than itself
            cache.entries[benchmark.problem_key(env)]["objective"] += 1.0
            self.assertTrue(cache.offer(env, design, "policy"))
            self.assertEqual(cache.entries[benchmark.problem_key(env)]["source"], "policy")

    def test_problem_key_separates_problems_and_weights(self):
        first = BeamOptimizationEnv(width=4, height=3, beam_type=1)
        second = BeamOptimizationEnv(width=4, height=3, beam_type=3)
        third = BeamOptimizationEnv(width=4, height=3, beam_type=1, reward_weights={"w_contact": 0.0})
        keys = {benchmark.problem_key(env) for env in (first, second, third)}
        self.assertEqual(len(keys), 3)
        self.assertEqual(benchmark.problem_key(first), benchmark.problem_key(BeamOptimizationEnv(width=4, height=3, beam_type=1)))


class EndToEndTests(unittest.TestCase):
    def test_short_training_run_then_benchmark(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            config = train_cnn.make_config(
                width=4,
                height=3,
                target_density=0.5,
                timesteps=40,
                num_envs=3,
                vector="sync",
                rollout_steps=16,
                minibatch_size=16,
                update_epochs=1,
                base_channels=8,
                device="cpu",
                eval_every=2,
                log_dir=directory / "logs",
                checkpoint_dir=directory / "checkpoints",
            )
            with contextlib.redirect_stdout(io.StringIO()):
                checkpoint, log = train_cnn.train(config)
            self.assertTrue(checkpoint.exists())
            self.assertTrue(checkpoint.with_suffix(".json").exists())
            with log.open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 3)                      # ceil(40 / 16) updates
            self.assertEqual(int(rows[-1]["vector_steps"]), 40)
            self.assertLessEqual(int(rows[-1]["transitions"]), 40 * 3)
            for column in ("approx_kl", "explained_variance", "construction_share", "env_seconds"):
                self.assertIn(column, rows[0])
            self.assertTrue(log.with_name(f"{log.stem}_eval.csv").exists())

            agent = benchmark.load_agent(checkpoint, 4, 3, device="cpu")
            with contextlib.redirect_stdout(io.StringIO()):
                problems = benchmark.build_problems(4, 3, 0.5, fixed_beams=(1, 3), random_count=2)
                cache = benchmark.ReferenceCache(directory / "references.json")
                results, summary = benchmark.benchmark(agent, problems, cache, verbose=False)
            self.assertEqual(len(results), 4)
            self.assertEqual(summary["problems"], 4)
            for row in results:
                # A reference is never worse than the policy's best design.
                self.assertGreaterEqual(row["best_gap"], -1e-9)
                self.assertGreaterEqual(row["final_gap"], row["best_gap"] - 1e-9)

    def test_make_config_rejects_unknown_options(self):
        with self.assertRaises(ValueError):
            train_cnn.make_config(widht=4)


if __name__ == "__main__":
    unittest.main()
