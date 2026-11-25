import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from env.mbb_cnn import BeamOptimizationEnv  # noqa: E402
from models.ppo_cnn import PPO, CNNActorCritic, TwoPathActorCritic  # noqa: E402
from utils import layout  # noqa: E402


def make_model(width, height, two_path=True, base_channels=8):
    if two_path:
        return TwoPathActorCritic(
            num_actions=2 * width * height + 1,
            width=height + 1,
            height=width + 1,
            base_channels=base_channels,
        )
    return CNNActorCritic(
        layout.NUM_CHANNELS,
        2 * width * height + 1,
        height + 1,
        width + 1,
        base_channels=base_channels,
    )


def make_agent(width=3, height=2, num_envs=2, rollout_steps=4, **kwargs):
    options = dict(
        state_channels=layout.NUM_CHANNELS,
        width=height + 1,
        height=width + 1,
        action_dim=2 * width * height + 1,
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.9,
        K_epochs=2,
        eps_clip=0.2,
        gae_lambda=1.0,
        num_envs=num_envs,
        device="cpu",
        use_two_path=True,
        rollout_steps=rollout_steps,
        minibatch_size=4,
        base_channels=8,
    )
    options.update(kwargs)
    return PPO(**options)


def collect_rollout(agent, envs, steps):
    """Fill the agent's buffer from real environments, with next-step resets."""
    observations = [env.reset(seed=index)[0] for index, env in enumerate(envs)]
    previous_done = np.zeros(len(envs), dtype=bool)
    for _ in range(steps):
        states = torch.as_tensor(np.stack(observations), dtype=torch.float32)
        actions, logprobs, values = agent.select_action(states)
        rewards, dones = [], []
        for index, env in enumerate(envs):
            if previous_done[index]:
                observations[index] = env.reset()[0]
                rewards.append(0.0)
                dones.append(False)
                continue
            observations[index], reward, terminated, truncated, _ = env.step(int(actions[index]))
            rewards.append(reward)
            dones.append(terminated or truncated)
        agent.buffer.store(
            states,
            actions,
            logprobs,
            torch.tensor(rewards),
            values,
            torch.tensor(dones),
            torch.as_tensor(~previous_done),
        )
        previous_done = np.array(dones)
    return torch.as_tensor(np.stack(observations), dtype=torch.float32)


class FullyConvolutionalPolicyTests(unittest.TestCase):
    def test_parameter_count_is_independent_of_resolution(self):
        for two_path in (True, False):
            small = make_model(6, 6, two_path)
            large = make_model(50, 50, two_path)
            small_count = sum(parameter.numel() for parameter in small.parameters())
            large_count = sum(parameter.numel() for parameter in large.parameters())
            self.assertEqual(small_count, large_count)
            self.assertGreater(small.receptive_field, 50)

    def test_default_width_matches_the_original_network(self):
        model = TwoPathActorCritic(num_actions=2 * 36 + 1, width=7, height=7)
        self.assertEqual(model.feature_channels, 256)

    def test_actor_emits_two_logits_per_cell_plus_stop(self):
        model = make_model(5, 3)
        states = torch.zeros(4, layout.NUM_CHANNELS, 4, 6)
        logits = model.action_output(model.forward_features(states))
        self.assertEqual(tuple(logits.shape), (4, 2 * 15 + 1))
        self.assertEqual(logits.dtype, torch.float32)
        with self.assertRaises(ValueError):
            TwoPathActorCritic(num_actions=2 * 15, width=4, height=6)

    def test_policy_mask_matches_the_environment_everywhere(self):
        rng = np.random.default_rng(0)
        cases = (
            (6, 6, 0.4, None),
            (5, 3, 0.25, None),
            (3, 5, 0.8, 2),
            (2, 2, 0.99, None),    # every cell solid: no swap is possible
            (4, 3, 0.05, None),    # tiny budget
        )
        checked = 0
        for width, height, density, swaps in cases:
            env = BeamOptimizationEnv(
                width=width, height=height, density=density,
                use_random_beam=True, max_refinement_swaps=swaps,
            )
            model = make_model(width, height, two_path=False)
            for seed in range(6):
                obs, _ = env.reset(seed=seed)
                while True:
                    policy_mask = model.valid_action_mask(torch.as_tensor(obs).unsqueeze(0))[0]
                    np.testing.assert_array_equal(policy_mask.numpy(), env.valid_action_mask())
                    checked += 1
                    valid = np.flatnonzero(env.valid_action_mask())
                    if env.stop_action in valid and len(valid) > 1 and rng.random() < 0.8:
                        valid = valid[valid != env.stop_action]
                    obs, _, terminated, truncated, _ = env.step(int(rng.choice(valid)))
                    if terminated or truncated:
                        # A finished episode's observation must still admit an
                        # action, because the vector environment asks for one
                        # on its reset step.
                        model.valid_action_mask(torch.as_tensor(obs).unsqueeze(0))
                        break
        self.assertGreater(checked, 500)

    def test_sampled_actions_are_always_valid(self):
        torch.manual_seed(0)
        env = BeamOptimizationEnv(width=4, height=4, density=0.5, use_random_beam=True)
        model = make_model(4, 4)
        for seed in range(4):
            obs, _ = env.reset(seed=seed)
            while True:
                action, logprob, value = model.act(torch.as_tensor(obs).unsqueeze(0))
                self.assertTrue(env.valid_action_mask()[int(action)])
                self.assertTrue(torch.isfinite(logprob).all() and torch.isfinite(value).all())
                obs, _, terminated, truncated, info = env.step(int(action))
                self.assertTrue(info["action_was_valid"])
                if terminated or truncated:
                    break

    def test_rejects_wrong_channels_and_continuous_actions(self):
        with self.assertRaises(ValueError):
            TwoPathActorCritic(num_actions=9, width=3, height=3, num_channels=6)
        with self.assertRaises(ValueError):
            make_agent(has_continuous_action_space=True)
        model = make_model(3, 2)
        with self.assertRaises(ValueError):
            model.act(torch.zeros(1, layout.NUM_CHANNELS, 5, 5))


class PPOTests(unittest.TestCase):
    def test_gae_bootstraps_non_terminal_rollout_boundary(self):
        agent = make_agent(num_envs=1, rollout_steps=1)
        rewards = torch.tensor([[0.0]])
        values = torch.tensor([[1.0]])
        returns, advantages = agent.compute_gae(
            rewards, values, torch.tensor([[False]]), last_state_values=torch.tensor([2.0])
        )
        self.assertAlmostEqual(float(advantages[0, 0]), 0.8, places=6)
        self.assertAlmostEqual(float(returns[0, 0]), 1.8, places=6)

        terminal_returns, terminal_advantages = agent.compute_gae(
            rewards, values, torch.tensor([[True]]), last_state_values=torch.tensor([2.0])
        )
        self.assertAlmostEqual(float(terminal_advantages[0, 0]), -1.0, places=6)
        self.assertAlmostEqual(float(terminal_returns[0, 0]), 0.0, places=6)

    def test_gae_does_not_leak_across_episode_boundaries(self):
        agent = make_agent(num_envs=1, rollout_steps=2)
        returns, advantages = agent.compute_gae(
            rewards=torch.tensor([[1.0], [1.0]]),
            state_values=torch.zeros(2, 1),
            is_terminals=torch.tensor([[True], [False]]),
            last_state_values=torch.tensor([0.0]),
        )
        self.assertAlmostEqual(float(advantages[0, 0]), 1.0, places=6)
        self.assertAlmostEqual(float(advantages[1, 0]), 1.0, places=6)
        torch.testing.assert_close(returns, advantages)

    def test_update_trains_only_on_valid_transitions(self):
        torch.manual_seed(3)
        # Tiny grids finish quickly, so the rollout contains reset steps.
        envs = [BeamOptimizationEnv(width=2, height=2, density=0.5) for _ in range(2)]
        agent = make_agent(width=2, height=2, num_envs=2, rollout_steps=24)
        last_states = collect_rollout(agent, envs, 24)
        invalid = int((~agent.buffer.valid[:24]).sum())
        self.assertGreater(invalid, 0)
        metrics = agent.update(agent.estimate_values(last_states))
        self.assertEqual(agent.buffer.ptr, 0)
        self.assertEqual(metrics["samples"], 48 - invalid)
        for key in (
            "actor_loss", "critic_loss", "entropy", "grad_norm", "approx_kl",
            "clip_fraction", "value_offset",
        ):
            self.assertTrue(np.isfinite(metrics[key]), key)
        self.assertEqual(agent.updates_done, 1)

    def test_store_defaults_to_valid_and_rejects_overflow(self):
        agent = make_agent(num_envs=2, rollout_steps=1)
        states = torch.zeros(2, layout.NUM_CHANNELS, 3, 4)
        actions, logprobs, values = agent.select_action(states)
        agent.buffer.store(states, actions, logprobs, torch.zeros(2), values, torch.zeros(2, dtype=torch.bool))
        self.assertTrue(bool(agent.buffer.valid[0].all()))
        with self.assertRaises(IndexError):
            agent.buffer.store(states, actions, logprobs, torch.zeros(2), values, torch.zeros(2, dtype=torch.bool))

    def test_schedules_decay_learning_rate_and_entropy(self):
        envs = [BeamOptimizationEnv(width=3, height=2, density=0.5) for _ in range(2)]
        agent = make_agent(total_updates=4, entropy_coef=0.01, entropy_coef_final=0.002)
        seen_lr, seen_entropy = [], []
        for _ in range(4):
            last_states = collect_rollout(agent, envs, 4)
            metrics = agent.update(agent.estimate_values(last_states))
            seen_lr.append(metrics["learning_rate"])
            seen_entropy.append(metrics["entropy_coef"])
        np.testing.assert_allclose(seen_lr, [3e-4, 2.25e-4, 1.5e-4, 0.75e-4], rtol=1e-6)
        np.testing.assert_allclose(seen_entropy, [0.01, 0.008, 0.006, 0.004], rtol=1e-6)
        self.assertAlmostEqual(agent.entropy_coef, 0.002, places=9)

        constant = make_agent()
        self.assertEqual(constant.entropy_coef, 0.01)

    def test_target_kl_stops_an_update_early(self):
        torch.manual_seed(0)
        envs = [BeamOptimizationEnv(width=3, height=2, density=0.5) for _ in range(2)]
        agent = make_agent(target_kl=1e-9, K_epochs=8, lr_actor=1e-2, rollout_steps=8)
        last_states = collect_rollout(agent, envs, 8)
        metrics = agent.update(agent.estimate_values(last_states))
        self.assertTrue(metrics["stopped_early"])
        self.assertLess(metrics["gradient_steps"], 8 * 4)

    def test_checkpoint_restores_policy_optimizer_and_schedule(self):
        envs = [BeamOptimizationEnv(width=3, height=2, density=0.5) for _ in range(2)]
        agent = make_agent(total_updates=10)
        last_states = collect_rollout(agent, envs, 4)
        agent.update(agent.estimate_values(last_states))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pth"
            agent.save(checkpoint)
            restored = make_agent(total_updates=10)
            restored.load(checkpoint)
            for expected, actual in zip(
                agent.policy_old.parameters(), restored.policy_old.parameters()
            ):
                torch.testing.assert_close(expected, actual)
            self.assertEqual(restored.updates_done, 1)
            self.assertAlmostEqual(
                restored.optimizer.param_groups[0]["lr"], agent.optimizer.param_groups[0]["lr"]
            )

            # The network has no size-dependent weights, so the same file
            # loads into an agent built for a different grid.
            larger = make_agent(width=7, height=5)
            larger.load(checkpoint, weights_only=True)
            states = torch.zeros(1, layout.NUM_CHANNELS, 6, 8)
            larger.select_action(states)

            mismatched = make_agent(use_two_path=False)
            with self.assertRaises(ValueError):
                mismatched.load(checkpoint)

    def test_checkpoints_from_the_old_layout_are_rejected_clearly(self):
        agent = make_agent()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "old.pth"
            torch.save({"policy_state_dict": {}, "architecture": "two_path"}, checkpoint)
            with self.assertRaisesRegex(ValueError, "older observation and action layout"):
                agent.load(checkpoint)


if __name__ == "__main__":
    unittest.main()
