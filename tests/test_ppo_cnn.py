import sys
import tempfile
import unittest
from pathlib import Path

import torch


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from models.ppo_cnn import PPO, TwoPathActorCritic  # noqa: E402


class FullyConvolutionalPolicyTests(unittest.TestCase):
    def make_model(self, grid_size):
        observation_size = grid_size + 1
        return TwoPathActorCritic(
            num_actions=2 * grid_size * grid_size,
            width=observation_size,
            height=observation_size,
            has_continuous_action_space=False,
            action_std_init=0.6,
        )

    def test_parameter_count_is_independent_of_resolution(self):
        small = self.make_model(6)
        large = self.make_model(50)
        small_count = sum(parameter.numel() for parameter in small.parameters())
        large_count = sum(parameter.numel() for parameter in large.parameters())
        self.assertEqual(small_count, large_count)
        self.assertLess(large_count, 4_000_000)
        self.assertGreaterEqual(large.receptive_field, 51)

    def test_actor_emits_exactly_two_actions_per_physical_cell(self):
        model = TwoPathActorCritic(
            num_actions=12,
            width=4,  # observation height for a 3x2 physical grid
            height=3,
            has_continuous_action_space=False,
            action_std_init=0.6,
        )
        states = torch.zeros(5, 5, 4, 3)
        states[:, 0, :3, :2] = 0.4
        features = model.forward_features(states)
        logits = model.action_output(features)
        self.assertEqual(logits.shape, (5, 12))
        actions, logprobs, values = model.act(states)
        self.assertEqual(actions.shape, (5,))
        self.assertEqual(logprobs.shape, (5,))
        self.assertEqual(values.shape, (5,))

    def test_policy_assigns_zero_probability_to_saturated_edits(self):
        model = TwoPathActorCritic(
            num_actions=12,
            width=4,
            height=3,
            has_continuous_action_space=False,
            action_std_init=0.6,
        )
        states = torch.zeros(1, 5, 4, 3)
        states[:, 0, :3, :2] = 0.4
        states[:, 0, 0, 0] = 0.001  # cell 0 was assigned void
        states[:, 0, 0, 1] = 1.0  # cell 1 was assigned solid
        logits = torch.zeros(1, 12)
        logits[0, 0] = 100.0
        logits[0, 7] = 100.0

        distribution = model.distribution(logits, states)
        self.assertEqual(float(distribution.probs[0, 0]), 0.0)
        self.assertEqual(float(distribution.probs[0, 1]), 0.0)
        self.assertEqual(float(distribution.probs[0, 6]), 0.0)
        self.assertEqual(float(distribution.probs[0, 7]), 0.0)
        samples = distribution.sample((256,))
        self.assertFalse(bool((samples == 0).any()))
        self.assertFalse(bool((samples == 7).any()))

        deterministic_action, _, _ = model.act(states, deterministic=True)
        mask = model.valid_action_mask(states)
        self.assertTrue(bool(mask[0, deterministic_action.item()]))

        budget_filled = states.clone()
        budget_filled[:, 0, 1, 0] = 1.0
        solid_logits = torch.zeros(1, 12)
        solid_logits[:, 6:] = 100.0
        budget_distribution = model.distribution(solid_logits, budget_filled)
        self.assertEqual(float(budget_distribution.probs[:, 6:].sum()), 0.0)
        self.assertAlmostEqual(
            float(budget_distribution.probs[:, :6].sum()), 1.0, places=6
        )

    def test_policy_mask_alternates_refinement_removal_and_addition(self):
        model = TwoPathActorCritic(
            num_actions=12,
            width=4,
            height=3,
            has_continuous_action_space=False,
            action_std_init=0.6,
            num_channels=6,
        )
        total_cells = 6
        states = torch.zeros(1, 6, 4, 3)
        states[:, 0, :3, :2] = 0.001
        states[:, 0, 0, 0] = 1.0
        states[:, 0, 0, 1] = 1.0

        removal_mask = model.valid_action_mask(states)
        expected_removals = torch.zeros_like(removal_mask)
        expected_removals[0, 0:2] = True
        torch.testing.assert_close(removal_mask, expected_removals)

        # Cell 0 has been selected for removal and is marked in channel 5.
        states[:, 0, 0, 0] = 0.001
        states[:, 5, 0, 0] = 1.0
        addition_mask = model.valid_action_mask(states)
        self.assertFalse(bool(addition_mask[:, :total_cells].any()))
        self.assertFalse(bool(addition_mask[0, total_cells]))
        self.assertFalse(bool(addition_mask[0, total_cells + 1]))
        self.assertEqual(int(addition_mask.sum()), total_cells - 2)


class PPOCorrectnessTests(unittest.TestCase):
    def make_agent(self, num_envs=2, rollout_steps=4):
        return PPO(
            state_channels=5,
            width=4,
            height=3,
            action_dim=12,
            lr_actor=3e-4,
            lr_critic=1e-3,
            gamma=0.9,
            K_epochs=1,
            eps_clip=0.2,
            has_continuous_action_space=False,
            gae_lambda=1.0,
            num_envs=num_envs,
            device="cpu",
            use_two_path=True,
            rollout_steps=rollout_steps,
            minibatch_size=4,
        )

    def test_gae_bootstraps_non_terminal_rollout_boundary(self):
        agent = self.make_agent(num_envs=1, rollout_steps=1)
        rewards = torch.tensor([[0.0]])
        values = torch.tensor([[1.0]])
        returns, advantages = agent.compute_gae(
            rewards,
            values,
            torch.tensor([[False]]),
            last_state_values=torch.tensor([2.0]),
        )
        self.assertAlmostEqual(float(advantages[0, 0]), 0.8, places=6)
        self.assertAlmostEqual(float(returns[0, 0]), 1.8, places=6)

        terminal_returns, terminal_advantages = agent.compute_gae(
            rewards,
            values,
            torch.tensor([[True]]),
            last_state_values=torch.tensor([2.0]),
        )
        self.assertAlmostEqual(float(terminal_advantages[0, 0]), -1.0, places=6)
        self.assertAlmostEqual(float(terminal_returns[0, 0]), 0.0, places=6)

    def test_gae_does_not_leak_across_episode_boundaries(self):
        agent = self.make_agent(num_envs=1, rollout_steps=2)
        returns, advantages = agent.compute_gae(
            rewards=torch.tensor([[1.0], [1.0]]),
            state_values=torch.zeros(2, 1),
            is_terminals=torch.tensor([[True], [False]]),
            last_state_values=torch.tensor([0.0]),
        )
        self.assertAlmostEqual(float(advantages[0, 0]), 1.0, places=6)
        self.assertAlmostEqual(float(advantages[1, 0]), 1.0, places=6)
        torch.testing.assert_close(returns, advantages)

    def test_minibatched_update_clears_rollout_and_returns_metrics(self):
        torch.manual_seed(3)
        agent = self.make_agent()
        states = torch.rand(2, 5, 4, 3)
        for step in range(4):
            actions, logprobs, values = agent.select_action(states)
            agent.buffer.store(
                states,
                actions,
                logprobs,
                torch.tensor([0.2, -0.1]),
                values,
                torch.tensor([False, step == 3]),
            )
            states = torch.rand(2, 5, 4, 3)
        last_values = agent.estimate_values(states)
        metrics = agent.update(last_values)
        self.assertEqual(agent.buffer.ptr, 0)
        self.assertEqual(metrics["samples"], 8)
        for key in ("actor_loss", "critic_loss", "entropy", "grad_norm"):
            self.assertTrue(torch.isfinite(torch.tensor(metrics[key])), key)

    def test_checkpoint_restores_policy_optimizer_and_scheduler(self):
        agent = self.make_agent()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pth"
            agent.save(checkpoint)
            restored = self.make_agent()
            restored.load(checkpoint)
            for expected, actual in zip(
                agent.policy_old.parameters(), restored.policy_old.parameters()
            ):
                torch.testing.assert_close(expected, actual)


if __name__ == "__main__":
    unittest.main()
