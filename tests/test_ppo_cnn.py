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
        features = model.forward_features(states)
        logits = model.action_output(features)
        self.assertEqual(logits.shape, (5, 12))
        actions, logprobs, values = model.act(states)
        self.assertEqual(actions.shape, (5,))
        self.assertEqual(logprobs.shape, (5,))
        self.assertEqual(values.shape, (5,))


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
