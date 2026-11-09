"""Focused tests for the fixed-reservoir SAC implementation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from agentrc import DuffingReservoirPolicy
from reservoir_sac import ReplayBuffer, ReservoirSACAgent


def make_encoder(seed: int = 3) -> DuffingReservoirPolicy:
    return DuffingReservoirPolicy(
        observation_size=2,
        action_limit=0.05,
        units=4,
        dt=0.04,
        substeps=2,
        seed=seed,
    )


def make_agent(
    seed: int = 3, memory_mode: str = "persistent"
) -> ReservoirSACAgent:
    return ReservoirSACAgent(
        encoder=make_encoder(seed),
        action_size=2,
        action_limit=0.05,
        hidden_sizes=(16, 16),
        memory_mode=memory_mode,
        seed=seed,
    )


class ReplayBufferTests(unittest.TestCase):
    def test_circular_storage_keeps_aligned_transitions(self) -> None:
        replay = ReplayBuffer(capacity=3, feature_size=2, action_size=1, seed=1)
        for index in range(4):
            replay.add(
                features=[index, index + 0.25],
                action=[-index],
                reward=float(index),
                next_features=[index + 1, index + 1.25],
                terminal=index % 2 == 0,
            )

        self.assertEqual(len(replay), 3)
        for row in range(3):
            index = int(replay.rewards[row, 0])
            np.testing.assert_allclose(
                replay.next_features[row], replay.features[row] + 1.0
            )
            self.assertEqual(replay.actions[row, 0], -index)
            self.assertEqual(replay.terminals[row, 0], float(index % 2 == 0))

    def test_sampling_returns_expected_shapes(self) -> None:
        replay = ReplayBuffer(capacity=10, feature_size=3, action_size=2, seed=2)
        for index in range(6):
            replay.add(np.ones(3) * index, np.zeros(2), index, np.ones(3), False)
        batch = replay.sample(4)

        self.assertEqual(batch["features"].shape, (4, 3))
        self.assertEqual(batch["actions"].shape, (4, 2))
        self.assertEqual(batch["rewards"].shape, (4, 1))
        self.assertEqual(batch["next_features"].shape, (4, 3))
        self.assertEqual(batch["terminals"].shape, (4, 1))


class ReservoirSACAgentTests(unittest.TestCase):
    def test_persistent_encoder_features_depend_on_history(self) -> None:
        agent = make_agent()
        current_observation = np.zeros(2)
        agent.reset()
        agent.encode(np.array([0.8, -0.2]))
        with_history = agent.encode(current_observation)

        agent.reset()
        without_history = agent.encode(current_observation)

        self.assertEqual(with_history.shape, (agent.feature_size,))
        self.assertFalse(np.allclose(with_history, without_history))

    def test_reset_each_step_encoder_removes_history(self) -> None:
        agent = make_agent(memory_mode="reset_each_step")
        current_observation = np.zeros(2)
        agent.reset()
        agent.encode(np.array([0.8, -0.2]))
        after_history = agent.encode(current_observation)

        agent.reset()
        without_history = agent.encode(current_observation)

        np.testing.assert_allclose(after_history, without_history)

    def test_one_update_changes_actor_and_returns_finite_metrics(self) -> None:
        agent = make_agent()
        replay = ReplayBuffer(64, agent.feature_size, agent.action_size, seed=4)
        rng = np.random.default_rng(5)
        for _ in range(64):
            features = rng.normal(size=agent.feature_size).astype(np.float32)
            next_features = rng.normal(size=agent.feature_size).astype(np.float32)
            action = rng.uniform(-0.05, 0.05, size=2).astype(np.float32)
            replay.add(features, action, rng.normal(), next_features, False)

        before = agent.actor.mean.weight.detach().clone()
        metrics = agent.update(replay.sample(32))
        after = agent.actor.mean.weight.detach()

        self.assertFalse(torch.equal(before, after))
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
        self.assertEqual(agent.update_steps, 1)

    def test_deterministic_action_stays_within_environment_limit(self) -> None:
        agent = make_agent()
        agent.reset()
        features = agent.encode(np.array([0.2, -0.4]))
        action = agent.act_features(features, deterministic=True)

        self.assertEqual(action.shape, (2,))
        self.assertTrue(np.all(np.abs(action) <= 0.05))

    def test_save_load_reproduces_features_and_action(self) -> None:
        agent = make_agent(seed=7, memory_mode="reset_each_step")
        observation = np.array([0.3, -0.6])
        agent.reset()
        expected_features = agent.encode(observation)
        expected_action = agent.act_features(expected_features, deterministic=True)

        with tempfile.TemporaryDirectory() as directory:
            path = agent.save(Path(directory) / "reservoir_sac")
            loaded = ReservoirSACAgent.load(path)

        loaded.reset()
        actual_features = loaded.encode(observation)
        actual_action = loaded.act_features(actual_features, deterministic=True)
        self.assertEqual(loaded.memory_mode, "reset_each_step")
        np.testing.assert_allclose(actual_features, expected_features)
        np.testing.assert_allclose(actual_action, expected_action)


if __name__ == "__main__":
    unittest.main()
