"""Tests for matched LSTM/ESN recurrent SAC infrastructure."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from recurrent_sac import (
    EpisodeSequenceReplayBuffer,
    RecurrentSACAgent,
    RecurrentSACConfig,
)


def sample_episode(length: int = 12):
    observations = [np.array([index / 10, -index / 20], dtype=np.float32) for index in range(length + 1)]
    actions = [np.array([0.01, -0.01], dtype=np.float32) for _ in range(length)]
    rewards = [0.1] * length
    terminals = [False] * (length - 1) + [True]
    return observations, actions, rewards, terminals


class SequenceReplayTests(unittest.TestCase):
    def test_samples_contiguous_boundary_safe_sequences(self) -> None:
        replay = EpisodeSequenceReplayBuffer(capacity=100, seed=3)
        replay.add_episode(*sample_episode(12))
        batch = replay.sample(batch_size=4, burn_in=3, sequence_length=5)
        self.assertEqual(batch["observations"].shape, (4, 9, 2))
        self.assertEqual(batch["actions"].shape, (4, 8, 2))
        differences = np.diff(batch["observations"][:, :, 0], axis=1)
        np.testing.assert_allclose(differences, 0.1, atol=1e-6)


class RecurrentAgentTests(unittest.TestCase):
    def make_agent(self, encoder_type: str) -> RecurrentSACAgent:
        return RecurrentSACAgent(
            RecurrentSACConfig(
                observation_size=2,
                action_size=2,
                action_limit=0.05,
                encoder_type=encoder_type,
                recurrent_units=8,
                actor_hidden_sizes=(8,),
                critic_hidden_sizes=(8,),
                seed=7,
            )
        )

    def make_batch(self):
        replay = EpisodeSequenceReplayBuffer(capacity=100, seed=2)
        replay.add_episode(*sample_episode(12))
        return replay.sample(batch_size=2, burn_in=2, sequence_length=4)

    def test_reset_replays_same_esn_action(self) -> None:
        agent = self.make_agent("esn")
        observation = np.array([0.4, -0.2], dtype=np.float32)
        first, _ = agent.predict(observation, deterministic=True)
        agent.predict(np.array([-0.9, 0.7], dtype=np.float32), deterministic=True)
        agent.reset()
        replayed, _ = agent.predict(observation, deterministic=True)
        np.testing.assert_allclose(first, replayed, atol=1e-7)

    def test_lstm_and_esn_complete_sac_update(self) -> None:
        batch = self.make_batch()
        for encoder_type in ("lstm", "esn"):
            with self.subTest(encoder_type=encoder_type):
                agent = self.make_agent(encoder_type)
                losses = agent.update(batch, burn_in=2)
                self.assertEqual(
                    set(losses),
                    {"critic_loss", "actor_loss", "temperature_loss", "temperature"},
                )
                self.assertTrue(all(np.isfinite(value) for value in losses.values()))

    def test_esn_has_fewer_trainable_parameters_than_lstm(self) -> None:
        lstm = self.make_agent("lstm").parameter_counts()
        esn = self.make_agent("esn").parameter_counts()
        self.assertLess(esn["trainable_parameters"], lstm["trainable_parameters"])
        self.assertGreater(esn["fixed_encoder_parameters"], 0)

    def test_checkpoint_round_trip_preserves_policy(self) -> None:
        agent = self.make_agent("lstm")
        observation = np.array([0.25, -0.75], dtype=np.float32)
        expected, _ = agent.predict(observation, deterministic=True)
        with tempfile.TemporaryDirectory() as directory:
            path = agent.save(Path(directory) / "agent.pt")
            loaded = RecurrentSACAgent.load(path)
        actual, _ = loaded.predict(observation, deterministic=True)
        np.testing.assert_allclose(expected, actual, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
