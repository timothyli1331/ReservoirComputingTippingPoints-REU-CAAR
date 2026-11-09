"""Focused tests for the stateful Duffing reservoir policy."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from agentrc import (
    DuffingReservoirPolicy,
    ReservoirPolicy,
    build_agent,
    train_cem,
)


class _Space:
    def __init__(self, shape, high=None) -> None:
        self.shape = shape
        self.high = high


class _TinyEnv:
    """Minimal continuous environment for a fast CEM compatibility check."""

    observation_space = _Space((2,))
    action_space = _Space((2,), np.array([0.05, 0.05], dtype=np.float32))

    def __init__(self) -> None:
        self.state = np.zeros(2, dtype=float)
        self.steps = 0

    def reset(self, *, seed=None):
        del seed
        self.state.fill(0.0)
        self.steps = 0
        return self.state.astype(np.float32), {}

    def step(self, action):
        self.state += np.asarray(action, dtype=float)
        self.steps += 1
        reward = -float(np.linalg.norm(self.state))
        return self.state.astype(np.float32), reward, False, self.steps >= 3, {}


class DuffingReservoirPolicyTests(unittest.TestCase):
    def make_policy(self) -> DuffingReservoirPolicy:
        policy = DuffingReservoirPolicy(
            observation_size=2,
            action_limit=0.05,
            units=4,
            dt=0.04,
            substeps=2,
            seed=3,
        )
        # Remove autonomous bias so the memory test isolates input history.
        policy.input_bias.fill(0.0)
        return policy

    def test_state_persists_after_input_and_reset_clears_it(self) -> None:
        policy = self.make_policy()
        policy.predict(np.array([1.0, -0.5]))
        state_after_pulse = policy.reservoir_state
        policy.predict(np.zeros(2))
        state_after_input_removed = policy.reservoir_state

        self.assertGreater(np.linalg.norm(state_after_pulse), 0.0)
        self.assertGreater(np.linalg.norm(state_after_input_removed), 0.0)

        policy.reset()
        np.testing.assert_array_equal(policy.reservoir_state, np.zeros(8))

    def test_same_observation_has_different_action_after_different_history(self) -> None:
        policy = self.make_policy()
        policy.readout_weights.fill(0.0)
        first_position_feature = 1 + policy.observation_size
        policy.readout_weights[first_position_feature, 0] = 50.0

        policy.predict(np.array([1.0, -0.5]))
        action_with_history, _ = policy.predict(np.zeros(2))

        policy.reset()
        action_without_history, _ = policy.predict(np.zeros(2))

        self.assertGreater(
            abs(float(action_with_history[0] - action_without_history[0])),
            1e-7,
        )

    def test_reset_replays_the_same_trajectory(self) -> None:
        policy = self.make_policy()
        observation = np.array([0.7, -0.2])
        first_action, _ = policy.predict(observation)
        first_state = policy.reservoir_state

        policy.predict(np.array([-0.4, 0.8]))
        policy.reset()
        replay_action, _ = policy.predict(observation)

        np.testing.assert_allclose(policy.reservoir_state, first_state)
        np.testing.assert_allclose(replay_action, first_action)

    def test_base_loader_accepts_saved_duffing_policy(self) -> None:
        policy = self.make_policy()
        policy.predict(np.array([0.2, 0.1]))
        with tempfile.TemporaryDirectory() as directory:
            path = policy.save(Path(directory) / "duffing_policy")
            loaded = ReservoirPolicy.load(path)

        self.assertIsInstance(loaded, DuffingReservoirPolicy)
        np.testing.assert_allclose(loaded.reservoir_state, policy.reservoir_state)

    def test_build_agent_and_cem_accept_duffing_backend(self) -> None:
        env = _TinyEnv()
        policy = build_agent(env, units=3, seed=1, backend="duffing")
        history = train_cem(
            policy,
            env,
            generations=1,
            population_size=2,
            episodes_per_candidate=1,
            validation_episodes=1,
            seed=1,
        )

        self.assertIsInstance(policy, DuffingReservoirPolicy)
        self.assertEqual(policy.readout_weights.shape, (9, 2))
        self.assertEqual(len(history), 1)
        self.assertIn("validation_return", history[0])


if __name__ == "__main__":
    unittest.main()
