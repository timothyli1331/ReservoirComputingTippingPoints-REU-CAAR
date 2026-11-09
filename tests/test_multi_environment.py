"""Tests for configurable observations in the multi-system environment."""

from __future__ import annotations

import unittest

import numpy as np

from multi_environment import MultiDoubleWellEnv
from systems import integrate_step


class MultiEnvironmentObservationTests(unittest.TestCase):
    def test_acceleration_and_stretching_can_be_removed_independently(self) -> None:
        cases = (
            (True, True, 4),
            (True, False, 3),
            (False, True, 3),
            (False, False, 2),
        )
        for include_acceleration, include_stretching, expected_size in cases:
            with self.subTest(
                acceleration=include_acceleration,
                stretching=include_stretching,
            ):
                env = MultiDoubleWellEnv(
                    include_local_acceleration=include_acceleration,
                    include_local_stretching=include_stretching,
                    use_global_boundary=False,
                )
                observation, _ = env.reset(seed=7)
                self.assertEqual(env.observation_space.shape, (expected_size,))
                self.assertEqual(observation.shape, (expected_size,))
                env.close()

    def test_multiple_dynamics_substeps_apply_action_only_once(self) -> None:
        env = MultiDoubleWellEnv(
            system_names=("duffing",),
            evolve_dynamics=True,
            dynamics_substeps=3,
            parameter_walk=False,
            use_global_boundary=False,
        )
        initial = np.array([0.4, -0.1])
        action = np.array([0.02, 0.01])
        env.reset(seed=11, options={"state": initial})
        expected = initial + action
        for substep in range(3):
            expected = integrate_step(
                expected,
                env.current_system,
                env.dt,
                substep * env.dt,
                env.np_random,
            )

        _, _, _, _, info = env.step(action)

        np.testing.assert_allclose(env.state, expected)
        self.assertAlmostEqual(env.time, 3 * env.dt)
        self.assertEqual(info["dynamics_substeps"], 3)
        env.close()


if __name__ == "__main__":
    unittest.main()
