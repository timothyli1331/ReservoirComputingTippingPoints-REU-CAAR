"""Continuous-control environment for learning to track a basin boundary."""

from __future__ import annotations

from collections.abc import Mapping

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from boundary import GlobalBoundary
from dynamics import (
    duffing_derivative,
    step_system,
    walk_duffing_params,
)
from indicators import basin_label, basin_sensitivity, local_stretching_rate


DEFAULT_PARAMS = {
    "alpha": -1.0,
    "beta": 1.0,
    "damping": 0.2,
    "gamma": 0.3,
    "omega": 1.2,
    "max_action": 0.05,
}

# Per-action scales have approximately the same accumulated diffusion over 50
# steps as one call to the original, coarser evaluation-time parameter walk.
PER_ACTION_WALK_SCALES = {
    "alpha": 0.025 / np.sqrt(50.0),
    "beta": 0.025 / np.sqrt(50.0),
    "damping": 0.005 / np.sqrt(50.0),
    "gamma": 0.005 / np.sqrt(50.0),
    "omega": 0.005 / np.sqrt(50.0),
}


class DuffingBoundaryEnv(gym.Env):
    """Move in phase space while remaining close to the basin separatrix.

    The reward numerically probes nearby trajectories at the current forcing
    phase. It therefore detects curved, moving finite-time basin boundaries
    instead of assuming the boundary is a fixed line through the origin.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        params: Mapping[str, float] | None = None,
        dt: float = 0.03,
        max_steps: int = 300,
        state_limit: float = 2.5,
        include_basin_sensitivity: bool = False,
        parameter_walk: bool = False,
        parameter_walk_interval_steps: int = 1,
        parameter_walk_interval_episodes: int | None = None,
        evolve_dynamics: bool = False,
        use_basin_mixing: bool = False,
        use_global_boundary: bool = True,
        global_boundary_resolution: int = 41,
        global_boundary_horizon: float = 6.0,
        global_boundary_dt: float = 0.08,
        boundary_radius: float = 0.05,
        boundary_samples: int = 6,
        boundary_horizon: float = 2.4,
        boundary_dt: float = 0.06,
        boundary_update_interval: int = 5,
        boundary_start_probability: float = 0.6,
    ) -> None:
        super().__init__()
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.dt = float(dt)
        self.max_steps = int(max_steps)
        self.state_limit = float(state_limit)
        self.include_basin_sensitivity = include_basin_sensitivity
        self.parameter_walk_interval_steps = int(parameter_walk_interval_steps)
        self.parameter_walk_interval_episodes = (
            None
            if parameter_walk_interval_episodes is None
            else int(parameter_walk_interval_episodes)
        )
        self.evolve_dynamics = bool(evolve_dynamics)
        self.use_basin_mixing = bool(use_basin_mixing)
        self.use_global_boundary = bool(use_global_boundary)
        self.global_boundary_horizon = float(global_boundary_horizon)
        self.global_boundary_dt = float(global_boundary_dt)
        self.boundary_radius = float(boundary_radius)
        self.boundary_samples = int(boundary_samples)
        self.boundary_horizon = float(boundary_horizon)
        self.boundary_dt = float(boundary_dt)
        self.boundary_update_interval = int(boundary_update_interval)
        self.boundary_start_probability = float(boundary_start_probability)
        if self.boundary_radius <= 0 or self.boundary_samples < 2:
            raise ValueError("boundary_radius must be positive and boundary_samples at least 2")
        if self.boundary_horizon <= 0 or self.boundary_dt <= 0:
            raise ValueError("boundary_horizon and boundary_dt must be positive")
        if self.boundary_update_interval < 1:
            raise ValueError("boundary_update_interval must be at least 1")
        if self.parameter_walk_interval_steps < 1:
            raise ValueError("parameter_walk_interval_steps must be at least 1")
        if (
            self.parameter_walk_interval_episodes is not None
            and self.parameter_walk_interval_episodes < 1
        ):
            raise ValueError("parameter_walk_interval_episodes must be at least 1")
        if not 0.0 <= self.boundary_start_probability <= 1.0:
            raise ValueError("boundary_start_probability must be between 0 and 1")
        action_limit = float(self.params["max_action"])
        self.action_space = spaces.Box(-action_limit, action_limit, (2,), np.float32)
        # Position, velocity, forcing phase, local acceleration, and stretching.
        obs_size = 7 if include_basin_sensitivity else 6
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_size,), np.float32)
        self.state = np.zeros(2, dtype=float)
        self.parameter_walk = parameter_walk
        self.episode_count = 0
        self.time = 0.0
        self.forcing_phase = 0.0
        self.steps = 0
        self._basin_sensitivity = 0.0
        self.global_boundary = GlobalBoundary(
            (-1.6, 1.6), (-1.8, 1.8), global_boundary_resolution
        )
        self._global_boundary_resolution = int(global_boundary_resolution)
        self._global_boundary_cache: dict[
            tuple[tuple[str, float], ...], GlobalBoundary
        ] = {}
        self._global_boundary_ready = False
        self._global_boundary_signature: tuple[tuple[str, float], ...] | None = None

    def _rebuild_global_boundary(self) -> None:
        params = dict(self.params)
        signature = tuple(sorted((name, float(value)) for name, value in params.items()))
        boundary = GlobalBoundary(
            (-1.6, 1.6), (-1.8, 1.8), self._global_boundary_resolution
        )
        boundary.rebuild(
            lambda state: basin_label(
                state,
                params,
                horizon=self.global_boundary_horizon,
                dt=self.global_boundary_dt,
                start_time=self._forcing_time(),
            )
        )
        self.global_boundary = boundary
        if not self.parameter_walk:
            self._global_boundary_cache[signature] = boundary
        self._global_boundary_ready = True
        self._global_boundary_signature = signature

    def _global_boundary_distance(self, state: np.ndarray) -> float:
        return self.global_boundary.distance(state, scales=(1.0, 1.0))

    def _boundary_distance(self, state: np.ndarray) -> float:
        damping = float(self.params["damping"])
        alpha = float(self.params["alpha"])
        stable_lambda = (-damping - np.sqrt(damping**2 - 4.0 * alpha)) / 2.0
        # Stable eigendirection is [1, stable_lambda].
        normal = np.array([-stable_lambda, 1.0])
        return float(abs(normal @ state) / np.linalg.norm(normal))

    def _forcing_time(self) -> float:
        """Return an equivalent time whose forcing angle is continuous."""
        return self.forcing_phase / float(self.params["omega"])

    def _observation(self) -> np.ndarray:
        acceleration = duffing_derivative(
            self._forcing_time(), self.state, self.params
        )[1]
        values = [
            *self.state,
            np.sin(self.forcing_phase),
            np.cos(self.forcing_phase),
            acceleration,
            local_stretching_rate(self.state, self.params),
        ]
        if self.include_basin_sensitivity:
            values.append(self._basin_sensitivity)
        return np.asarray(values, dtype=np.float32)

    def _update_boundary_diagnostic(self) -> None:
        """Estimate finite-time basin mixing near the current state."""
        self._basin_sensitivity = basin_sensitivity(
            self.state,
            self.params,
            radius=self.boundary_radius,
            samples=self.boundary_samples,
            horizon=self.boundary_horizon,
            dt=self.boundary_dt,
            start_time=self._forcing_time(),
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        parameters_changed = (
            self.parameter_walk
            and self.parameter_walk_interval_episodes is not None
            and self.episode_count > 0
            and self.episode_count % self.parameter_walk_interval_episodes == 0
        )
        if parameters_changed:
            walk_duffing_params(self.params, self.np_random)
        options = options or {}
        if "state" in options:
            state = np.asarray(options["state"], dtype=float)
            if state.shape != (2,):
                raise ValueError("options['state'] must have shape (2,)")
            self.state = state.copy()
        elif self.np_random.random() < self.boundary_start_probability:
            x = self.np_random.uniform(-0.8, 0.8)
            stable_lambda = (
                -float(self.params["damping"])
                - np.sqrt(float(self.params["damping"]) ** 2 - 4 * float(self.params["alpha"]))
            ) / 2
            self.state = np.array([x, stable_lambda * x])
            self.state += self.np_random.normal(0.0, 0.12, size=2)
        else:
            # Broad starts force the policy to encounter boundary geometry away
            # from the origin instead of specializing only to the saddle.
            self.state = self.np_random.uniform(
                low=(-1.4, -1.4), high=(1.4, 1.4), size=2
            )
        self.time = 0.0
        self.forcing_phase = 0.0
        self.steps = 0
        current_signature = tuple(
            sorted((name, float(value)) for name, value in self.params.items())
        )
        if self.use_global_boundary and not self.use_basin_mixing and (
            not self._global_boundary_ready
            or self._global_boundary_signature != current_signature
        ):
            cached = self._global_boundary_cache.get(current_signature)
            if cached is not None:
                self.global_boundary = cached
                self._global_boundary_ready = True
                self._global_boundary_signature = current_signature
            else:
                self._rebuild_global_boundary()
        self.episode_count += 1
        self._basin_sensitivity = 0.0
        if self.use_basin_mixing:
            self._update_boundary_diagnostic()
        linear_distance = self._boundary_distance(self.state)
        global_distance = (
            self._global_boundary_distance(self.state)
            if self.use_global_boundary and not self.use_basin_mixing
            else np.nan
        )
        boundary_distance = (
            1.0 - self._basin_sensitivity
            if self.use_basin_mixing
            else global_distance if self.use_global_boundary else linear_distance
        )
        return self._observation(), {
            "boundary_distance": boundary_distance,
            "basin_sensitivity": self._basin_sensitivity,
            "linear_boundary_distance": linear_distance,
            "global_boundary_distance": global_distance,
            "boundary_metric": (
                "basin_mixing"
                if self.use_basin_mixing
                else "global_curve" if self.use_global_boundary else "local_tangent"
            ),
            "parameters_changed": parameters_changed,
            "evolve_dynamics": self.evolve_dynamics,
        }

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (2,):
            raise ValueError("action must have shape (2,)")
        clipped = np.clip(action, self.action_space.low, self.action_space.high)
        parameters_changed = bool(
            self.parameter_walk
            and (self.steps + 1) % self.parameter_walk_interval_steps == 0
        )
        if parameters_changed:
            walk_duffing_params(
                self.params,
                self.np_random,
                step_scales=PER_ACTION_WALK_SCALES,
            )
            self._global_boundary_ready = False
        if self.evolve_dynamics:
            self.state = step_system(
                self.state,
                clipped,
                self.params,
                self.dt,
                self._forcing_time(),
            )
        else:
            # Agent-controlled mode: move only by the chosen phase-space action.
            self.state = self.state + clipped
        self.forcing_phase = float(
            np.mod(
                self.forcing_phase + float(self.params["omega"]) * self.dt,
                2.0 * np.pi,
            )
        )
        self.time += self.dt
        self.steps += 1

        if self.use_basin_mixing and self.steps % self.boundary_update_interval == 0:
            self._update_boundary_diagnostic()
        linear_distance = self._boundary_distance(self.state)
        if self.use_basin_mixing:
            boundary_proximity = self._basin_sensitivity
            distance = 1.0 - boundary_proximity
            global_distance = np.nan
        else:
            if self.use_global_boundary and not self._global_boundary_ready:
                self._rebuild_global_boundary()
            distance = (
                self._global_boundary_distance(self.state)
                if self.use_global_boundary
                else linear_distance
            )
            global_distance = distance if self.use_global_boundary else np.nan
            boundary_proximity = float(np.exp(-distance / 0.10))
        stretching = local_stretching_rate(self.state, self.params)
        local_acceleration = float(
            duffing_derivative(self._forcing_time(), self.state, self.params)[1]
        )
        control_cost = float(np.sum((clipped / float(self.params["max_action"])) ** 2))
        reward = float(
            boundary_proximity
            + 0.10 * np.tanh(max(stretching, 0.0))
            - 0.02 * control_cost
        )
        terminated = bool(
            np.any(np.abs(self.state) > self.state_limit)
            or not np.all(np.isfinite(self.state))
        )
        truncated = self.steps >= self.max_steps
        info = {
            "boundary_distance": distance,
            "basin_sensitivity": self._basin_sensitivity,
            "linear_boundary_distance": linear_distance,
            "global_boundary_distance": global_distance,
            "boundary_metric": (
                "basin_mixing"
                if self.use_basin_mixing
                else "global_curve" if self.use_global_boundary else "local_tangent"
            ),
            "evolve_dynamics": self.evolve_dynamics,
            "parameters_changed": parameters_changed,
            "params": dict(self.params),
            "local_stretching_rate": stretching,
            "local_acceleration": local_acceleration,
            "control_cost": control_cost,
        }
        return self._observation(), reward, terminated, truncated, info
