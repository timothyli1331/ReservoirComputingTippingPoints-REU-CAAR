"""Gymnasium environment that samples from several double-well systems."""

from __future__ import annotations

from typing import Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from boundary import GlobalBoundary
from systems import (
    DEFAULT_SYSTEM_NAMES,
    SYSTEMS,
    DoubleWellSystem,
    basin_mixing_score,
    derivative,
    finite_time_basin_label,
    get_system,
    local_stretching_rate,
    parse_system_names,
    saddle_stable_eigenvalue,
    step_system,
    velocity_scale,
    walk_system_params,
)


class MultiDoubleWellEnv(gym.Env):
    """Track finite-time basin boundaries across double-well systems.

    The default observation deliberately omits the system name.  It contains
    normalized ``[x, v, local_acceleration, local_stretching_rate]`` by
    default, requiring the policy to use local dynamics rather than a
    model-specific lookup. Acceleration and stretching can each be omitted for
    partial-observation experiments. A one-hot system ID can be enabled for a
    multi-task control experiment, but it is less useful for zero-shot transfer.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        system_names: str | Sequence[str] = DEFAULT_SYSTEM_NAMES,
        selection: str = "random",
        include_system_id: bool = False,
        include_local_acceleration: bool = True,
        include_local_stretching: bool = True,
        parameter_walk: bool = False,
        parameter_walk_interval_steps: int = 1,
        parameter_walk_scale: float = 1.0,
        evolve_dynamics: bool = False,
        dynamics_substeps: int = 1,
        use_basin_mixing: bool = False,
        use_global_boundary: bool = True,
        global_boundary_resolution: int = 41,
        global_boundary_horizon: float = 6.0,
        global_boundary_dt: float = 0.08,
        dt: float = 0.03,
        max_steps: int = 300,
        action_limit: float = 0.05,
        boundary_radius: float = 0.05,
        boundary_samples: int = 6,
        boundary_horizon: float = 2.4,
        boundary_dt: float = 0.06,
        boundary_update_interval: int = 5,
        stochastic_rollouts: int = 2,
        boundary_start_probability: float = 0.6,
    ) -> None:
        super().__init__()
        self.system_names = parse_system_names(system_names)
        if selection not in {"random", "round_robin"}:
            raise ValueError("selection must be 'random' or 'round_robin'")
        self.selection = selection
        self.include_system_id = bool(include_system_id)
        self.include_local_acceleration = bool(include_local_acceleration)
        self.include_local_stretching = bool(include_local_stretching)
        self.parameter_walk = bool(parameter_walk)
        self.parameter_walk_interval_steps = int(parameter_walk_interval_steps)
        self.parameter_walk_scale = float(parameter_walk_scale)
        self.evolve_dynamics = bool(evolve_dynamics)
        self.dynamics_substeps = int(dynamics_substeps)
        self.use_basin_mixing = bool(use_basin_mixing)
        self.use_global_boundary = bool(use_global_boundary)
        self.global_boundary_resolution = int(global_boundary_resolution)
        self.global_boundary_horizon = float(global_boundary_horizon)
        self.global_boundary_dt = float(global_boundary_dt)
        self.dt = float(dt)
        self.max_steps = int(max_steps)
        self.action_limit = float(action_limit)
        self.boundary_radius = float(boundary_radius)
        self.boundary_samples = int(boundary_samples)
        self.boundary_horizon = float(boundary_horizon)
        self.boundary_dt = float(boundary_dt)
        self.boundary_update_interval = int(boundary_update_interval)
        self.stochastic_rollouts = int(stochastic_rollouts)
        self.boundary_start_probability = float(boundary_start_probability)
        if self.boundary_radius <= 0 or self.boundary_samples < 2:
            raise ValueError("boundary_radius must be positive and boundary_samples at least 2")
        if self.boundary_horizon <= 0 or self.boundary_dt <= 0:
            raise ValueError("boundary_horizon and boundary_dt must be positive")
        if self.boundary_update_interval < 1 or self.stochastic_rollouts < 1:
            raise ValueError("update interval and stochastic rollouts must be at least 1")
        if self.parameter_walk_interval_steps < 1:
            raise ValueError("parameter_walk_interval_steps must be at least 1")
        if self.parameter_walk_scale < 0:
            raise ValueError("parameter_walk_scale must be nonnegative")
        if self.dynamics_substeps < 1:
            raise ValueError("dynamics_substeps must be at least 1")
        if not 0.0 <= self.boundary_start_probability <= 1.0:
            raise ValueError("boundary_start_probability must be between 0 and 1")
        self.action_space = spaces.Box(
            -self.action_limit, self.action_limit, shape=(2,), dtype=np.float32
        )
        observation_size = (
            2
            + int(self.include_local_acceleration)
            + int(self.include_local_stretching)
        )
        observation_size += len(SYSTEMS) if self.include_system_id else 0
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(observation_size,), dtype=np.float32
        )
        self.current_system: DoubleWellSystem = get_system(self.system_names[0])
        self.state = np.zeros(2, dtype=float)
        self.time = 0.0
        self.steps = 0
        self.episode_count = 0
        self._basin_mixing = 0.0
        self._diagnostic_rng = np.random.default_rng()
        self._global_boundaries: dict[str, GlobalBoundary] = {}

    def _global_boundary(self) -> GlobalBoundary:
        system = self.current_system
        if system.name not in self._global_boundaries:
            boundary = GlobalBoundary(
                (-1.6 * system.well_location, 1.6 * system.well_location),
                (-1.8 * velocity_scale(system), 1.8 * velocity_scale(system)),
                self.global_boundary_resolution,
            )
            # Noisy trajectories are seeded deterministically so rebuilding a
            # cached Langevin boundary remains reproducible.
            rng = np.random.default_rng(17_071 + list(SYSTEMS).index(system.name))
            boundary.rebuild(
                lambda state: finite_time_basin_label(
                    state,
                    system,
                    horizon=self.global_boundary_horizon,
                    dt=self.global_boundary_dt,
                    start_time=0.0,
                    rng=rng if system.noise_std > 0 else None,
                )
            )
            self._global_boundaries[system.name] = boundary
        return self._global_boundaries[system.name]

    def _global_boundary_distance(self, state: np.ndarray) -> float:
        system = self.current_system
        return self._global_boundary().distance(
            state, scales=(system.well_location, velocity_scale(system))
        )

    def _choose_system(self, requested: str | None = None) -> None:
        if requested is not None:
            if requested not in self.system_names:
                raise ValueError(
                    f"requested system {requested!r} is not enabled: {self.system_names}"
                )
            name = requested
        elif self.selection == "round_robin":
            name = self.system_names[self.episode_count % len(self.system_names)]
        else:
            name = self.system_names[int(self.np_random.integers(len(self.system_names)))]
        self.current_system = get_system(name)

    def walk_parameters(self) -> DoubleWellSystem:
        """Advance the current system's private bounded parameter walk."""
        self.current_system = walk_system_params(
            self.current_system,
            self.np_random,
            scale_multiplier=self.parameter_walk_scale,
        )
        self._global_boundaries.pop(self.current_system.name, None)
        return self.current_system

    def _linear_boundary_distance_physical(self, state: np.ndarray) -> float:
        stable_lambda = saddle_stable_eigenvalue(self.current_system)
        normal = np.array([-stable_lambda, 1.0])
        return float(abs(normal @ state) / np.linalg.norm(normal))

    def _linear_boundary_distance(self, state: np.ndarray) -> float:
        return self._linear_boundary_distance_physical(state) / self.current_system.well_location

    def _update_boundary_diagnostic(self) -> None:
        if not self.use_basin_mixing:
            self._basin_mixing = 0.0
            return
        system = self.current_system
        rollouts = self.stochastic_rollouts if system.noise_std > 0 else 1
        self._basin_mixing = basin_mixing_score(
            self.state,
            system,
            radius=self.boundary_radius * system.well_location,
            samples=self.boundary_samples,
            horizon=self.boundary_horizon,
            dt=self.boundary_dt,
            start_time=self.time,
            rng=self._diagnostic_rng,
            rollouts_per_sample=rollouts,
        )

    def _observation(self) -> np.ndarray:
        system = self.current_system
        values = [
            self.state[0] / system.well_location,
            self.state[1] / velocity_scale(system),
        ]
        if self.include_local_acceleration:
            acceleration_scale = velocity_scale(system) ** 2 / system.well_location
            values.append(
                derivative(self.time, self.state, system)[1] / acceleration_scale
            )
        if self.include_local_stretching:
            values.append(local_stretching_rate(self.state, system))
        if self.include_system_id:
            one_hot = np.zeros(len(SYSTEMS), dtype=float)
            one_hot[list(SYSTEMS).index(system.name)] = 1.0
            values.extend(one_hot)
        return np.asarray(values, dtype=np.float32)

    def _info(self, control_cost: float = 0.0) -> dict:
        linear_distance = self._linear_boundary_distance(self.state)
        global_distance = (
            self._global_boundary_distance(self.state)
            if self.use_global_boundary and not self.use_basin_mixing
            else np.nan
        )
        boundary_distance = (
            1.0 - self._basin_mixing
            if self.use_basin_mixing
            else global_distance if self.use_global_boundary else linear_distance
        )
        return {
            "system_name": self.current_system.name,
            "system_label": self.current_system.label,
            "boundary_distance": boundary_distance,
            "basin_mixing": self._basin_mixing,
            "linear_boundary_distance": linear_distance,
            "linear_boundary_distance_physical": self._linear_boundary_distance_physical(
                self.state
            ),
            "global_boundary_distance": global_distance,
            "local_stretching_rate": local_stretching_rate(
                self.state, self.current_system
            ),
            "local_acceleration": float(
                derivative(self.time, self.state, self.current_system)[1]
            ),
            "control_cost": float(control_cost),
            "boundary_metric": (
                "basin_mixing"
                if self.use_basin_mixing
                else "global_curve" if self.use_global_boundary else "local_tangent"
            ),
            "evolve_dynamics": self.evolve_dynamics,
            "dynamics_substeps": self.dynamics_substeps,
            "parameter_walk": self.parameter_walk,
            "parameter_walk_scale": self.parameter_walk_scale,
            "params": dict(self.current_system.params),
            "noise_std": float(self.current_system.noise_std),
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        self._choose_system(options.get("system_name"))
        if "state" in options:
            state = np.asarray(options["state"], dtype=float)
            if state.shape != (2,):
                raise ValueError("options['state'] must have shape (2,)")
            self.state = state.copy()
        elif self.np_random.random() < self.boundary_start_probability:
            system = self.current_system
            x = self.np_random.uniform(-0.75, 0.75) * system.well_location
            velocity = saddle_stable_eigenvalue(system) * x
            velocity += self.np_random.normal(0.0, 0.12 * velocity_scale(system))
            self.state = np.array([x, velocity], dtype=float)
        else:
            system = self.current_system
            self.state = np.array(
                [
                    self.np_random.uniform(-1.4, 1.4) * system.well_location,
                    self.np_random.uniform(-1.4, 1.4) * velocity_scale(system),
                ],
                dtype=float,
            )
        self.time = 0.0
        self.steps = 0
        diagnostic_seed = int(self.np_random.integers(0, np.iinfo(np.int64).max))
        self._diagnostic_rng = np.random.default_rng(diagnostic_seed)
        if self.use_basin_mixing:
            self._update_boundary_diagnostic()
        else:
            self._basin_mixing = 0.0
        self.episode_count += 1
        return self._observation(), self._info()

    def step(self, action):
        action_array = np.asarray(action, dtype=float)
        if action_array.shape != (2,):
            raise ValueError("action must have shape (2,)")
        clipped = np.clip(action_array, self.action_space.low, self.action_space.high)
        parameters_changed = bool(
            self.parameter_walk
            and (self.steps + 1) % self.parameter_walk_interval_steps == 0
        )
        if parameters_changed:
            self.walk_parameters()
        if self.evolve_dynamics:
            self.state = step_system(
                self.state,
                clipped,
                self.current_system,
                self.dt,
                self.time,
                self.np_random,
                self.action_limit,
                self.dynamics_substeps,
            )
        else:
            # Agent-controlled mode: no drift and no Langevin kick.
            self.state = self.state + clipped
        self.time += self.dt * (self.dynamics_substeps if self.evolve_dynamics else 1)
        self.steps += 1

        if self.use_basin_mixing and self.steps % self.boundary_update_interval == 0:
            self._update_boundary_diagnostic()
        linear_distance = self._linear_boundary_distance(self.state)
        if self.use_basin_mixing:
            distance = 1.0 - self._basin_mixing
            boundary_proximity = self._basin_mixing
        else:
            distance = (
                self._global_boundary_distance(self.state)
                if self.use_global_boundary
                else linear_distance
            )
            boundary_proximity = float(np.exp(-distance / 0.10))
        stretching = local_stretching_rate(self.state, self.current_system)
        control_cost = float(np.sum((clipped / self.action_limit) ** 2))
        reward = float(
            boundary_proximity
            + 0.10 * np.tanh(max(stretching, 0.0))
            - 0.02 * control_cost
        )
        terminated = bool(
            np.any(np.abs(self.state) > self.current_system.state_limit)
            or not np.all(np.isfinite(self.state))
        )
        truncated = self.steps >= self.max_steps
        info = self._info(control_cost)
        info["parameters_changed"] = parameters_changed
        return self._observation(), reward, terminated, truncated, info
