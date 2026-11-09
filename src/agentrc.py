"""Stateful reservoir policies and gradient-free training utilities."""

from __future__ import annotations

import copy
import pickle
from pathlib import Path
from typing import Any

import numpy as np


class ReservoirPolicy:
    """A fixed ReservoirPy reservoir with an evolvable linear action readout.

    Reservoir weights remain fixed.  Training changes only ``readout_weights``
    using policy search, so no backpropagation or deep-learning framework is
    involved.
    """

    def __init__(
        self,
        observation_size: int,
        action_limit: float,
        units: int = 100,
        spectral_radius: float = 0.9,
        leaking_rate: float = 0.3,
        input_scaling: float = 0.5,
        seed: int = 0,
    ) -> None:
        try:
            from reservoirpy.nodes import Reservoir
        except ImportError as exc:
            raise ImportError(
                "This agent requires ReservoirPy. Install it with "
                "`python -m pip install reservoirpy`."
            ) from exc

        self.observation_size = int(observation_size)
        self.action_limit = float(action_limit)
        self.units = int(units)
        self.seed = int(seed)
        self.reservoir = Reservoir(
            units=self.units,
            sr=spectral_radius,
            lr=leaking_rate,
            input_scaling=input_scaling,
            seed=seed,
        )
        # Bias + normalized raw observation + reservoir state -> two actions.
        self.feature_size = 1 + self.observation_size + self.units
        rng = np.random.default_rng(seed)
        self.readout_weights = rng.normal(
            0.0, 0.05, size=(self.feature_size, 2)
        )

    def reset(self) -> None:
        """Clear recurrent state at the beginning of an episode."""
        # ReservoirPy 0.3.x creates state lazily on the first call to step().
        if hasattr(self.reservoir, "state"):
            self.reservoir.reset()

    def _normalize(self, observation) -> np.ndarray:
        obs = np.asarray(observation, dtype=float).reshape(-1)
        if obs.shape != (self.observation_size,):
            raise ValueError(
                f"expected observation shape ({self.observation_size},), got {obs.shape}"
            )
        return np.tanh(obs)

    def features(self, observation) -> np.ndarray:
        """Advance the fixed reservoir and return its policy feature vector."""
        obs = self._normalize(observation)
        reservoir_state = np.asarray(self.reservoir.step(obs)).reshape(-1)
        return np.concatenate(([1.0], obs, reservoir_state))

    def predict(self, observation, deterministic: bool = True):
        """Return a continuous ``[delta_x, delta_v]`` action."""
        del deterministic  # The trained policy itself is deterministic.
        features = self.features(observation)
        action = self.action_limit * np.tanh(features @ self.readout_weights)
        return action.astype(np.float32), None

    def clone(self) -> "ReservoirPolicy":
        return copy.deepcopy(self)

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        if destination.suffix != ".pkl":
            destination = destination.with_suffix(".pkl")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as file:
            pickle.dump(self, file)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "ReservoirPolicy":
        source = Path(path)
        if source.suffix != ".pkl":
            source = source.with_suffix(".pkl")
        with source.open("rb") as file:
            policy = pickle.load(file)
        if not isinstance(policy, cls):
            raise TypeError(f"{source} does not contain a ReservoirPolicy")
        return policy


class DuffingReservoirPolicy(ReservoirPolicy):
    """A fixed bank of driven Duffing oscillators with a linear readout.

    Each node retains its position and velocity between calls to
    :meth:`predict`, so those physical states provide fading memory of previous
    observations.  CEM changes only ``readout_weights``; the oscillator
    parameters and deterministic input projection remain fixed.
    """

    def __init__(
        self,
        observation_size: int,
        action_limit: float,
        units: int = 48,
        dt: float = 0.04,
        substeps: int = 2,
        alpha: float = -1.0,
        beta: float = 1.0,
        damping: float = 0.25,
        parameter_spread: float = 0.18,
        input_scaling: float = 0.8,
        state_clip: float = 4.0,
        seed: int = 0,
    ) -> None:
        if observation_size < 1 or units < 1:
            raise ValueError("observation_size and units must be positive")
        if action_limit <= 0 or dt <= 0 or substeps < 1:
            raise ValueError("action_limit and dt must be positive and substeps at least 1")
        if beta <= 0 or damping < 0:
            raise ValueError("beta must be positive and damping must be nonnegative")
        if not 0 <= parameter_spread < 1:
            raise ValueError("parameter_spread must be in [0, 1)")
        if input_scaling <= 0 or state_clip <= 0:
            raise ValueError("input_scaling and state_clip must be positive")

        self.observation_size = int(observation_size)
        self.action_limit = float(action_limit)
        self.units = int(units)
        self.seed = int(seed)
        self.dt = float(dt)
        self.substeps = int(substeps)
        self.input_scaling = float(input_scaling)
        self.state_clip = float(state_clip)

        # Parameter diversity supplies a range of nonlinear response and memory
        # timescales without introducing trainable recurrent connections.
        angles = 2.0 * np.pi * (np.arange(self.units) + 0.5) / self.units
        self.alphas = float(alpha) * (
            1.0 + parameter_spread * np.cos(angles)
        )
        self.betas = float(beta) * (
            1.0 + parameter_spread * np.sin(angles + np.pi / 5.0)
        )
        self.dampings = float(damping) * (
            1.0 + parameter_spread * np.cos(angles + np.pi / 3.0)
        )

        node_index = np.arange(1, self.units + 1, dtype=float)[:, None]
        input_index = np.arange(1, self.observation_size + 1, dtype=float)[None, :]
        golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0
        projection = np.sin(node_index * input_index * golden_ratio)
        projection += 0.5 * np.cos(node_index * (input_index + 0.5))
        norms = np.linalg.norm(projection, axis=1, keepdims=True)
        self.input_weights = projection / np.maximum(norms, 1e-12)
        self.input_bias = 0.15 * np.sin(angles)

        self.positions = np.zeros(self.units, dtype=float)
        self.velocities = np.zeros(self.units, dtype=float)

        # Bias + current observation + both physical state variables.  The
        # Duffing dynamics already provide the nonlinear feature expansion.
        self.feature_size = 1 + self.observation_size + 2 * self.units
        rng = np.random.default_rng(seed)
        self.readout_weights = rng.normal(
            0.0, 0.05, size=(self.feature_size, 2)
        )

    def reset(self) -> None:
        """Clear physical memory at the beginning of an episode."""
        self.positions.fill(0.0)
        self.velocities.fill(0.0)

    def _derivative(
        self, positions: np.ndarray, velocities: np.ndarray, drive: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        acceleration = (
            drive
            - self.dampings * velocities
            - self.alphas * positions
            - self.betas * positions**3
        )
        return velocities, acceleration

    def _advance(self, observation: np.ndarray) -> None:
        """Hold one projected input while advancing the oscillator bank."""
        drive = self.input_scaling * np.tanh(
            self.input_weights @ observation + self.input_bias
        )
        h = self.dt / self.substeps
        q = self.positions
        p = self.velocities
        for _ in range(self.substeps):
            k1_q, k1_p = self._derivative(q, p, drive)
            k2_q, k2_p = self._derivative(
                q + 0.5 * h * k1_q, p + 0.5 * h * k1_p, drive
            )
            k3_q, k3_p = self._derivative(
                q + 0.5 * h * k2_q, p + 0.5 * h * k2_p, drive
            )
            k4_q, k4_p = self._derivative(q + h * k3_q, p + h * k3_p, drive)
            q = q + (h / 6.0) * (k1_q + 2.0 * k2_q + 2.0 * k3_q + k4_q)
            p = p + (h / 6.0) * (k1_p + 2.0 * k2_p + 2.0 * k3_p + k4_p)
            q = np.clip(q, -self.state_clip, self.state_clip)
            p = np.clip(p, -self.state_clip, self.state_clip)
        self.positions = q
        self.velocities = p

    @property
    def reservoir_state(self) -> np.ndarray:
        """Return a copy of the current physical memory state."""
        return np.concatenate((self.positions, self.velocities)).copy()

    def features(self, observation) -> np.ndarray:
        """Advance the physical reservoir and return its policy features."""
        obs = self._normalize(observation)
        self._advance(obs)
        return np.concatenate(
            ([1.0], obs, self.positions, self.velocities)
        )

    def predict(self, observation, deterministic: bool = True):
        """Advance the physical reservoir and return a continuous action."""
        del deterministic
        features = self.features(observation)
        action = self.action_limit * np.tanh(features @ self.readout_weights)
        return action.astype(np.float32), None


def build_agent(
    env,
    units: int = 100,
    seed: int = 0,
    backend: str = "reservoirpy",
    **reservoir_kwargs,
):
    """Construct the selected stateful reservoir policy for an environment."""
    policy_classes = {
        "reservoirpy": ReservoirPolicy,
        "duffing": DuffingReservoirPolicy,
    }
    try:
        policy_class = policy_classes[backend.lower()]
    except KeyError as exc:
        choices = ", ".join(sorted(policy_classes))
        raise ValueError(f"unknown reservoir backend {backend!r}; choose from {choices}") from exc
    return policy_class(
        observation_size=int(env.observation_space.shape[0]),
        action_limit=float(env.action_space.high[0]),
        units=units,
        seed=seed,
        **reservoir_kwargs,
    )


def episode_return(policy: ReservoirPolicy, env, seed: int) -> float:
    """Evaluate one complete episode."""
    observation, _ = env.reset(seed=seed)
    policy.reset()
    total = 0.0
    done = False
    while not done:
        action, _ = policy.predict(observation)
        observation, reward, terminated, truncated, _ = env.step(action)
        total += float(reward)
        done = terminated or truncated
    return total


def _snapshot_environment(env) -> dict[str, Any]:
    """Capture mutable curriculum state that must be identical across candidates."""
    snapshot: dict[str, Any] = {}
    if hasattr(env, "params"):
        snapshot["params"] = copy.deepcopy(env.params)
    if hasattr(env, "current_system"):
        snapshot["current_system"] = copy.deepcopy(env.current_system)
    if hasattr(env, "episode_count"):
        snapshot["episode_count"] = int(env.episode_count)
    return snapshot


def _restore_environment(env, snapshot: dict[str, Any]) -> None:
    """Restore the curriculum state captured by :func:`_snapshot_environment`."""
    if "params" in snapshot:
        env.params.clear()
        env.params.update(copy.deepcopy(snapshot["params"]))
    if "current_system" in snapshot:
        env.current_system = copy.deepcopy(snapshot["current_system"])
    if "episode_count" in snapshot:
        env.episode_count = int(snapshot["episode_count"])


def _evaluate_weights(
    policy: ReservoirPolicy,
    env,
    weights: np.ndarray,
    episode_seeds: list[int],
    start_snapshot: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    """Evaluate one readout from a reproducible environment starting point."""
    _restore_environment(env, start_snapshot)
    policy.readout_weights = weights.copy()
    score = float(
        np.mean([episode_return(policy, env, episode_seed) for episode_seed in episode_seeds])
    )
    return score, _snapshot_environment(env)


def train_cem(
    policy: ReservoirPolicy,
    env,
    generations: int = 80,
    population_size: int = 32,
    elite_fraction: float = 0.2,
    episodes_per_candidate: int = 2,
    initial_std: float = 0.15,
    min_std: float = 0.001,
    max_std: float | None = None,
    std_decay: float = 0.95,
    smoothing: float = 0.2,
    mirrored_sampling: bool = True,
    validation_episodes: int = 8,
    validation_interval: int = 1,
    early_stopping_patience: int = 0,
    seed: int = 0,
    callback=None,
) -> list[dict[str, float]]:
    """Optimize readout weights using a validated, smoothed CEM search.

    Candidate policies use common random numbers within a generation.  A
    separate fixed validation suite selects the readout returned to the caller,
    preventing a noisy final generation from overwriting a stronger policy.
    """
    if generations < 1:
        raise ValueError("generations must be at least 1")
    if population_size < 2:
        raise ValueError("population_size must be at least 2")
    if not 0.0 < elite_fraction <= 1.0:
        raise ValueError("elite_fraction must be in (0, 1]")
    if episodes_per_candidate < 1:
        raise ValueError("episodes_per_candidate must be at least 1")
    if initial_std <= 0.0 or min_std < 0.0 or min_std > initial_std:
        raise ValueError("require 0 <= min_std <= initial_std and initial_std > 0")
    if max_std is not None and max_std < initial_std:
        raise ValueError("max_std must be at least initial_std when provided")
    if not 0.0 < std_decay <= 1.0:
        raise ValueError("std_decay must be in (0, 1]")
    if not 0.0 <= smoothing < 1.0:
        raise ValueError("smoothing must be in [0, 1)")
    if validation_episodes < 0 or validation_interval < 1:
        raise ValueError("validation_episodes cannot be negative and interval must be positive")
    if early_stopping_patience < 0:
        raise ValueError("early_stopping_patience cannot be negative")
    elite_count = max(2, int(np.ceil(population_size * elite_fraction)))
    elite_count = min(elite_count, population_size)
    rng = np.random.default_rng(seed)
    mean = policy.readout_weights.copy()
    std = np.full_like(mean, initial_std)
    history: list[dict[str, float]] = []
    validation_start = _snapshot_environment(env)
    validation_seeds = [
        int((seed + 1_000_000_000 + index) % (2**32 - 1))
        for index in range(validation_episodes)
    ]
    best_weights = mean.copy()
    best_validation_return = -np.inf
    validations_without_improvement = 0

    for generation in range(generations):
        if mirrored_sampling:
            pair_count = population_size // 2
            perturbations = rng.normal(size=(pair_count, *mean.shape))
            candidates = np.concatenate(
                [mean + std * perturbations, mean - std * perturbations], axis=0
            )
            if population_size % 2:
                candidates = np.concatenate([candidates, mean[None, ...]], axis=0)
        else:
            candidates = rng.normal(mean, std, size=(population_size, *mean.shape))
        scores = np.empty(population_size)
        episode_seeds = [seed + 10_000 * generation + i for i in range(episodes_per_candidate)]
        # Every candidate begins from the same curriculum state. Otherwise
        # candidate order is confounded with parameter or system progression.
        generation_start = _snapshot_environment(env)
        generation_end = None
        for index, weights in enumerate(candidates):
            scores[index], candidate_end = _evaluate_weights(
                policy,
                env,
                weights,
                episode_seeds,
                generation_start,
            )
            if generation_end is None:
                generation_end = candidate_end

        if generation_end is not None:
            _restore_environment(env, generation_end)

        elite_indices = np.argsort(scores)[-elite_count:]
        elites = candidates[elite_indices]
        elite_mean = elites.mean(axis=0)
        elite_std = elites.std(axis=0)
        mean = smoothing * mean + (1.0 - smoothing) * elite_mean
        std = np.maximum(
            smoothing * std + (1.0 - smoothing) * elite_std,
            np.maximum(min_std, std_decay * std),
        )
        if max_std is not None:
            std = np.minimum(std, max_std)
        policy.readout_weights = mean.copy()
        record = {
            "generation": float(generation + 1),
            "best_return": float(scores[elite_indices[-1]]),
            "mean_return": float(scores.mean()),
            "search_std": float(std.mean()),
        }

        should_validate = validation_episodes > 0 and (
            (generation + 1) % validation_interval == 0
            or generation + 1 == generations
        )
        if should_validate:
            post_generation = _snapshot_environment(env)
            validation_return, _ = _evaluate_weights(
                policy,
                env,
                mean,
                validation_seeds,
                validation_start,
            )
            _restore_environment(env, post_generation)
            record["validation_return"] = validation_return
            if validation_return > best_validation_return:
                best_validation_return = validation_return
                best_weights = mean.copy()
                validations_without_improvement = 0
            else:
                validations_without_improvement += 1
            record["best_validation_return"] = float(best_validation_return)

        history.append(record)
        if callback is not None:
            callback(record)

        if (
            early_stopping_patience > 0
            and should_validate
            and validations_without_improvement >= early_stopping_patience
        ):
            break

    policy.readout_weights = (
        best_weights.copy() if validation_episodes > 0 else mean.copy()
    )
    return history
