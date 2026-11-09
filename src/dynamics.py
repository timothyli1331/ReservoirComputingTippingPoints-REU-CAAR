"""Dynamics and numerical integration for a controlled Duffing oscillator."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence

import numpy as np


def _get(params: Mapping[str, float], name: str, default: float) -> float:
    return float(params.get(name, default))


def randomize_duffing_params(
    params: MutableMapping[str, float],
    rng: np.random.Generator | None = None,
    ranges: Mapping[str, tuple[float, float]] | None = None,
) -> MutableMapping[str, float]:
    """Randomize Duffing parameters in place and return ``params``.

    The defaults stay near a commonly studied periodically forced chaotic
    Duffing regime. Supply ``ranges`` to override or add parameters. Each
    range is sampled uniformly.
    """
    generator = rng if rng is not None else np.random.default_rng()
    parameter_ranges = {
        "alpha": (-1.25, -0.75),
        "beta": (0.75, 1.25),
        "damping": (0.18, 0.25),
        "gamma": (0.27, 0.33),
        "omega": (1.15, 1.25),
    }
    if ranges is not None:
        parameter_ranges.update(ranges)

    sampled = {}
    for name, bounds in parameter_ranges.items():
        low, high = map(float, bounds)
        if not low < high:
            raise ValueError(f"range for {name!r} must satisfy low < high")
        sampled[name] = float(generator.uniform(low, high))

    candidate = {**params, **sampled}
    if float(candidate.get("alpha", -1.0)) >= 0:
        raise ValueError("alpha must remain negative for the double-well regime")
    if float(candidate.get("beta", 1.0)) <= 0:
        raise ValueError("beta must remain positive for the double-well regime")
    if float(candidate.get("damping", 0.25)) < 0:
        raise ValueError("damping must be nonnegative")
    if "omega" in candidate and float(candidate["omega"]) <= 0:
        raise ValueError("omega must be positive")

    params.update(sampled)
    return params


def walk_duffing_params(
    params: MutableMapping[str, float],
    rng: np.random.Generator | None = None,
    step_scales: Mapping[str, float] | None = None,
    bounds: Mapping[str, tuple[float, float]] | None = None,
) -> MutableMapping[str, float]:
    """Take one small, bounded random-walk step in parameter space.

    Unlike :func:`randomize_duffing_params`, this starts from the current
    values, so successive calls produce slowly drifting, correlated dynamics.
    Proposals reflect at the bounds instead of clipping, which avoids values
    accumulating at the limits. The mapping is updated in place and returned.
    """
    generator = rng if rng is not None else np.random.default_rng()
    walk_bounds = {
        "alpha": (-1.25, -0.75),
        "beta": (0.75, 1.25),
        "damping": (0.18, 0.25),
        "gamma": (0.27, 0.33),
        "omega": (1.15, 1.25),
    }
    walk_scales = {
        "alpha": 0.025,
        "beta": 0.025,
        "damping": 0.005,
        "gamma": 0.005,
        "omega": 0.005,
    }
    if bounds is not None:
        walk_bounds.update(bounds)
    if step_scales is not None:
        walk_scales.update(step_scales)

    defaults = {
        "alpha": -1.0,
        "beta": 1.0,
        "damping": 0.2,
        "gamma": 0.3,
        "omega": 1.2,
    }
    updated = {}
    for name, scale in walk_scales.items():
        if name not in walk_bounds:
            raise ValueError(f"missing bounds for parameter {name!r}")
        low, high = map(float, walk_bounds[name])
        scale = float(scale)
        if not low < high:
            raise ValueError(f"bounds for {name!r} must satisfy low < high")
        if scale < 0:
            raise ValueError(f"step scale for {name!r} must be nonnegative")

        current = float(params.get(name, defaults.get(name, (low + high) / 2)))
        proposal = current + float(generator.normal(0.0, scale))

        # Reflect any overshoot back into [low, high]. This also handles a
        # custom step large enough to cross a boundary more than once.
        width = high - low
        folded = (proposal - low) % (2.0 * width)
        updated[name] = low + (folded if folded <= width else 2.0 * width - folded)

    params.update(updated)
    return params


def duffing_derivative(
    t: float,
    state: Sequence[float],
    params: Mapping[str, float],
) -> np.ndarray:
    """Return ``[dx/dt, dv/dt]`` for the optionally forced Duffing system.

    The convention is
    ``x'' + damping*x' + alpha*x + beta*x**3 = gamma*cos(omega*t)``.
    """
    x, v = np.asarray(state, dtype=float)
    alpha = _get(params, "alpha", -1.0)
    beta = _get(params, "beta", 1.0)
    damping = _get(params, "damping", 0.25)
    gamma = _get(params, "gamma", 0.0)
    omega = _get(params, "omega", 1.0)
    return np.array(
        [v, gamma * np.cos(omega * t) - damping * v - alpha * x - beta * x**3],
        dtype=float,
    )


def integrate_step(
    state: Sequence[float],
    params: Mapping[str, float],
    dt: float,
    t: float = 0.0,
) -> np.ndarray:
    """Advance one step using fourth-order Runge--Kutta integration."""
    y = np.asarray(state, dtype=float)
    h = float(dt)
    k1 = duffing_derivative(t, y, params)
    k2 = duffing_derivative(t + h / 2, y + h * k1 / 2, params)
    k3 = duffing_derivative(t + h / 2, y + h * k2 / 2, params)
    k4 = duffing_derivative(t + h, y + h * k3, params)
    return y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def step_system(
    state: Sequence[float],
    action: Sequence[float],
    params: Mapping[str, float],
    dt: float,
    t: float = 0.0,
) -> np.ndarray:
    """Apply a bounded phase-space displacement, then evolve the system."""
    state_array = np.asarray(state, dtype=float)
    action_array = np.asarray(action, dtype=float)
    if state_array.shape != (2,) or action_array.shape != (2,):
        raise ValueError("state and action must both have shape (2,)")
    max_action = _get(params, "max_action", np.inf)
    displaced = state_array + np.clip(action_array, -max_action, max_action)
    return integrate_step(displaced, params, dt, t=t)
