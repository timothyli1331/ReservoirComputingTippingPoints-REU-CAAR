"""Local observables and basin diagnostics for the Duffing oscillator."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from dynamics import integrate_step


def jacobian(state: Sequence[float], params: Mapping[str, float]) -> np.ndarray:
    """Jacobian of the autonomous Duffing vector field at ``state``."""
    x, _ = np.asarray(state, dtype=float)
    alpha = float(params.get("alpha", -1.0))
    beta = float(params.get("beta", 1.0))
    damping = float(params.get("damping", 0.25))
    return np.array([[0.0, 1.0], [-alpha - 3.0 * beta * x**2, -damping]])


def local_stretching_rate(state: Sequence[float], params: Mapping[str, float]) -> float:
    """Largest instantaneous Euclidean stretching rate.

    This is the largest eigenvalue of the symmetric part of the Jacobian.
    """
    matrix = jacobian(state, params)
    return float(np.linalg.eigvalsh((matrix + matrix.T) / 2.0)[-1])


def finite_time_divergence(
    state: Sequence[float],
    params: Mapping[str, float],
    horizon: float = 1.0,
    dt: float = 0.01,
    epsilon: float = 1e-6,
    start_time: float = 0.0,
) -> float:
    """Estimate the largest finite-time Lyapunov exponent locally."""
    base = np.asarray(state, dtype=float).copy()
    perturbed = base + np.array([epsilon, 0.0])
    steps = max(1, int(np.ceil(horizon / dt)))
    time = float(start_time)
    for _ in range(steps):
        base = integrate_step(base, params, dt, time)
        perturbed = integrate_step(perturbed, params, dt, time)
        time += dt
    separation = max(float(np.linalg.norm(perturbed - base)), np.finfo(float).tiny)
    return float(np.log(separation / epsilon) / (steps * dt))


def basin_label(
    state: Sequence[float],
    params: Mapping[str, float],
    horizon: float = 20.0,
    dt: float = 0.02,
    start_time: float = 0.0,
) -> int:
    """Classify a trajectory by the sign of its long-time position."""
    current = np.asarray(state, dtype=float).copy()
    time = float(start_time)
    for _ in range(max(1, int(np.ceil(horizon / dt)))):
        current = integrate_step(current, params, dt, time)
        time += dt
    return int(np.sign(current[0]))


def basin_sensitivity(
    state: Sequence[float],
    params: Mapping[str, float],
    radius: float = 0.03,
    samples: int = 8,
    horizon: float = 20.0,
    dt: float = 0.02,
    start_time: float = 0.0,
) -> float:
    """Return neighborhood basin mixing on a scale from zero to one.

    A value near one means the sampled neighborhood is evenly divided between
    the two long-time wells. This remains useful for curved and time-dependent
    boundaries, unlike distance from a fixed straight line.
    """
    center = np.asarray(state, dtype=float)
    labels = []
    for angle in np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False):
        offset = radius * np.array([np.cos(angle), np.sin(angle)])
        labels.append(
            basin_label(center + offset, params, horizon, dt, start_time=start_time)
        )
    right_fraction = float(np.mean(np.asarray(labels) > 0))
    return 4.0 * right_fraction * (1.0 - right_fraction)
