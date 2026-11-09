"""A compatible family of saddle-separated double-well oscillators.

Every system has the same state ``[x, v]`` and obeys

    x_dot = v
    v_dot = force(x) - damping * v + noise.

Keeping a common state and action interface makes cross-system policy transfer
well-defined.  The potential shapes, damping, and stochasticity can still vary.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class DoubleWellSystem:
    """Parameters describing one two-dimensional double-well system."""

    name: str
    label: str
    kind: str
    params: Mapping[str, float]
    well_location: float
    state_limit: float = 2.75
    noise_std: float = 0.0


SYSTEMS = {
    "duffing": DoubleWellSystem(
        name="duffing",
        label="Quartic Duffing",
        kind="quartic",
        params={"linear": 1.0, "cubic": 1.0, "damping": 0.25},
        well_location=1.0,
    ),
    "langevin": DoubleWellSystem(
        name="langevin",
        label="Noisy Duffing / Langevin",
        kind="quartic",
        params={"linear": 1.0, "cubic": 1.0, "damping": 0.25},
        well_location=1.0,
        noise_std=0.08,
    ),
    "sextic": DoubleWellSystem(
        name="sextic",
        label="Sextic potential",
        kind="sextic",
        params={
            "linear": 1.0,
            "cubic": 0.20,
            "quintic": 0.40,
            "damping": 0.30,
        },
        # Positive root of quintic*y**2 + cubic*y - linear = 0, y=x**2.
        well_location=float(np.sqrt((-0.20 + np.sqrt(0.20**2 + 4.0 * 0.40)) / (2.0 * 0.40))),
        state_limit=3.0,
    ),
    "smooth_biquadratic": DoubleWellSystem(
        name="smooth_biquadratic",
        label="Smooth biquadratic",
        kind="smooth_biquadratic",
        params={"stiffness": 0.60, "separation": 1.0, "epsilon": 0.40, "damping": 0.22},
        # Minima of 0.5*k*(sqrt(x**2 + eps**2) - separation)**2.
        well_location=float(np.sqrt(1.0**2 - 0.40**2)),
    ),
}

DEFAULT_SYSTEM_NAMES = tuple(SYSTEMS)


# Bounds and one-step standard deviations for slowly varying evaluation
# dynamics. The bounds preserve a saddle at the origin and two stable wells.
SYSTEM_PARAMETER_WALKS = {
    "duffing": {
        "linear": (0.75, 1.25, 0.025 / np.sqrt(50.0)),
        "cubic": (0.75, 1.25, 0.025 / np.sqrt(50.0)),
        "damping": (0.18, 0.25, 0.005 / np.sqrt(50.0)),
    },
    "langevin": {
        "linear": (0.75, 1.25, 0.025 / np.sqrt(50.0)),
        "cubic": (0.75, 1.25, 0.025 / np.sqrt(50.0)),
        "damping": (0.18, 0.25, 0.005 / np.sqrt(50.0)),
        "noise_std": (0.05, 0.11, 0.005 / np.sqrt(50.0)),
    },
    "sextic": {
        "linear": (0.75, 1.25, 0.025 / np.sqrt(50.0)),
        "cubic": (0.10, 0.30, 0.010 / np.sqrt(50.0)),
        "quintic": (0.30, 0.50, 0.010 / np.sqrt(50.0)),
        "damping": (0.22, 0.38, 0.005 / np.sqrt(50.0)),
    },
    "smooth_biquadratic": {
        "stiffness": (0.45, 0.75, 0.015 / np.sqrt(50.0)),
        "separation": (0.90, 1.10, 0.010 / np.sqrt(50.0)),
        "epsilon": (0.30, 0.50, 0.010 / np.sqrt(50.0)),
        "damping": (0.16, 0.28, 0.005 / np.sqrt(50.0)),
    },
}


def parse_system_names(value: str | Sequence[str]) -> tuple[str, ...]:
    """Parse and validate a comma-separated or iterable system list."""
    if isinstance(value, str):
        names = tuple(name.strip() for name in value.split(",") if name.strip())
    else:
        names = tuple(value)
    if not names:
        raise ValueError("at least one system name is required")
    unknown = sorted(set(names) - set(SYSTEMS))
    if unknown:
        raise ValueError(f"unknown systems {unknown}; choose from {sorted(SYSTEMS)}")
    if len(set(names)) != len(names):
        raise ValueError("system names must not contain duplicates")
    return names


def get_system(name: str) -> DoubleWellSystem:
    """Return one registered system."""
    try:
        return SYSTEMS[name]
    except KeyError as exc:
        raise ValueError(f"unknown system {name!r}; choose from {sorted(SYSTEMS)}") from exc


def _reflected_walk_value(
    current: float,
    low: float,
    high: float,
    scale: float,
    rng: np.random.Generator,
) -> float:
    """Take one Gaussian step and reflect it into the supplied bounds."""
    proposal = float(current) + float(rng.normal(0.0, scale))
    width = high - low
    folded = (proposal - low) % (2.0 * width)
    return float(low + (folded if folded <= width else 2.0 * width - folded))


def _well_location(kind: str, params: Mapping[str, float]) -> float:
    """Return the positive well location for one valid parameter set."""
    if kind == "quartic":
        return float(np.sqrt(params["linear"] / params["cubic"]))
    if kind == "sextic":
        linear = params["linear"]
        cubic = params["cubic"]
        quintic = params["quintic"]
        squared_location = (
            -cubic + np.sqrt(cubic**2 + 4.0 * quintic * linear)
        ) / (2.0 * quintic)
        return float(np.sqrt(squared_location))
    if kind == "smooth_biquadratic":
        return float(np.sqrt(params["separation"] ** 2 - params["epsilon"] ** 2))
    raise ValueError(f"unsupported system kind {kind!r}")


def walk_system_params(
    system: DoubleWellSystem,
    rng: np.random.Generator | None = None,
    scale_multiplier: float = 1.0,
) -> DoubleWellSystem:
    """Return a system after one small, bounded random-walk parameter step.

    A new dataclass and parameter dictionary are returned so the registered
    system definitions and other environments remain unchanged.
    """
    scale_multiplier = float(scale_multiplier)
    if scale_multiplier < 0:
        raise ValueError("scale_multiplier must be nonnegative")
    generator = rng if rng is not None else np.random.default_rng()
    specs = SYSTEM_PARAMETER_WALKS[system.name]
    params = dict(system.params)
    noise_std = float(system.noise_std)
    for name, (low, high, scale) in specs.items():
        current = noise_std if name == "noise_std" else float(params[name])
        value = _reflected_walk_value(
            current, low, high, scale * scale_multiplier, generator
        )
        if name == "noise_std":
            noise_std = value
        else:
            params[name] = value
    return replace(
        system,
        params=params,
        well_location=_well_location(system.kind, params),
        noise_std=noise_std,
    )


def force(x: float | np.ndarray, system: DoubleWellSystem):
    """Return the conservative force ``-dV/dx``."""
    p = system.params
    if system.kind == "quartic":
        return p["linear"] * x - p["cubic"] * x**3
    if system.kind == "sextic":
        return p["linear"] * x - p["cubic"] * x**3 - p["quintic"] * x**5
    if system.kind == "smooth_biquadratic":
        radius = np.sqrt(x**2 + p["epsilon"] ** 2)
        return p["stiffness"] * (p["separation"] / radius - 1.0) * x
    raise ValueError(f"unsupported system kind {system.kind!r}")


def force_derivative(x: float, system: DoubleWellSystem) -> float:
    """Return ``d(force)/dx`` for the local Jacobian."""
    p = system.params
    if system.kind == "quartic":
        return float(p["linear"] - 3.0 * p["cubic"] * x**2)
    if system.kind == "sextic":
        return float(
            p["linear"] - 3.0 * p["cubic"] * x**2 - 5.0 * p["quintic"] * x**4
        )
    if system.kind == "smooth_biquadratic":
        radius = np.sqrt(x**2 + p["epsilon"] ** 2)
        return float(
            p["stiffness"]
            * (p["separation"] * p["epsilon"] ** 2 / radius**3 - 1.0)
        )
    raise ValueError(f"unsupported system kind {system.kind!r}")


def potential(x: float | np.ndarray, system: DoubleWellSystem):
    """Return the potential energy, up to an irrelevant additive constant."""
    p = system.params
    if system.kind == "quartic":
        return -0.5 * p["linear"] * x**2 + 0.25 * p["cubic"] * x**4
    if system.kind == "sextic":
        return (
            -0.5 * p["linear"] * x**2
            + 0.25 * p["cubic"] * x**4
            + (p["quintic"] / 6.0) * x**6
        )
    if system.kind == "smooth_biquadratic":
        radius = np.sqrt(x**2 + p["epsilon"] ** 2)
        return 0.5 * p["stiffness"] * (radius - p["separation"]) ** 2
    raise ValueError(f"unsupported system kind {system.kind!r}")


def derivative(t: float, state: Sequence[float], system: DoubleWellSystem) -> np.ndarray:
    """Return ``[x_dot, v_dot]`` for a system's deterministic drift."""
    del t
    x, velocity = np.asarray(state, dtype=float)
    acceleration = force(x, system) - system.params["damping"] * velocity
    return np.array([velocity, acceleration], dtype=float)


def jacobian(state: Sequence[float], system: DoubleWellSystem) -> np.ndarray:
    """Jacobian of the deterministic vector field."""
    x, _ = np.asarray(state, dtype=float)
    return np.array(
        [[0.0, 1.0], [force_derivative(float(x), system), -system.params["damping"]]],
        dtype=float,
    )


def local_stretching_rate(state: Sequence[float], system: DoubleWellSystem) -> float:
    """Largest eigenvalue of the symmetric part of the local Jacobian."""
    matrix = jacobian(state, system)
    return float(np.linalg.eigvalsh((matrix + matrix.T) / 2.0)[-1])


def saddle_stable_eigenvalue(system: DoubleWellSystem) -> float:
    """Stable eigenvalue of the saddle at ``(x, v) = (0, 0)``."""
    damping = float(system.params["damping"])
    slope = force_derivative(0.0, system)
    if slope <= 0:
        raise ValueError(f"{system.name} does not have an unstable saddle at x=0")
    return float((-damping - np.sqrt(damping**2 + 4.0 * slope)) / 2.0)


def velocity_scale(system: DoubleWellSystem) -> float:
    """Natural local velocity scale used to normalize observations."""
    return float(system.well_location * np.sqrt(force_derivative(0.0, system)))


def integrate_step(
    state: Sequence[float],
    system: DoubleWellSystem,
    dt: float,
    t: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Advance one RK4 drift step and, if configured, one Langevin kick."""
    y = np.asarray(state, dtype=float)
    h = float(dt)
    k1 = derivative(t, y, system)
    k2 = derivative(t + h / 2.0, y + h * k1 / 2.0, system)
    k3 = derivative(t + h / 2.0, y + h * k2 / 2.0, system)
    k4 = derivative(t + h, y + h * k3, system)
    next_state = y + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    if system.noise_std > 0:
        if rng is None:
            raise ValueError("an RNG is required for a stochastic system")
        next_state[1] += system.noise_std * np.sqrt(h) * rng.normal()
    return next_state


def finite_time_basin_label(
    state: Sequence[float],
    system: DoubleWellSystem,
    horizon: float = 2.4,
    dt: float = 0.06,
    start_time: float = 0.0,
    rng: np.random.Generator | None = None,
) -> int:
    """Classify a state by its finite-time left/right outcome.

    For a noisy system this is one stochastic realization, so callers should
    aggregate several nearby samples rather than treating one label as exact.
    """
    if horizon <= 0 or dt <= 0:
        raise ValueError("horizon and dt must be positive")
    if system.noise_std > 0 and rng is None:
        raise ValueError("an RNG is required for a stochastic system")
    current = np.asarray(state, dtype=float).copy()
    steps = max(1, int(np.ceil(horizon / dt)))
    time = float(start_time)
    for _ in range(steps):
        current = integrate_step(current, system, dt, time, rng)
        time += dt
    return 1 if current[0] >= 0 else -1


def basin_mixing_score(
    state: Sequence[float],
    system: DoubleWellSystem,
    radius: float,
    samples: int = 6,
    horizon: float = 2.4,
    dt: float = 0.06,
    start_time: float = 0.0,
    rng: np.random.Generator | None = None,
    rollouts_per_sample: int = 1,
) -> float:
    """Measure local left/right outcome mixing from zero to one.

    Nearby initial conditions are integrated instead of projecting onto the
    saddle's tangent line. The score is one for an even left/right split and
    zero when every sampled trajectory has the same finite-time outcome.
    Multiple stochastic rollouts can be requested for noisy systems.
    """
    if radius <= 0:
        raise ValueError("radius must be positive")
    if samples < 2:
        raise ValueError("samples must be at least 2")
    if rollouts_per_sample < 1:
        raise ValueError("rollouts_per_sample must be at least 1")
    if system.noise_std > 0 and rng is None:
        raise ValueError("an RNG is required for a stochastic system")

    center = np.asarray(state, dtype=float)
    labels = []
    for angle in np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False):
        offset = radius * np.array([np.cos(angle), np.sin(angle)])
        for _ in range(rollouts_per_sample):
            labels.append(
                finite_time_basin_label(
                    center + offset,
                    system,
                    horizon=horizon,
                    dt=dt,
                    start_time=start_time,
                    rng=rng,
                )
            )
    right_fraction = float(np.mean(np.asarray(labels) > 0))
    return 4.0 * right_fraction * (1.0 - right_fraction)


def step_system(
    state: Sequence[float],
    action: Sequence[float],
    system: DoubleWellSystem,
    dt: float,
    t: float,
    rng: np.random.Generator,
    action_limit: float,
    dynamics_substeps: int = 1,
) -> np.ndarray:
    """Apply one action, then evolve for one or more full dynamics intervals."""
    state_array = np.asarray(state, dtype=float)
    action_array = np.asarray(action, dtype=float)
    if state_array.shape != (2,) or action_array.shape != (2,):
        raise ValueError("state and action must both have shape (2,)")
    if dynamics_substeps < 1:
        raise ValueError("dynamics_substeps must be at least 1")
    evolved = state_array + np.clip(action_array, -action_limit, action_limit)
    for substep in range(dynamics_substeps):
        evolved = integrate_step(
            evolved,
            system,
            dt,
            t + substep * dt,
            rng,
        )
    return evolved
