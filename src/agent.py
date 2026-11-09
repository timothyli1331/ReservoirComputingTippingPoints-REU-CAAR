"""Stable-Baselines3 agent construction and loading helpers."""

from __future__ import annotations

from pathlib import Path


def build_agent(env, algorithm: str = "sac", **kwargs):
    """Build an SAC or PPO agent for the continuous environment."""
    try:
        from stable_baselines3 import PPO, SAC
    except ImportError as exc:
        raise ImportError(
            "Training requires stable-baselines3. Install it with "
            "`python -m pip install stable-baselines3`."
        ) from exc

    algorithms = {"sac": SAC, "ppo": PPO}
    name = algorithm.lower()
    if name not in algorithms:
        raise ValueError(f"algorithm must be one of {sorted(algorithms)}, got {algorithm!r}")
    defaults = {"verbose": 1, "seed": 0}
    if name == "sac":
        defaults.update(buffer_size=100_000, learning_starts=1_000)
    defaults.update(kwargs)
    return algorithms[name]("MlpPolicy", env, **defaults)


def load_agent(model_path: str | Path, env=None, algorithm: str = "sac"):
    """Load a previously saved SAC or PPO model."""
    try:
        from stable_baselines3 import PPO, SAC
    except ImportError as exc:
        raise ImportError("Loading a model requires stable-baselines3.") from exc
    algorithms = {"sac": SAC, "ppo": PPO}
    name = algorithm.lower()
    if name not in algorithms:
        raise ValueError(f"unknown algorithm: {algorithm!r}")
    return algorithms[name].load(str(model_path), env=env)
