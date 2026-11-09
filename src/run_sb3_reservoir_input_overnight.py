"""Run matched feedforward, LSTM, and ESN Soft Actor-Critic comparisons.

The controller suite is:

    SAC:                 raw observation -> standard SB3 SAC
    SB3 reservoir SAC:  [raw observation, fixed ESN state] -> SB3 SAC
    LSTM-SAC:           learned recurrent actor and twin recurrent critics
    ESN-SAC:            matched actor/critics with fixed recurrent cores

The LSTM and ESN agents use identical Gaussian actor and twin-Q critic heads,
contiguous sequence replay, burn-in, target critics, and entropy tuning.  Their
only architectural difference is learned LSTM recurrence versus fixed ESN
recurrence.  All recurrent states reset at episode boundaries.

By default, this script trains seeds 0,1,2,3,4 for every condition that already
has a complete set of regular-SAC baselines under models/overnight_memory_tests.
It then evaluates each new model against the same-seed baseline and writes raw
episode results, per-seed comparison PNGs, and across-training-seed CSV/PNG
summaries.

Example overnight run:

    .venv/bin/python src/run_sb3_reservoir_input_overnight.py \
        > logs/sb3_reservoir_input_overnight.log 2>&1

Use --dry-run to verify all inputs and planned output paths without training.
Existing new checkpoints are skipped unless --force is supplied, so an
interrupted run can be resumed with the same command.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

_matplotlib_cache = Path(tempfile.gettempdir()) / "duffing_matplotlib_cache"
_matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_matplotlib_cache))

import gymnasium as gym
import matplotlib
import numpy as np
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

from multi_environment import MultiDoubleWellEnv
from recurrent_sac import RecurrentSACAgent
from systems import DEFAULT_SYSTEM_NAMES, get_system

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Condition:
    name: str
    boundary_start_probability: float
    parameter_walk_scale: float
    include_local_acceleration: bool
    include_local_stretching: bool
    dynamics_substeps: int
    description: str


# These match the completed regular-SAC conditions produced by
# run_overnight_experiments.py.  state_only_strong_drift is omitted because no
# corresponding regular-SAC checkpoints currently exist.
CONDITIONS = (
    Condition(
        "far_start",
        0.15,
        1.0,
        True,
        True,
        1,
        "85% of resets begin away from the boundary",
    ),
    Condition(
        "high_drift",
        0.60,
        4.0,
        True,
        True,
        1,
        "four-times-larger parameter random-walk increments",
    ),
    Condition(
        "no_stretch",
        0.60,
        1.0,
        True,
        False,
        1,
        "local stretching rate removed from observations",
    ),
    Condition(
        "no_accel_no_stretch",
        0.60,
        1.0,
        False,
        False,
        1,
        "observations contain only normalized position and velocity",
    ),
)

CONTROLLERS = ("sb3_reservoir", "lstm", "esn")
MODEL_LABELS = {
    "baseline": "SAC",
    "sb3_reservoir": "SB3-Reservoir-SAC",
    "lstm": "LSTM-SAC",
    "esn": "ESN-SAC",
}


class ReservoirObservationWrapper(gym.Wrapper):
    """Append a fixed ReservoirPy state to each raw environment observation."""

    def __init__(
        self,
        env: gym.Env,
        *,
        units: int = 100,
        spectral_radius: float = 0.9,
        leaking_rate: float = 0.3,
        input_scaling: float = 0.5,
        reservoir_seed: int = 0,
    ) -> None:
        super().__init__(env)
        try:
            from reservoirpy.nodes import Reservoir
        except ImportError as exc:
            raise ImportError(
                "ReservoirPy is required: python -m pip install reservoirpy"
            ) from exc

        if units < 1:
            raise ValueError("units must be positive")
        if not 0.0 < leaking_rate <= 1.0:
            raise ValueError("leaking_rate must be in (0, 1]")
        if spectral_radius < 0.0 or input_scaling <= 0.0:
            raise ValueError(
                "spectral_radius must be nonnegative and input_scaling positive"
            )
        if not isinstance(env.observation_space, spaces.Box):
            raise TypeError("ReservoirObservationWrapper requires a Box observation")
        if len(env.observation_space.shape) != 1:
            raise ValueError("the wrapped observation must be one-dimensional")

        self.units = int(units)
        self.spectral_radius = float(spectral_radius)
        self.leaking_rate = float(leaking_rate)
        self.input_scaling = float(input_scaling)
        self.reservoir_seed = int(reservoir_seed)
        self.raw_observation_size = int(env.observation_space.shape[0])
        self.reservoir = Reservoir(
            units=self.units,
            sr=self.spectral_radius,
            lr=self.leaking_rate,
            input_scaling=self.input_scaling,
            seed=self.reservoir_seed,
        )

        # Preserve the exact raw observation supplied to the regular-SAC
        # baseline and append bounded reservoir activations.  No explicit bias
        # feature is needed because SB3's neural layers already contain biases.
        low = np.concatenate(
            (
                np.asarray(env.observation_space.low, dtype=np.float32),
                -np.ones(self.units, dtype=np.float32),
            )
        )
        high = np.concatenate(
            (
                np.asarray(env.observation_space.high, dtype=np.float32),
                np.ones(self.units, dtype=np.float32),
            )
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def _reset_reservoir(self) -> None:
        # ReservoirPy initializes its state lazily on the first call to step().
        if hasattr(self.reservoir, "state"):
            self.reservoir.reset()

    def _augment(self, observation: np.ndarray) -> np.ndarray:
        raw = np.asarray(observation, dtype=np.float32).reshape(-1)
        if raw.shape != (self.raw_observation_size,):
            raise ValueError(
                f"expected observation shape ({self.raw_observation_size},), "
                f"got {raw.shape}"
            )

        # Only the reservoir drive is squashed.  The raw-observation portion of
        # the SB3 input remains identical to the regular-SAC baseline.
        reservoir_input = np.tanh(raw.astype(float))
        reservoir_state = np.asarray(
            self.reservoir.step(reservoir_input), dtype=np.float32
        ).reshape(-1)
        return np.concatenate((raw, reservoir_state)).astype(np.float32)

    def reset(self, **kwargs):
        self._reset_reservoir()
        observation, info = self.env.reset(**kwargs)
        return self._augment(observation), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self._augment(observation), reward, terminated, truncated, info


class TrainingHistoryCallback(BaseCallback):
    """Collect lightweight, machine-readable SB3 learning-curve data."""

    def __init__(self, log_interval: int = 1_000) -> None:
        super().__init__(verbose=0)
        self.log_interval = int(log_interval)
        self.next_log = self.log_interval
        self.current_return = 0.0
        self.completed_episodes = 0
        self.recent_returns: list[float] = []
        self.history: list[dict[str, float | int]] = []

    def _on_step(self) -> bool:
        rewards = np.asarray(self.locals["rewards"]).reshape(-1)
        dones = np.asarray(self.locals["dones"]).reshape(-1)
        self.current_return += float(rewards[0])
        if bool(dones[0]):
            self.completed_episodes += 1
            self.recent_returns.append(self.current_return)
            self.recent_returns = self.recent_returns[-20:]
            self.current_return = 0.0
        if self.num_timesteps >= self.next_log:
            self.history.append(
                {
                    "step": self.num_timesteps,
                    "episodes": self.completed_episodes,
                    "return20": (
                        float(mean(self.recent_returns))
                        if self.recent_returns
                        else float("nan")
                    ),
                }
            )
            self.next_log += self.log_interval
        return True


def parse_int_list(text: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not values:
        raise argparse.ArgumentTypeError("the list cannot be empty")
    return values


def parse_condition_list(text: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in text.split(",") if item.strip())
    valid = {condition.name for condition in CONDITIONS}
    unknown = sorted(set(values) - valid)
    if not values:
        raise argparse.ArgumentTypeError("the condition list cannot be empty")
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown conditions {unknown}; choose from {sorted(valid)}"
        )
    return values


def parse_controller_list(text: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in text.split(",") if item.strip())
    unknown = sorted(set(values) - set(CONTROLLERS))
    if not values:
        raise argparse.ArgumentTypeError("the controller list cannot be empty")
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown controllers {unknown}; choose from {list(CONTROLLERS)}"
        )
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seeds", type=parse_int_list, default=parse_int_list("0,1,2,3,4"))
    parser.add_argument(
        "--conditions",
        type=parse_condition_list,
        default=tuple(condition.name for condition in CONDITIONS),
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument(
        "--recurrent-checkpoint",
        choices=("best", "latest"),
        default="best",
        help="Use the validation-selected or final recurrent checkpoint.",
    )
    parser.add_argument(
        "--controllers",
        type=parse_controller_list,
        default=CONTROLLERS,
        help="Comma-separated: sb3_reservoir,lstm,esn.",
    )
    parser.add_argument("--reservoir-units", type=int, default=100)
    parser.add_argument("--spectral-radius", type=float, default=0.9)
    parser.add_argument("--leaking-rate", type=float, default=0.3)
    parser.add_argument("--input-scaling", type=float, default=0.5)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sequence-batch-size", type=int, default=32)
    parser.add_argument("--burn-in", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--gradient-steps-per-transition", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--verbose", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=Path("models/overnight_memory_tests"),
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path("models/recurrent_controller_tests"),
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path("results/recurrent_controller_tests"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--retrain-baseline",
        action="store_true",
        help="Retrain regular SB3 SAC for matched timing/learning-history data.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain and overwrite new reservoir-input checkpoints.",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Train models without evaluating against existing baselines.",
    )
    args = parser.parse_args()

    if args.timesteps < 1 or args.episodes < 1:
        parser.error("timesteps and episodes must be positive")
    if (
        args.reservoir_units < 1
        or args.buffer_size < 1
        or args.batch_size < 1
        or args.sequence_batch_size < 1
        or args.sequence_length < 1
    ):
        parser.error("units, buffer size, and batch size must be positive")
    if args.learning_starts < 0 or args.burn_in < 0:
        parser.error("learning-starts and burn-in cannot be negative")
    if args.burn_in + args.sequence_length > 300:
        parser.error("burn-in + sequence-length cannot exceed 300")
    if args.gradient_steps_per_transition <= 0:
        parser.error("gradient-steps-per-transition must be positive")
    if not 0.0 < args.leaking_rate <= 1.0:
        parser.error("leaking-rate must be in (0, 1]")
    if args.spectral_radius < 0.0 or args.input_scaling <= 0.0:
        parser.error("spectral-radius must be nonnegative and input-scaling positive")
    return args


def status(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def resolve_under(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def make_base_env(
    condition: Condition,
    *,
    system_names=DEFAULT_SYSTEM_NAMES,
    selection: str = "random",
    boundary_update_interval: int = 5,
) -> MultiDoubleWellEnv:
    return MultiDoubleWellEnv(
        system_names=system_names,
        selection=selection,
        include_system_id=False,
        include_local_acceleration=condition.include_local_acceleration,
        include_local_stretching=condition.include_local_stretching,
        parameter_walk=True,
        parameter_walk_interval_steps=1,
        parameter_walk_scale=condition.parameter_walk_scale,
        boundary_start_probability=condition.boundary_start_probability,
        evolve_dynamics=True,
        dynamics_substeps=condition.dynamics_substeps,
        use_basin_mixing=True,
        boundary_update_interval=boundary_update_interval,
    )


def wrap_env(
    env: gym.Env,
    args: argparse.Namespace,
    reservoir_seed: int,
) -> ReservoirObservationWrapper:
    return ReservoirObservationWrapper(
        env,
        units=args.reservoir_units,
        spectral_radius=args.spectral_radius,
        leaking_rate=args.leaking_rate,
        input_scaling=args.input_scaling,
        reservoir_seed=reservoir_seed,
    )


def checkpoint_stem(model_root: Path, condition: Condition, seed: int) -> Path:
    return model_root / condition.name / f"sb3_reservoir_sac_seed_{seed}"


def train_baseline_one(
    args: argparse.Namespace,
    condition: Condition,
    seed: int,
) -> Path:
    destination = args.model_root / condition.name / f"tracked_sac_seed_{seed}"
    checkpoint = destination.with_suffix(".zip")
    if checkpoint.is_file() and not args.force:
        status(f"SKIP baseline training {condition.name} seed {seed}: checkpoint exists")
        return checkpoint
    if args.dry_run:
        status(f"PLAN baseline training {condition.name} seed {seed} -> {checkpoint}")
        return checkpoint

    destination.parent.mkdir(parents=True, exist_ok=True)
    env = make_base_env(condition)
    model = SAC(
        "MlpPolicy",
        env,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        seed=seed,
        device=args.device,
        verbose=args.verbose,
    )
    callback = TrainingHistoryCallback(log_interval=1_000)
    status(f"START baseline training {condition.name} seed {seed}")
    training_started = datetime.now()
    model.learn(total_timesteps=args.timesteps, callback=callback)
    training_seconds = (datetime.now() - training_started).total_seconds()
    model.save(destination)
    env.close()
    metadata = {
        "condition": asdict(condition),
        "seed": seed,
        "timesteps": args.timesteps,
        "algorithm": "stable_baselines3.SAC",
        "policy": "MlpPolicy",
        "policy_input": "raw_observation",
        "training_seconds": training_seconds,
        "environment_steps_per_second": args.timesteps / max(training_seconds, 1e-9),
        "learning_history": callback.history,
    }
    destination.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    status(f"DONE baseline training {condition.name} seed {seed}: {checkpoint}")
    return checkpoint


def train_one(
    args: argparse.Namespace,
    condition: Condition,
    seed: int,
) -> Path:
    destination = checkpoint_stem(args.model_root, condition, seed)
    checkpoint = destination.with_suffix(".zip")
    if checkpoint.is_file() and not args.force:
        status(f"SKIP training {condition.name} seed {seed}: {checkpoint} exists")
        return checkpoint
    if args.dry_run:
        status(f"PLAN train {condition.name} seed {seed} -> {checkpoint}")
        return checkpoint

    destination.parent.mkdir(parents=True, exist_ok=True)
    env = wrap_env(make_base_env(condition), args, reservoir_seed=seed)
    model = SAC(
        "MlpPolicy",
        env,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        seed=seed,
        device=args.device,
        verbose=args.verbose,
    )
    status(f"START training {condition.name} seed {seed}")
    training_started = datetime.now()
    history_callback = TrainingHistoryCallback(log_interval=1_000)
    model.learn(total_timesteps=args.timesteps, callback=history_callback)
    training_seconds = (datetime.now() - training_started).total_seconds()
    model.save(destination)
    env.close()

    metadata = {
        "condition": asdict(condition),
        "seed": seed,
        "timesteps": args.timesteps,
        "algorithm": "stable_baselines3.SAC",
        "policy": "MlpPolicy",
        "policy_input": "raw_observation_plus_reservoir_state",
        "reservoir_units": args.reservoir_units,
        "spectral_radius": args.spectral_radius,
        "leaking_rate": args.leaking_rate,
        "input_scaling": args.input_scaling,
        "reservoir_input": "tanh(raw_observation)",
        "reservoir_seed": seed,
        "action_limit": 0.05,
        "buffer_size": args.buffer_size,
        "learning_starts": args.learning_starts,
        "batch_size": args.batch_size,
        "training_seconds": training_seconds,
        "environment_steps_per_second": args.timesteps / max(training_seconds, 1e-9),
        "learning_history": history_callback.history,
    }
    destination.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    status(f"DONE training {condition.name} seed {seed}: {checkpoint}")
    return checkpoint


def recurrent_checkpoint(
    model_root: Path,
    condition: Condition,
    encoder: str,
    seed: int,
) -> Path:
    return model_root / condition.name / f"{encoder}_sac_seed_{seed}.pt"


def train_recurrent_one(
    args: argparse.Namespace,
    condition: Condition,
    encoder: str,
    seed: int,
) -> Path:
    checkpoint = recurrent_checkpoint(args.model_root, condition, encoder, seed)
    completion_metadata = checkpoint.with_suffix(".json")
    # The recurrent trainer saves a best-so-far .pt at validation time.  Its
    # JSON metadata is written only after the requested training budget ends,
    # so require both files before treating an interrupted run as complete.
    if checkpoint.is_file() and completion_metadata.is_file() and not args.force:
        status(f"SKIP training {condition.name} {encoder} seed {seed}: checkpoint exists")
        return checkpoint
    if args.dry_run:
        status(f"PLAN train {condition.name} {encoder} seed {seed} -> {checkpoint}")
        return checkpoint

    trainer = Path(__file__).resolve().parent / "train_recurrent_sac_multi.py"
    boolean_flags = [
        (
            "--include-local-acceleration"
            if condition.include_local_acceleration
            else "--no-include-local-acceleration"
        ),
        (
            "--include-local-stretching"
            if condition.include_local_stretching
            else "--no-include-local-stretching"
        ),
    ]
    command = [
        sys.executable,
        str(trainer),
        "--encoder",
        encoder,
        "--timesteps",
        str(args.timesteps),
        "--seed",
        str(seed),
        "--model-path",
        str(checkpoint),
        "--recurrent-units",
        str(args.reservoir_units),
        "--spectral-radius",
        str(args.spectral_radius),
        "--leaking-rate",
        str(args.leaking_rate),
        "--input-scaling",
        str(args.input_scaling),
        "--buffer-size",
        str(args.buffer_size),
        "--learning-starts",
        str(args.learning_starts),
        "--batch-size",
        str(args.sequence_batch_size),
        "--burn-in",
        str(args.burn_in),
        "--sequence-length",
        str(args.sequence_length),
        "--gradient-steps-per-transition",
        str(args.gradient_steps_per_transition),
        "--parameter-walk-scale",
        str(condition.parameter_walk_scale),
        "--boundary-start-probability",
        str(condition.boundary_start_probability),
        "--dynamics-substeps",
        str(condition.dynamics_substeps),
        "--boundary-update-interval",
        "5",
        "--device",
        args.device,
        *boolean_flags,
    ]
    status(f"START training {condition.name} {encoder} seed {seed}")
    subprocess.run(command, check=True)
    status(f"DONE training {condition.name} {encoder} seed {seed}: {checkpoint}")
    return checkpoint


def run_episode(
    model,
    env: gym.Env,
    system_name: str,
    seed: int,
    *,
    recurrent: bool = False,
) -> dict[str, float]:
    observation, _ = env.reset(seed=seed, options={"system_name": system_name})
    if recurrent:
        model.reset()
    rewards: list[float] = []
    boundary_distances: list[float] = []
    done = False
    while not done:
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        boundary_distances.append(float(info["boundary_distance"]))
        done = bool(terminated or truncated)
    return {
        "return": float(np.sum(rewards)),
        "mean_boundary_score": float(np.mean(boundary_distances)),
        "survived_steps": float(len(rewards)),
    }


def plot_seed_comparison(rows: list[dict[str, object]], output: Path) -> None:
    """Plot paired episode return and boundary-distance comparisons."""
    preferred = tuple(MODEL_LABELS.values())
    present = {str(row["model"]) for row in rows}
    model_names = tuple(name for name in preferred if name in present)
    color_map = {
        "SAC": "tab:blue",
        "SB3-Reservoir-SAC": "tab:orange",
        "LSTM-SAC": "tab:green",
        "ESN-SAC": "tab:red",
    }
    fig, axes = plt.subplots(
        2,
        len(DEFAULT_SYSTEM_NAMES),
        figsize=(4.2 * len(DEFAULT_SYSTEM_NAMES), 7.5),
        squeeze=False,
    )
    for column, system_name in enumerate(DEFAULT_SYSTEM_NAMES):
        system_rows = [row for row in rows if row["system"] == system_name]
        return_means = []
        return_stds = []
        boundary_means = []
        boundary_stds = []
        for model_name in model_names:
            selected = [row for row in system_rows if row["model"] == model_name]
            returns = [float(row["return"]) for row in selected]
            boundaries = [float(row["mean_boundary_score"]) for row in selected]
            return_means.append(float(np.mean(returns)))
            return_stds.append(float(np.std(returns)))
            boundary_means.append(float(np.mean(boundaries)))
            boundary_stds.append(float(np.std(boundaries)))

        positions = np.arange(len(model_names))
        axes[0, column].bar(
            positions,
            return_means,
            yerr=return_stds,
            color=[color_map[name] for name in model_names],
            alpha=0.82,
            capsize=4,
        )
        axes[1, column].bar(
            positions,
            boundary_means,
            yerr=boundary_stds,
            color=[color_map[name] for name in model_names],
            alpha=0.82,
            capsize=4,
        )
        for axis in axes[:, column]:
            axis.set_xticks(
                positions,
                [name.replace("-", "\n", 1) for name in model_names],
                fontsize=8,
            )
            axis.grid(axis="y", alpha=0.2)
        axes[0, column].set_title(get_system(system_name).label)
        axes[0, column].set_ylabel("episode return\n(higher is better)")
        axes[1, column].set_ylabel("boundary distance\n(lower is better)")

    training_seed = rows[0]["training_seed"]
    condition = rows[0]["condition"]
    fig.suptitle(
        f"Recurrent-controller comparison — {condition}, seed {training_seed}"
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def evaluate_one(
    args: argparse.Namespace,
    condition: Condition,
    training_seed: int,
    baseline_checkpoint: Path,
    controller_checkpoints: dict[str, Path],
) -> Path:
    output = (
        args.result_root
        / condition.name
        / f"controller_comparison_seed_{training_seed}.csv"
    )
    if args.dry_run:
        comparisons = ", ".join(
            f"{name}={path.name}" for name, path in controller_checkpoints.items()
        )
        status(
            f"PLAN evaluate {condition.name} seed {training_seed}: "
            f"baseline={baseline_checkpoint.name}; {comparisons}"
        )
        return output

    baseline = SAC.load(baseline_checkpoint, device=args.device)
    rows: list[dict[str, object]] = []
    baseline_counts = {
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in baseline.policy.parameters()
            if parameter.requires_grad
        ),
        "trainable_actor_parameters": sum(
            parameter.numel()
            for parameter in baseline.actor.parameters()
            if parameter.requires_grad
        ),
        "fixed_encoder_parameters": 0,
    }
    models: list[tuple[str, object, str, dict[str, int]]] = [
        (MODEL_LABELS["baseline"], baseline, "raw", baseline_counts)
    ]
    for controller, checkpoint in controller_checkpoints.items():
        if controller == "sb3_reservoir":
            model = SAC.load(checkpoint, device=args.device)
            counts = {
                "trainable_parameters": sum(
                    parameter.numel()
                    for parameter in model.policy.parameters()
                    if parameter.requires_grad
                ),
                "trainable_actor_parameters": sum(
                    parameter.numel()
                    for parameter in model.actor.parameters()
                    if parameter.requires_grad
                ),
                "fixed_encoder_parameters": (
                    args.reservoir_units**2
                    + args.reservoir_units
                    * (
                        2
                        + int(condition.include_local_acceleration)
                        + int(condition.include_local_stretching)
                    )
                    + args.reservoir_units
                ),
            }
            models.append((MODEL_LABELS[controller], model, "wrapped", counts))
        else:
            model = RecurrentSACAgent.load(checkpoint, device=args.device)
            models.append(
                (MODEL_LABELS[controller], model, "recurrent", model.parameter_counts())
            )
    for system_name in DEFAULT_SYSTEM_NAMES:
        for model_name, model, input_kind, counts in models:
            for episode in range(args.episodes):
                base_env = make_base_env(
                    condition,
                    system_names=(system_name,),
                    boundary_update_interval=1,
                )
                env: gym.Env = (
                    wrap_env(base_env, args, reservoir_seed=training_seed)
                    if input_kind == "wrapped"
                    else base_env
                )
                evaluation_seed = 2_000_000 + training_seed * 100 + episode
                result = run_episode(
                    model,
                    env,
                    system_name,
                    evaluation_seed,
                    recurrent=input_kind == "recurrent",
                )
                env.close()
                rows.append(
                    {
                        "condition": condition.name,
                        "training_seed": training_seed,
                        "system": system_name,
                        "model": model_name,
                        "episode": episode,
                        "evaluation_seed": evaluation_seed,
                        "trainable_parameters": counts["trainable_parameters"],
                        "trainable_actor_parameters": counts[
                            "trainable_actor_parameters"
                        ],
                        "fixed_encoder_parameters": counts[
                            "fixed_encoder_parameters"
                        ],
                        **result,
                    }
                )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_output = output.with_suffix(".png")
    plot_seed_comparison(rows, plot_output)
    status(
        f"DONE evaluation {condition.name} seed {training_seed}: "
        f"{output.name}, {plot_output.name}"
    )
    return output


def aggregate_condition(
    args: argparse.Namespace,
    condition: Condition,
    episode_files: list[Path],
) -> Path:
    per_seed: dict[tuple[str, str, int], list[dict[str, str]]] = {}
    for path in episode_files:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                key = (row["model"], row["system"], int(row["training_seed"]))
                per_seed.setdefault(key, []).append(row)

    across_seeds: dict[
        tuple[str, str], list[tuple[float, float, float, int, int, int]]
    ] = {}
    for (model, system, _seed), rows in per_seed.items():
        seed_value = (
            mean(float(row["return"]) for row in rows),
            mean(float(row["mean_boundary_score"]) for row in rows),
            mean(float(row["survived_steps"]) for row in rows),
            int(rows[0]["trainable_parameters"]),
            int(rows[0]["trainable_actor_parameters"]),
            int(rows[0]["fixed_encoder_parameters"]),
        )
        across_seeds.setdefault((model, system), []).append(seed_value)

    output = args.result_root / condition.name / "summary.csv"
    fields = (
        "condition",
        "model",
        "system",
        "training_seeds",
        "mean_return",
        "std_return",
        "mean_boundary_score",
        "std_boundary_score",
        "mean_survived_steps",
        "std_survived_steps",
        "trainable_parameters",
        "trainable_actor_parameters",
        "fixed_encoder_parameters",
    )
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for (model, system), seed_values in sorted(across_seeds.items()):
            returns, boundaries, survived, trainable, actor_trainable, fixed = zip(
                *seed_values
            )
            writer.writerow(
                {
                    "condition": condition.name,
                    "model": model,
                    "system": system,
                    "training_seeds": len(seed_values),
                    "mean_return": mean(returns),
                    "std_return": stdev(returns) if len(returns) > 1 else 0.0,
                    "mean_boundary_score": mean(boundaries),
                    "std_boundary_score": (
                        stdev(boundaries) if len(boundaries) > 1 else 0.0
                    ),
                    "mean_survived_steps": mean(survived),
                    "std_survived_steps": (
                        stdev(survived) if len(survived) > 1 else 0.0
                    ),
                    "trainable_parameters": int(mean(trainable)),
                    "trainable_actor_parameters": int(mean(actor_trainable)),
                    "fixed_encoder_parameters": int(mean(fixed)),
                }
            )
    status(f"WROTE summary: {output}")
    plot_summary(output)
    return output


def plot_summary(summary_csv: Path) -> Path:
    """Plot mean returns across training seeds for one condition."""
    with summary_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    preferred = tuple(MODEL_LABELS.values())
    present = {row["model"] for row in rows}
    model_names = tuple(name for name in preferred if name in present)
    color_map = {
        "SAC": "tab:blue",
        "SB3-Reservoir-SAC": "tab:orange",
        "LSTM-SAC": "tab:green",
        "ESN-SAC": "tab:red",
    }
    systems = tuple(DEFAULT_SYSTEM_NAMES)
    positions = np.arange(len(systems), dtype=float)
    width = 0.8 / len(model_names)
    fig, axis = plt.subplots(figsize=(10.5, 5.5))
    for index, model_name in enumerate(model_names):
        selected = {
            row["system"]: row for row in rows if row["model"] == model_name
        }
        means = [float(selected[name]["mean_return"]) for name in systems]
        errors = [float(selected[name]["std_return"]) for name in systems]
        offset = (index - (len(model_names) - 1) / 2.0) * width
        axis.bar(
            positions + offset,
            means,
            width,
            yerr=errors,
            label=model_name,
            color=color_map[model_name],
            alpha=0.82,
            capsize=4,
        )
    axis.set_xticks(positions, [get_system(name).label for name in systems])
    axis.set_ylabel("mean episode return across training seeds")
    axis.set_title(f"Recurrent-controller comparison — {rows[0]['condition']}")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    output = summary_csv.with_suffix(".png")
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    status(f"WROTE summary plot: {output}")
    return output


def preflight(args: argparse.Namespace, conditions: tuple[Condition, ...]) -> None:
    if any(name in args.controllers for name in ("lstm", "esn")):
        trainer = Path(__file__).resolve().parent / "train_recurrent_sac_multi.py"
        if not trainer.is_file():
            raise FileNotFoundError(f"Missing recurrent trainer: {trainer}")
    if args.retrain_baseline:
        return
    missing = []
    for condition in conditions:
        for seed in args.seeds:
            checkpoint = (
                args.baseline_root / condition.name / f"regular_sac_seed_{seed}.zip"
            )
            if not checkpoint.is_file():
                missing.append(checkpoint)
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Missing existing regular-SAC checkpoints required for comparison:\n"
            + formatted
        )


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    args.baseline_root = resolve_under(root, args.baseline_root)
    args.model_root = resolve_under(root, args.model_root)
    args.result_root = resolve_under(root, args.result_root)
    selected = tuple(
        condition for condition in CONDITIONS if condition.name in args.conditions
    )
    preflight(args, selected)

    manifest = {
        "created_by": Path(__file__).name,
        "timesteps": args.timesteps,
        "seeds": list(args.seeds),
        "episodes": args.episodes,
        "controllers": list(args.controllers),
        "retrain_baseline": args.retrain_baseline,
        "baseline_root": str(args.baseline_root),
        "model_root": str(args.model_root),
        "result_root": str(args.result_root),
        "conditions": [asdict(condition) for condition in selected],
        "reservoir": {
            "units": args.reservoir_units,
            "spectral_radius": args.spectral_radius,
            "leaking_rate": args.leaking_rate,
            "input_scaling": args.input_scaling,
            "input": "tanh(raw_observation)",
            "output_to_sac": "raw_observation_plus_reservoir_state",
        },
        "sac": {
            "baseline_implementation": "stable_baselines3.SAC",
            "sb3_reservoir_policy": "MlpPolicy",
            "buffer_size": args.buffer_size,
            "learning_starts": args.learning_starts,
            "batch_size": args.batch_size,
            "sequence_batch_size": args.sequence_batch_size,
            "burn_in": args.burn_in,
            "sequence_length": args.sequence_length,
            "gradient_steps_per_transition": args.gradient_steps_per_transition,
            "matched_recurrent_heads": {
                "actor_hidden_sizes": [128],
                "critic_hidden_sizes": [128, 128],
            },
            "device": args.device,
        },
    }
    if not args.dry_run:
        args.result_root.mkdir(parents=True, exist_ok=True)
        (args.result_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )

    status(
        f"Running {len(selected)} conditions x {len(args.seeds)} seeds x "
        f"{len(args.controllers)} controllers ({', '.join(args.controllers)}); "
        f"reservoir={args.reservoir_units} units, sr={args.spectral_radius:g}, "
        f"leak={args.leaking_rate:g}, input_scale={args.input_scaling:g}"
    )
    for condition in selected:
        status(f"CONDITION {condition.name}: {condition.description}")
        episode_files = []
        for seed in args.seeds:
            baseline_checkpoint = (
                train_baseline_one(args, condition, seed)
                if args.retrain_baseline
                else args.baseline_root
                / condition.name
                / f"regular_sac_seed_{seed}.zip"
            )
            controller_checkpoints: dict[str, Path] = {}
            if "sb3_reservoir" in args.controllers:
                controller_checkpoints["sb3_reservoir"] = train_one(
                    args, condition, seed
                )
            for encoder in ("lstm", "esn"):
                if encoder in args.controllers:
                    checkpoint = train_recurrent_one(
                        args, condition, encoder, seed
                    )
                    if args.recurrent_checkpoint == "latest":
                        latest = checkpoint.with_name(
                            f"{checkpoint.stem}_latest{checkpoint.suffix}"
                        )
                        if not latest.is_file():
                            raise FileNotFoundError(
                                f"Requested latest checkpoint does not exist: {latest}"
                            )
                        checkpoint = latest
                    controller_checkpoints[encoder] = checkpoint
            if not args.skip_evaluation:
                episode_files.append(
                    evaluate_one(
                        args,
                        condition,
                        seed,
                        baseline_checkpoint,
                        controller_checkpoints,
                    )
                )
        if episode_files and not args.dry_run:
            aggregate_condition(args, condition, episode_files)
    status("ALL RECURRENT-CONTROLLER EXPERIMENTS COMPLETE")


if __name__ == "__main__":
    main()
