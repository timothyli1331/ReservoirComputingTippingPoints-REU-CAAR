"""Plot matched evaluation paths for SAC, LSTM-SAC, and ESN-SAC.

The existing controller-comparison CSV files contain episode-level metrics,
not the time-resolved state path.  This script reloads the saved checkpoints
and replays one deterministic evaluation episode for every controller using
the same condition, system, and evaluation seed.

Example:

    .venv/bin/python src/plot_recurrent_paths.py \
        --condition no_stretch \
        --system duffing \
        --training-seed 0 \
        --episode 0 \
        --output results/recurrent_controller_tests/no_stretch/paths_duffing_seed0.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import SAC

from recurrent_sac import RecurrentSACAgent
from run_sb3_reservoir_input_overnight import (
    CONDITIONS,
    DEFAULT_SYSTEM_NAMES,
    make_base_env,
)
from evaluate_transfer import estimate_system_boundary
from systems import get_system


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot matched state paths for recurrent SAC controllers."
    )
    parser.add_argument("--condition", choices=[c.name for c in CONDITIONS], required=True)
    parser.add_argument(
        "--system",
        choices=("all", *DEFAULT_SYSTEM_NAMES),
        default="all",
        help="System to plot, or 'all' for the four-column comparison.",
    )
    parser.add_argument("--training-seed", type=int, default=0)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of evaluation episodes used for the return bars.",
    )
    parser.add_argument(
        "--baseline-root", type=Path, default=Path("models/overnight_memory_tests")
    )
    parser.add_argument(
        "--model-root", type=Path, default=Path("models/sb3_reservoir_input_tests")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--recurrent-checkpoint",
        choices=("best", "latest"),
        default="latest",
        help="Use the validation-selected or final LSTM/ESN checkpoint.",
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def record_path(model, env, system: str, evaluation_seed: int, recurrent: bool):
    observation, _ = env.reset(seed=evaluation_seed, options={"system_name": system})
    if recurrent:
        model.reset()

    states = [np.asarray(env.unwrapped.state, dtype=float).copy()]
    rewards = []
    boundary_distances = []
    done = False
    while not done:
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        states.append(np.asarray(env.unwrapped.state, dtype=float).copy())
        rewards.append(float(reward))
        boundary_distances.append(float(info["boundary_distance"]))
        done = bool(terminated or truncated)
    return (
        np.asarray(states),
        float(np.sum(rewards)),
        float(np.mean(boundary_distances)),
    )


def main() -> None:
    args = parse_args()
    condition = next(c for c in CONDITIONS if c.name == args.condition)
    suffix = "_latest" if args.recurrent_checkpoint == "latest" else ""

    checkpoints = {
        "SAC": args.baseline_root / condition.name / f"regular_sac_seed_{args.training_seed}.zip",
        "LSTM-SAC": args.model_root / condition.name / f"lstm_sac_seed_{args.training_seed}{suffix}.pt",
        "ESN-SAC": args.model_root / condition.name / f"esn_sac_seed_{args.training_seed}{suffix}.pt",
    }
    missing = [str(path) for path in checkpoints.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing checkpoint(s):\n" + "\n".join(missing))

    models = {
        "SAC": (SAC.load(checkpoints["SAC"], device=args.device), False),
        "LSTM-SAC": (
            RecurrentSACAgent.load(checkpoints["LSTM-SAC"], device=args.device),
            True,
        ),
        "ESN-SAC": (
            RecurrentSACAgent.load(checkpoints["ESN-SAC"], device=args.device),
            True,
        ),
    }

    systems = DEFAULT_SYSTEM_NAMES if args.system == "all" else (args.system,)
    paths = {system: {} for system in systems}
    metrics = {system: {name: [] for name in models} for system in systems}
    for system in systems:
        for name, (model, recurrent) in models.items():
            for episode in range(args.episodes):
                env = make_base_env(
                    condition,
                    system_names=(system,),
                    selection="round_robin",
                    boundary_update_interval=1,
                )
                episode_seed = 2_000_000 + args.training_seed * 100 + episode
                states, episode_return, boundary_distance = record_path(
                    model, env, system, episode_seed, recurrent
                )
                env.close()
                metrics[system][name].append(
                    (episode_return, boundary_distance)
                )
                if episode == args.episode:
                    paths[system][name] = states

    colors = {"SAC": "tab:blue", "LSTM-SAC": "tab:green", "ESN-SAC": "tab:red"}
    fig, axes = plt.subplots(
        2,
        len(systems),
        figsize=(4.3 * len(systems), 8.0),
        squeeze=False,
    )
    for column, system in enumerate(systems):
        metric_axis = axes[0, column]
        trajectory_axis = axes[1, column]
        names = list(models)
        positions = np.arange(len(names))
        return_means = [np.mean([item[0] for item in metrics[system][name]]) for name in names]
        return_stds = [np.std([item[0] for item in metrics[system][name]]) for name in names]
        boundary_means = [np.mean([item[1] for item in metrics[system][name]]) for name in names]
        metric_axis.bar(
            positions,
            return_means,
            yerr=return_stds,
            color=[colors[name] for name in names],
            alpha=0.82,
            capsize=4,
        )
        metric_axis.set_xticks(positions, names, rotation=25, ha="right", fontsize=8)
        metric_axis.set_title(get_system(system).label)
        metric_axis.set_ylabel("return (higher is better)")
        for position, value, boundary_distance in zip(
            positions, return_means, boundary_means
        ):
            metric_axis.text(
                position,
                value,
                f"b={boundary_distance:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        boundary_points = estimate_system_boundary(get_system(system), resolution=41)
        trajectory_axis.scatter(
            boundary_points[:, 0],
            boundary_points[:, 1],
            s=5,
            color="black",
            alpha=0.5,
            label="initial numerical boundary",
        )
        for name in names:
            path = paths[system][name]
            trajectory_axis.plot(
                path[:, 0], path[:, 1], color=colors[name], label=name, lw=1.5
            )
            trajectory_axis.scatter(
                path[0, 0], path[0, 1], color=colors[name], marker="o", s=22
            )
            trajectory_axis.scatter(
                path[-1, 0], path[-1, 1], color=colors[name], marker="x", s=30
            )
        trajectory_axis.set(xlabel="position $x$", ylabel="velocity $v$", title="paired trajectory")
        trajectory_axis.legend(fontsize=7, loc="best")
        trajectory_axis.grid(alpha=0.2)
        metric_axis.grid(axis="y", alpha=0.2)

    fig.suptitle(
        f"Recurrent-controller comparison --- {args.condition}, "
        f"training seed {args.training_seed}"
    )
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {args.output}")
    for system in systems:
        for name in models:
            values = metrics[system][name]
            print(
                f"{system} / {name}: return={np.mean([v[0] for v in values]):.3f}, "
                f"boundary={np.mean([v[1] for v in values]):.4f}"
            )


if __name__ == "__main__":
    main()
