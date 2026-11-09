"""Evaluate SAC/PPO and ReservoirPy policies on each double-well system."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

from agent import load_agent
from agentrc import ReservoirPolicy
from boundary import GlobalBoundary
from multi_environment import MultiDoubleWellEnv
from reservoir_sac import ReservoirSACAgent
from systems import (
    DEFAULT_SYSTEM_NAMES,
    SYSTEMS,
    DoubleWellSystem,
    finite_time_basin_label,
    get_system,
    parse_system_names,
    velocity_scale,
)


def run_episode(
    model,
    system_name: str,
    seed: int,
    include_system_id: bool,
    include_local_acceleration: bool,
    include_local_stretching: bool,
    recurrent: bool,
    evolve_dynamics: bool,
    dynamics_substeps: int,
    use_basin_mixing: bool,
    parameter_walk_interval: int = 0,
    parameter_walk_scale: float = 1.0,
    boundary_start_probability: float = 0.6,
):
    env = MultiDoubleWellEnv(
        system_names=(system_name,),
        include_system_id=include_system_id,
        include_local_acceleration=include_local_acceleration,
        include_local_stretching=include_local_stretching,
        parameter_walk=parameter_walk_interval > 0,
        parameter_walk_interval_steps=max(1, parameter_walk_interval),
        parameter_walk_scale=parameter_walk_scale,
        boundary_start_probability=boundary_start_probability,
        evolve_dynamics=evolve_dynamics,
        dynamics_substeps=dynamics_substeps,
        use_basin_mixing=use_basin_mixing,
        boundary_update_interval=1,
    )
    observation, _ = env.reset(seed=seed, options={"system_name": system_name})
    if recurrent:
        model.reset()
    states = [env.state.copy()]
    system_history = [env.current_system]
    rewards = []
    distances = []
    done = False
    while not done:
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        states.append(env.state.copy())
        system_history.append(env.current_system)
        rewards.append(float(reward))
        distances.append(float(info["boundary_distance"]))
        done = terminated or truncated
    env.close()
    return {
        "states": np.asarray(states),
        "system_history": system_history,
        "return": float(np.sum(rewards)),
        "mean_boundary_score": float(np.mean(distances)),
        "survived_steps": len(rewards),
    }


def estimate_system_boundary(
    system: DoubleWellSystem,
    resolution: int,
) -> np.ndarray:
    """Calculate an instantaneous numerical boundary for visualization."""
    boundary = GlobalBoundary(
        (-1.6 * system.well_location, 1.6 * system.well_location),
        (-1.8 * velocity_scale(system), 1.8 * velocity_scale(system)),
        resolution,
    )
    rng = np.random.default_rng(17_071 + list(SYSTEMS).index(system.name))
    boundary.rebuild(
        lambda state: finite_time_basin_label(
            state,
            system,
            horizon=6.0,
            dt=0.08,
            rng=rng if system.noise_std > 0 else None,
        )
    )
    return boundary.points


def format_system_parameters(system: DoubleWellSystem) -> str:
    """Return a compact label for the parameters currently being animated."""
    if system.kind == "quartic":
        values = [
            f"alpha={-system.params['linear']:.3f}",
            f"beta={system.params['cubic']:.3f}",
            f"damping={system.params['damping']:.3f}",
        ]
    else:
        names = {
            "sextic": ("linear", "cubic", "quintic", "damping"),
            "smooth_biquadratic": (
                "stiffness",
                "separation",
                "epsilon",
                "damping",
            ),
        }[system.kind]
        values = [f"{name}={system.params[name]:.3f}" for name in names]
    if system.noise_std > 0:
        values.append(f"noise={system.noise_std:.3f}")
    return ", ".join(values)


def save_transfer_animation(
    first_runs,
    system_names: tuple[str, ...],
    model_names: tuple[str, ...],
    colors: dict[str, str],
    output: Path,
    stride: int,
    fps: int,
    resolution: int,
) -> None:
    """Animate the paired policy trajectories on every selected system."""
    longest_run = max(
        len(first_runs[(system_name, model_name)]["states"])
        for system_name in system_names
        for model_name in model_names
    )
    frame_indices = list(range(0, longest_run, stride))
    if frame_indices[-1] != longest_run - 1:
        frame_indices.append(longest_run - 1)

    fig, axes = plt.subplots(
        1,
        len(system_names),
        figsize=(4.8 * len(system_names), 4.8),
        squeeze=False,
    )
    trajectory_artists = {}
    boundary_artists = {}
    axes_by_system = {}
    boundary_cache = {}
    reference_runs = {
        system_name: max(
            (first_runs[(system_name, model_name)] for model_name in model_names),
            key=lambda run: len(run["states"]),
        )
        for system_name in system_names
    }
    for column, system_name in enumerate(system_names):
        axis = axes[0, column]
        axes_by_system[system_name] = axis
        reference_run = reference_runs[system_name]
        points = estimate_system_boundary(
            reference_run["system_history"][0], resolution
        )
        boundary_cache[(system_name, 0)] = points
        boundary_artists[system_name] = axis.scatter(
            points[:, 0],
            points[:, 1],
            s=5,
            color="black",
            alpha=0.45,
            label="numerical boundary",
        )

        all_states = np.concatenate(
            [
                first_runs[(system_name, model_name)]["states"]
                for model_name in model_names
            ]
        )
        systems = reference_run["system_history"]
        x_extent = max(1.7 * system.well_location for system in systems)
        v_extent = max(1.9 * velocity_scale(system) for system in systems)
        plotted_points = np.concatenate((points, all_states), axis=0)
        x_padding = max(0.1, 0.08 * np.ptp(plotted_points[:, 0]))
        v_padding = max(0.1, 0.08 * np.ptp(plotted_points[:, 1]))
        axis.set(
            xlim=(
                min(plotted_points[:, 0].min() - x_padding, -x_extent),
                max(plotted_points[:, 0].max() + x_padding, x_extent),
            ),
            ylim=(
                min(plotted_points[:, 1].min() - v_padding, -v_extent),
                max(plotted_points[:, 1].max() + v_padding, v_extent),
            ),
            xlabel="x",
            ylabel="v",
        )
        initial_system = reference_run["system_history"][0]
        axis.set_title(
            f"{initial_system.label}\n{format_system_parameters(initial_system)}",
            fontsize=9,
        )
        for model_name in model_names:
            (line,) = axis.plot(
                [], [], color=colors[model_name], lw=1.8, label=model_name
            )
            marker = axis.scatter([], [], color=colors[model_name], s=48, zorder=3)
            trajectory_artists[(system_name, model_name)] = (line, marker)
        axis.legend(fontsize=8, loc="best")

    title = fig.suptitle("Multi-system policy trajectories — step 0")
    fig.tight_layout()

    def draw(frame_number: int):
        step = frame_indices[frame_number]
        artists = [title]
        for system_name in system_names:
            reference_run = reference_runs[system_name]
            reference_index = min(step, len(reference_run["states"]) - 1)
            cache_key = (system_name, reference_index)
            if cache_key not in boundary_cache:
                boundary_cache[cache_key] = estimate_system_boundary(
                    reference_run["system_history"][reference_index], resolution
                )
            boundary = boundary_cache[cache_key]
            boundary_artist = boundary_artists[system_name]
            boundary_artist.set_offsets(boundary)
            system = reference_run["system_history"][reference_index]
            axes_by_system[system_name].set_title(
                f"{system.label}\n{format_system_parameters(system)}",
                fontsize=9,
            )
            artists.append(boundary_artist)
            for model_name in model_names:
                states = first_runs[(system_name, model_name)]["states"]
                index = min(step, len(states) - 1)
                line, marker = trajectory_artists[(system_name, model_name)]
                line.set_data(states[: index + 1, 0], states[: index + 1, 1])
                marker.set_offsets(states[index : index + 1])
                artists.extend((line, marker))
        title.set_text(f"Multi-system policy trajectories — step {step}")
        return artists

    animation = FuncAnimation(fig, draw, frames=len(frame_indices), blit=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = (
        FFMpegWriter(fps=fps)
        if output.suffix.lower() == ".mp4"
        else PillowWriter(fps=fps)
    )
    animation.save(output, writer=writer, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("sac", "ppo"), default="sac")
    parser.add_argument("--sb3-model", type=Path, default=Path("models/multi_sac"))
    parser.add_argument(
        "--reservoir-model", type=Path, default=Path("models/multi_reservoir.pkl")
    )
    parser.add_argument(
        "--reservoir-sac-model",
        type=Path,
        default=None,
        help="Compare against a reservoir-SAC .pt checkpoint instead of CEM reservoir.",
    )
    parser.add_argument("--systems", default=",".join(DEFAULT_SYSTEM_NAMES))
    parser.add_argument("--include-system-id", action="store_true")
    parser.add_argument(
        "--include-local-acceleration",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--include-local-stretching",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--evolve-dynamics", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--dynamics-substeps",
        type=int,
        default=1,
        help="Full physical-dynamics intervals applied after each agent action.",
    )
    parser.add_argument(
        "--use-basin-mixing", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument(
        "--boundary-start-probability",
        type=float,
        default=0.6,
        help="Evaluation probability of starting near the boundary.",
    )
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--output", type=Path, default=Path("transfer_comparison.png"))
    parser.add_argument("--csv", type=Path, default=Path("transfer_results.csv"))
    parser.add_argument(
        "--animate-output",
        type=Path,
        help="Optional .gif or .mp4 showing paired trajectories on all systems.",
    )
    parser.add_argument(
        "--evaluation-parameter-walk-interval",
        type=int,
        default=1,
        help=(
            "Change each system's parameters every N control steps; use 0 to disable."
        ),
    )
    parser.add_argument(
        "--evaluation-parameter-walk-scale",
        type=float,
        default=1.0,
        help="Multiplier for each evaluation-time parameter-walk increment.",
    )
    parser.add_argument("--animation-stride", type=int, default=5)
    parser.add_argument("--animation-resolution", type=int, default=25)
    parser.add_argument("--animation-fps", type=int, default=10)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be at least 1")
    if args.evaluation_parameter_walk_interval < 0:
        parser.error("--evaluation-parameter-walk-interval cannot be negative")
    if args.evaluation_parameter_walk_scale < 0:
        parser.error("--evaluation-parameter-walk-scale must be nonnegative")
    if args.dynamics_substeps < 1:
        parser.error("--dynamics-substeps must be at least 1")
    if not 0.0 <= args.boundary_start_probability <= 1.0:
        parser.error("--boundary-start-probability must be in [0, 1]")
    if (
        args.animation_stride < 1
        or args.animation_resolution < 10
        or args.animation_fps < 1
    ):
        parser.error("animation stride/fps must be positive and resolution at least 10")
    if args.animate_output and args.animate_output.suffix.lower() not in {
        ".gif",
        ".mp4",
    }:
        parser.error("--animate-output must end in .gif or .mp4")

    system_names = parse_system_names(args.systems)
    probe_env = MultiDoubleWellEnv(
        include_system_id=args.include_system_id,
        include_local_acceleration=args.include_local_acceleration,
        include_local_stretching=args.include_local_stretching,
        parameter_walk=args.evaluation_parameter_walk_interval > 0,
        parameter_walk_interval_steps=max(
            1, args.evaluation_parameter_walk_interval
        ),
        parameter_walk_scale=args.evaluation_parameter_walk_scale,
        boundary_start_probability=args.boundary_start_probability,
        evolve_dynamics=args.evolve_dynamics,
        dynamics_substeps=args.dynamics_substeps,
        use_basin_mixing=args.use_basin_mixing,
        boundary_update_interval=1,
    )
    sb3 = load_agent(args.sb3_model, env=probe_env, algorithm=args.algorithm)
    if args.reservoir_sac_model is not None:
        reservoir = ReservoirSACAgent.load(args.reservoir_sac_model)
        reservoir_name = "Reservoir-SAC"
    else:
        reservoir = ReservoirPolicy.load(args.reservoir_model)
        reservoir_name = "Reservoir"
    models = {
        args.algorithm.upper(): (sb3, False),
        reservoir_name: (reservoir, True),
    }
    walk_interval = args.evaluation_parameter_walk_interval
    metric_label = (
        "finite-time boundary score"
        if args.use_basin_mixing
        else "global curved-boundary distance"
    )

    results = []
    first_runs = {}
    for system_name in system_names:
        for model_name, (model, recurrent) in models.items():
            for episode in range(args.episodes):
                run = run_episode(
                    model,
                    system_name,
                    args.seed + episode,
                    args.include_system_id,
                    args.include_local_acceleration,
                    args.include_local_stretching,
                    recurrent,
                    args.evolve_dynamics,
                    args.dynamics_substeps,
                    args.use_basin_mixing,
                    walk_interval,
                    args.evaluation_parameter_walk_scale,
                    args.boundary_start_probability,
                )
                results.append(
                    {
                        "system": system_name,
                        "model": model_name,
                        "episode": episode,
                        "seed": args.seed + episode,
                        "return": run["return"],
                        "mean_boundary_score": run["mean_boundary_score"],
                        "survived_steps": run["survived_steps"],
                    }
                )
                if episode == 0:
                    first_runs[(system_name, model_name)] = run

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    fig, axes = plt.subplots(2, len(system_names), figsize=(4.3 * len(system_names), 8))
    if len(system_names) == 1:
        axes = np.asarray(axes).reshape(2, 1)
    colors = {args.algorithm.upper(): "tab:blue", reservoir_name: "tab:orange"}
    for column, system_name in enumerate(system_names):
        system = get_system(system_name)
        metric_axis = axes[0, column]
        trajectory_axis = axes[1, column]
        model_names = list(models)
        return_means = []
        return_stds = []
        distance_means = []
        for model_name in model_names:
            rows = [
                row
                for row in results
                if row["system"] == system_name and row["model"] == model_name
            ]
            return_means.append(np.mean([row["return"] for row in rows]))
            return_stds.append(np.std([row["return"] for row in rows]))
            distance_means.append(np.mean([row["mean_boundary_score"] for row in rows]))
        positions = np.arange(len(model_names))
        metric_axis.bar(
            positions,
            return_means,
            yerr=return_stds,
            color=[colors[name] for name in model_names],
            alpha=0.8,
        )
        metric_axis.set_xticks(positions, model_names)
        metric_axis.set_title(system.label)
        metric_axis.set_ylabel("return (higher is better)")
        for position, distance in zip(positions, distance_means):
            metric_axis.text(
                position,
                return_means[position],
                f"b={distance:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        for model_name in model_names:
            states = first_runs[(system_name, model_name)]["states"]
            trajectory_axis.plot(
                states[:, 0], states[:, 1], label=model_name, color=colors[model_name]
            )
        reference_model = next(iter(models))
        initial_system = first_runs[(system_name, reference_model)]["system_history"][0]
        boundary_points = estimate_system_boundary(initial_system, resolution=41)
        trajectory_axis.scatter(
            boundary_points[:, 0],
            boundary_points[:, 1],
            s=5,
            color="black",
            alpha=0.55,
            label=(
                "initial numerical boundary"
                if walk_interval
                else "numerical curved boundary"
            ),
        )
        trajectory_axis.set(xlabel="x", ylabel="v", title="paired trajectory")
        trajectory_axis.legend(fontsize=8)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)

    if args.animate_output:
        save_transfer_animation(
            first_runs,
            system_names,
            tuple(models),
            colors,
            args.animate_output,
            args.animation_stride,
            args.animation_fps,
            args.animation_resolution,
        )
        print(f"Saved multi-system animation to {args.animate_output}")

    probe_env.close()

    print(f"Saved comparison plot to {args.output}")
    print(f"Saved episode-level results to {args.csv}")
    print(
        f"Evaluation mode: dynamics={'on' if args.evolve_dynamics else 'off'}, "
        f"dynamics substeps={args.dynamics_substeps}, "
        f"metric={metric_label}, "
        f"parameter walk={'off' if not walk_interval else f'every {walk_interval} steps'}"
    )
    for system_name in system_names:
        print(f"\n{get_system(system_name).label}")
        for model_name in models:
            rows = [
                row
                for row in results
                if row["system"] == system_name and row["model"] == model_name
            ]
            mean_return = np.mean([row["return"] for row in rows])
            mean_distance = np.mean([row["mean_boundary_score"] for row in rows])
            print(
                f"  {model_name:10s} return={mean_return:8.3f}, "
                f"{metric_label}={mean_distance:.4f}"
            )


if __name__ == "__main__":
    main()
