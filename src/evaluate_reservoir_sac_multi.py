"""Evaluate a trained reservoir-SAC checkpoint on a balanced system suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from evaluate_transfer import estimate_system_boundary, save_transfer_animation
from multi_environment import MultiDoubleWellEnv
from reservoir_sac import ReservoirSACAgent
from systems import DEFAULT_SYSTEM_NAMES, SYSTEMS, get_system, parse_system_names
from train_reservoir_sac_multi import evaluate_agent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/multi_reservoir_sac.pt"),
    )
    parser.add_argument("--metadata-path", type=Path, default=None)
    parser.add_argument("--systems", default=None)
    parser.add_argument(
        "--include-local-acceleration",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--include-local-stretching",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--episodes-per-system", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--boundary-start-probability", type=float, default=None)
    parser.add_argument(
        "--evolve-dynamics", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--dynamics-substeps", type=int, default=None)
    parser.add_argument(
        "--use-basin-mixing", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--parameter-walk", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--parameter-walk-interval", type=int, default=None)
    parser.add_argument("--parameter-walk-scale", type=float, default=None)
    parser.add_argument("--boundary-update-interval", type=int, default=None)
    parser.add_argument("--boundary-resolution", type=int, default=None)
    parser.add_argument("--balance-penalty", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2_000_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("results/multi_reservoir_sac_evaluation.json"),
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=Path("results/multi_reservoir_sac_evaluation.png"),
    )
    parser.add_argument(
        "--gif-path",
        type=Path,
        default=Path("results/multi_reservoir_sac_evaluation.gif"),
    )
    parser.add_argument(
        "--generate-gif", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--animation-stride", type=int, default=10)
    parser.add_argument("--animation-resolution", type=int, default=25)
    parser.add_argument("--animation-fps", type=int, default=8)
    args = parser.parse_args()
    if args.episodes_per_system < 1:
        parser.error("episodes-per-system must be positive")
    if args.dynamics_substeps is not None and args.dynamics_substeps < 1:
        parser.error("dynamics-substeps must be at least 1")
    if (
        args.animation_stride < 1
        or args.animation_resolution < 10
        or args.animation_fps < 1
    ):
        parser.error("animation stride/fps must be positive and resolution at least 10")
    if args.gif_path.suffix.lower() != ".gif":
        parser.error("gif-path must end in .gif")
    return args


def setting(value, configuration: dict, name: str, fallback):
    return value if value is not None else configuration.get(name, fallback)


def record_visual_episode(
    agent: ReservoirSACAgent,
    settings,
    system_name: str,
    seed: int,
) -> dict[str, object]:
    """Record one deterministic trajectory using the evaluation configuration."""
    env = MultiDoubleWellEnv(
        system_names=(system_name,),
        selection="round_robin",
        include_system_id=settings.include_system_id,
        include_local_acceleration=settings.include_local_acceleration,
        include_local_stretching=settings.include_local_stretching,
        parameter_walk=settings.parameter_walk,
        parameter_walk_interval_steps=settings.parameter_walk_interval,
        parameter_walk_scale=settings.parameter_walk_scale,
        boundary_start_probability=settings.boundary_start_probability,
        evolve_dynamics=settings.evolve_dynamics,
        dynamics_substeps=settings.dynamics_substeps,
        use_basin_mixing=settings.use_basin_mixing,
        boundary_update_interval=settings.boundary_update_interval,
        global_boundary_resolution=settings.boundary_resolution,
        max_steps=settings.max_steps,
    )
    observation, _ = env.reset(seed=seed, options={"system_name": system_name})
    agent.reset()
    states = [env.state.copy()]
    system_history = [env.current_system]
    rewards = []
    proximities = []
    done = False
    while not done:
        action, _ = agent.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        states.append(env.state.copy())
        system_history.append(env.current_system)
        rewards.append(float(reward))
        if info["boundary_metric"] == "basin_mixing":
            proximity = float(info["basin_mixing"])
        else:
            proximity = float(np.exp(-float(info["boundary_distance"]) / 0.10))
        proximities.append(proximity)
        done = terminated or truncated
    env.close()
    return {
        "states": np.asarray(states),
        "system_history": system_history,
        "return": float(np.sum(rewards)),
        "boundary_proximity": float(np.mean(proximities)),
        "survived_steps": len(rewards),
    }


def save_static_trajectory_plot(
    runs: dict[tuple[str, str], dict[str, object]],
    system_names: tuple[str, ...],
    model_name: str,
    output: Path,
    boundary_resolution: int,
) -> None:
    """Save one phase-space trajectory and initial boundary for every system."""
    fig, axes = plt.subplots(
        1,
        len(system_names),
        figsize=(4.7 * len(system_names), 4.7),
        squeeze=False,
    )
    for column, system_name in enumerate(system_names):
        axis = axes[0, column]
        run = runs[(system_name, model_name)]
        states = run["states"]
        initial_system = run["system_history"][0]
        boundary = estimate_system_boundary(
            initial_system, resolution=boundary_resolution
        )
        axis.scatter(
            boundary[:, 0],
            boundary[:, 1],
            s=5,
            color="black",
            alpha=0.45,
            label="initial numerical boundary",
        )
        axis.plot(
            states[:, 0],
            states[:, 1],
            color="tab:purple",
            linewidth=1.8,
            label="Reservoir-SAC trajectory",
        )
        axis.scatter(
            states[0, 0], states[0, 1], color="tab:green", s=45, label="start"
        )
        axis.scatter(
            states[-1, 0], states[-1, 1], color="tab:red", s=45, label="end"
        )
        axis.set(
            xlabel="x",
            ylabel="v",
            title=(
                f"{get_system(system_name).label}\n"
                f"return={run['return']:.2f}, "
                f"proximity={run['boundary_proximity']:.3f}"
            ),
        )
        axis.legend(fontsize=7, loc="best")
    fig.suptitle("Reservoir-SAC held-out trajectories")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    metadata_path = args.metadata_path or args.model_path.with_suffix(".json")
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
    configuration = metadata.get("configuration", {})

    system_names = parse_system_names(
        args.systems
        if args.systems is not None
        else configuration.get("systems", DEFAULT_SYSTEM_NAMES)
    )
    agent = ReservoirSACAgent.load(args.model_path, device=args.device)
    include_local_acceleration = setting(
        args.include_local_acceleration,
        configuration,
        "include_local_acceleration",
        True,
    )
    include_local_stretching = setting(
        args.include_local_stretching,
        configuration,
        "include_local_stretching",
        True,
    )
    base_observation_size = (
        2 + int(include_local_acceleration) + int(include_local_stretching)
    )
    include_system_id = agent.encoder.observation_size == base_observation_size + len(SYSTEMS)
    expected_size = base_observation_size + (len(SYSTEMS) if include_system_id else 0)
    if agent.encoder.observation_size != expected_size:
        raise ValueError(
            "checkpoint observation size is incompatible with MultiDoubleWellEnv"
        )

    environment_settings = SimpleNamespace(
        systems=system_names,
        include_local_acceleration=include_local_acceleration,
        include_local_stretching=include_local_stretching,
        boundary_start_probability=setting(
            args.boundary_start_probability,
            configuration,
            "boundary_start_probability",
            0.6,
        ),
        include_system_id=include_system_id,
        parameter_walk=setting(
            args.parameter_walk, configuration, "parameter_walk", True
        ),
        parameter_walk_interval=setting(
            args.parameter_walk_interval,
            configuration,
            "parameter_walk_interval",
            1,
        ),
        parameter_walk_scale=setting(
            args.parameter_walk_scale,
            configuration,
            "parameter_walk_scale",
            1.0,
        ),
        evolve_dynamics=setting(
            args.evolve_dynamics, configuration, "evolve_dynamics", True
        ),
        dynamics_substeps=setting(
            args.dynamics_substeps, configuration, "dynamics_substeps", 1
        ),
        use_basin_mixing=setting(
            args.use_basin_mixing, configuration, "use_basin_mixing", True
        ),
        boundary_update_interval=setting(
            args.boundary_update_interval,
            configuration,
            "boundary_update_interval",
            5,
        ),
        boundary_resolution=setting(
            args.boundary_resolution,
            configuration,
            "boundary_resolution",
            41,
        ),
        max_steps=setting(args.max_steps, configuration, "max_steps", 300),
    )
    env = MultiDoubleWellEnv(
        system_names=environment_settings.systems,
        selection="round_robin",
        include_system_id=environment_settings.include_system_id,
        include_local_acceleration=environment_settings.include_local_acceleration,
        include_local_stretching=environment_settings.include_local_stretching,
        parameter_walk=environment_settings.parameter_walk,
        parameter_walk_interval_steps=environment_settings.parameter_walk_interval,
        parameter_walk_scale=environment_settings.parameter_walk_scale,
        boundary_start_probability=environment_settings.boundary_start_probability,
        evolve_dynamics=environment_settings.evolve_dynamics,
        dynamics_substeps=environment_settings.dynamics_substeps,
        use_basin_mixing=environment_settings.use_basin_mixing,
        boundary_update_interval=environment_settings.boundary_update_interval,
        global_boundary_resolution=environment_settings.boundary_resolution,
        max_steps=environment_settings.max_steps,
    )
    metrics = evaluate_agent(
        agent,
        env,
        system_names,
        episodes_per_system=args.episodes_per_system,
        seed=args.seed,
        balance_penalty=args.balance_penalty,
    )
    env.close()

    model_name = f"Reservoir-SAC ({agent.memory_mode})"
    visual_runs = {
        (system_name, model_name): record_visual_episode(
            agent,
            environment_settings,
            system_name,
            args.seed + 100_000 + system_index,
        )
        for system_index, system_name in enumerate(system_names)
    }
    save_static_trajectory_plot(
        visual_runs,
        system_names,
        model_name,
        args.plot_path,
        args.animation_resolution,
    )
    print(f"Saved trajectory plot to {args.plot_path}")
    if args.generate_gif:
        save_transfer_animation(
            visual_runs,
            system_names,
            (model_name,),
            {model_name: "tab:purple"},
            args.gif_path,
            args.animation_stride,
            args.animation_fps,
            args.animation_resolution,
        )
        print(f"Saved trajectory animation to {args.gif_path}")

    print("system                    return      std   boundary  saturation")
    for system_name in system_names:
        values = metrics["per_system"][system_name]
        print(
            f"{system_name:22s} {values['return']:9.3f} "
            f"{values['return_std']:8.3f} "
            f"{values['boundary_proximity']:9.3f} "
            f"{values['action_saturation']:11.1%}"
        )
    print(f"\nMean return:         {metrics['mean_return']:.3f}")
    print(f"Worst-system return: {metrics['worst_system_return']:.3f}")
    print(f"Balanced score:      {metrics['robust_score']:.3f}")

    report = {
        "model_path": str(args.model_path),
        "memory_mode": agent.memory_mode,
        "metadata_path": str(metadata_path) if metadata_path.exists() else None,
        "episodes_per_system": args.episodes_per_system,
        "seed": args.seed,
        "environment": vars(environment_settings),
        "metrics": metrics,
        "plot_path": str(args.plot_path),
        "gif_path": str(args.gif_path) if args.generate_gif else None,
    }
    report["environment"]["systems"] = list(system_names)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved evaluation report to {args.output_path}")


if __name__ == "__main__":
    main()
