"""Compare and animate SB3 and ReservoirPy on paired Duffing episodes."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

from agent import load_agent
from agentrc import ReservoirPolicy
from environment import DuffingBoundaryEnv
from indicators import basin_label


def run_episode(model, env, seed: int, reset_policy: bool = False):
    observation, _ = env.reset(seed=seed)
    if reset_policy:
        model.reset()

    states = [env.state.copy()]
    params_history = [dict(env.params)]
    forcing_times = [env._forcing_time()]
    rewards = []
    distances = []
    done = False
    while not done:
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        states.append(env.state.copy())
        params_history.append(dict(env.params))
        forcing_times.append(env._forcing_time())
        rewards.append(float(reward))
        distances.append(float(info["boundary_distance"]))
        done = terminated or truncated

    return {
        "states": np.asarray(states),
        "params_history": params_history,
        "forcing_times": np.asarray(forcing_times),
        "rewards": np.asarray(rewards),
        "distances": np.asarray(distances),
        "return": float(np.sum(rewards)),
        "mean_distance": float(np.mean(distances)),
    }


def estimate_basin_map(
    params: dict[str, float],
    forcing_time: float,
    xs: np.ndarray,
    vs: np.ndarray,
) -> np.ndarray:
    """Estimate the instantaneous finite-time basin map for one frame."""
    labels = np.empty((len(vs), len(xs)))
    for row, velocity in enumerate(vs):
        for column, position in enumerate(xs):
            labels[row, column] = basin_label(
                (position, velocity),
                params,
                horizon=6.0,
                dt=0.08,
                start_time=forcing_time,
            )
    return labels


def save_comparison_animation(
    sb3_run,
    reservoir_run,
    algorithm: str,
    output: Path,
    resolution: int,
    stride: int,
    fps: int,
) -> float:
    """Animate paired policies and their running phase-space difference."""
    runs = {algorithm.upper(): sb3_run, "Reservoir": reservoir_run}
    colors = {algorithm.upper(): "tab:blue", "Reservoir": "tab:orange"}
    longest = max(len(run["states"]) for run in runs.values())
    frame_indices = list(range(0, longest, stride))
    if frame_indices[-1] != longest - 1:
        frame_indices.append(longest - 1)

    paired_length = min(
        len(sb3_run["states"]),
        len(reservoir_run["states"]),
    )
    state_differences = np.linalg.norm(
        sb3_run["states"][:paired_length]
        - reservoir_run["states"][:paired_length],
        axis=1,
    )
    mean_state_difference = float(np.mean(state_differences))

    xs = np.linspace(-1.6, 1.6, resolution)
    vs = np.linspace(-1.8, 1.8, resolution)
    map_cache: dict[tuple[tuple[tuple[str, float], ...], float], np.ndarray] = {}
    all_states = np.concatenate([run["states"] for run in runs.values()])
    x_limit = max(1.6, float(np.max(np.abs(all_states[:, 0]))) + 0.15)
    v_limit = max(1.8, float(np.max(np.abs(all_states[:, 1]))) + 0.15)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), sharex=True, sharey=True)
    title = fig.suptitle("")

    def draw(frame_number: int):
        step = frame_indices[frame_number]
        artists = [title]
        for axis, (model_name, run) in zip(axes, runs.items()):
            index = min(step, len(run["states"]) - 1)
            params = run["params_history"][index]
            forcing_time = float(run["forcing_times"][index])
            signature = tuple(
                sorted((name, float(value)) for name, value in params.items())
            )
            cache_key = (signature, round(forcing_time, 12))
            if cache_key not in map_cache:
                map_cache[cache_key] = estimate_basin_map(
                    params,
                    forcing_time,
                    xs,
                    vs,
                )

            axis.clear()
            labels = map_cache[cache_key]
            axis.contourf(
                xs,
                vs,
                labels,
                levels=[-1.5, 0.0, 1.5],
                alpha=0.35,
                cmap="coolwarm",
            )
            axis.contour(
                xs,
                vs,
                labels,
                levels=[0.0],
                colors="black",
                linewidths=1.2,
            )
            states = run["states"]
            axis.plot(
                states[: index + 1, 0],
                states[: index + 1, 1],
                color=colors[model_name],
                lw=1.8,
            )
            axis.scatter(
                states[index, 0],
                states[index, 1],
                color=colors[model_name],
                s=55,
                zorder=3,
            )
            completed_steps = min(index, len(run["distances"]))
            mean_score = (
                float(np.mean(run["distances"][:completed_steps]))
                if completed_steps
                else float("nan")
            )
            axis.set(
                xlim=(-x_limit, x_limit),
                ylim=(-v_limit, v_limit),
                xlabel="x",
                ylabel="v",
                title=(
                    f"{model_name} | mean boundary score={mean_score:.3f}\n"
                    f"alpha={params['alpha']:.3f}, beta={params['beta']:.3f}, "
                    f"damping={params['damping']:.3f}"
                ),
            )
            artists.extend(axis.collections + axis.lines)

        difference_index = min(step, paired_length - 1)
        current_difference = float(state_differences[difference_index])
        running_difference = float(
            np.mean(state_differences[: difference_index + 1])
        )
        title.set_text(
            f"Paired Duffing tracking — step {step} | "
            f"current state difference={current_difference:.4f} | "
            f"running mean difference={running_difference:.4f} | "
            f"full mean={mean_state_difference:.4f}"
        )
        return artists

    animation = FuncAnimation(fig, draw, frames=len(frame_indices), blit=False)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = (
        FFMpegWriter(fps=fps)
        if output.suffix.lower() == ".mp4"
        else PillowWriter(fps=fps)
    )
    animation.save(output, writer=writer, dpi=130)
    plt.close(fig)
    return mean_state_difference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sb3-model",
        type=Path,
        default=Path("models/adaptive_duffing_sac"),
    )
    parser.add_argument(
        "--reservoir-model",
        type=Path,
        default=Path("models/adaptive_duffing_reservoir.pkl"),
    )
    parser.add_argument("--algorithm", choices=("sac", "ppo"), default="sac")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--evolve-dynamics", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--use-basin-mixing", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--parameter-walk-interval", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("model_comparison.png"))
    parser.add_argument(
        "--animate-output",
        type=Path,
        help="Optional side-by-side Duffing comparison as a .gif or .mp4.",
    )
    parser.add_argument("--animation-stride", type=int, default=5)
    parser.add_argument("--animation-resolution", type=int, default=31)
    parser.add_argument("--animation-fps", type=int, default=8)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be at least 1")
    if args.parameter_walk_interval < 0:
        parser.error("--parameter-walk-interval cannot be negative")
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

    sb3_file = args.sb3_model.with_suffix(".zip")
    if not sb3_file.exists():
        raise FileNotFoundError(
            f"Trained SB3 model not found at {sb3_file}. Run `python src/train.py` first."
        )
    reservoir_file = args.reservoir_model.with_suffix(".pkl")
    if not reservoir_file.exists():
        raise FileNotFoundError(
            "Trained reservoir model not found at "
            f"{reservoir_file}. Run `python src/trainrc.py` first."
        )

    env_options = {
        "parameter_walk": args.parameter_walk_interval > 0,
        "parameter_walk_interval_steps": max(1, args.parameter_walk_interval),
        "evolve_dynamics": args.evolve_dynamics,
        "use_basin_mixing": args.use_basin_mixing,
        "boundary_update_interval": 1,
    }
    sb3_env = DuffingBoundaryEnv(**env_options)
    reservoir_env = DuffingBoundaryEnv(**env_options)
    sb3 = load_agent(args.sb3_model, env=sb3_env, algorithm=args.algorithm)
    reservoir = ReservoirPolicy.load(args.reservoir_model)

    seeds = np.arange(args.seed, args.seed + args.episodes)
    sb3_runs = []
    rc_runs = []
    for seed in seeds:
        sb3_runs.append(run_episode(sb3, sb3_env, int(seed)))
        rc_runs.append(run_episode(reservoir, reservoir_env, int(seed), True))

    sb3_returns = np.array([run["return"] for run in sb3_runs])
    rc_returns = np.array([run["return"] for run in rc_runs])
    sb3_distances = np.array([run["mean_distance"] for run in sb3_runs])
    rc_distances = np.array([run["mean_distance"] for run in rc_runs])
    metric_title = (
        "Mean finite-time boundary score (lower is better)"
        if args.use_basin_mixing
        else "Mean global curved-boundary distance (lower is better)"
    )
    metric_ylabel = (
        "1 - local basin mixing"
        if args.use_basin_mixing
        else "nearest numerical-boundary distance"
    )
    metric_label = (
        "finite-time boundary score"
        if args.use_basin_mixing
        else "global curved-boundary distance"
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Overlay the first paired paths on the actual initial finite-time boundary,
    # rather than on a tangent line forced through the origin.
    first_sb3 = sb3_runs[0]["states"]
    first_rc = rc_runs[0]["states"]
    initial_xs = np.linspace(-1.6, 1.6, 31)
    initial_vs = np.linspace(-1.8, 1.8, 31)
    initial_basin = estimate_basin_map(
        sb3_runs[0]["params_history"][0],
        float(sb3_runs[0]["forcing_times"][0]),
        initial_xs,
        initial_vs,
    )
    axes[0, 0].contourf(
        initial_xs,
        initial_vs,
        initial_basin,
        levels=[-1.5, 0.0, 1.5],
        alpha=0.35,
        cmap="coolwarm",
    )
    axes[0, 0].contour(
        initial_xs,
        initial_vs,
        initial_basin,
        levels=[0.0],
        colors="black",
        linewidths=1.2,
    )
    axes[0, 0].plot(first_sb3[:, 0], first_sb3[:, 1], label="SAC/PPO", lw=2)
    axes[0, 0].plot(first_rc[:, 0], first_rc[:, 1], label="Reservoir", lw=2)
    axes[0, 0].set(
        title=f"Initial finite-time boundary and paths (seed {seeds[0]})",
        xlabel="x",
        ylabel="v",
    )
    axes[0, 0].legend()

    episode_numbers = np.arange(1, args.episodes + 1)
    axes[0, 1].plot(episode_numbers, sb3_returns, "o-", label="SAC/PPO")
    axes[0, 1].plot(episode_numbers, rc_returns, "o-", label="Reservoir")
    axes[0, 1].set(title="Return on identical test seeds", xlabel="episode", ylabel="return")
    axes[0, 1].legend()

    axes[1, 0].plot(episode_numbers, sb3_distances, "o-", label="SAC/PPO")
    axes[1, 0].plot(episode_numbers, rc_distances, "o-", label="Reservoir")
    axes[1, 0].set(
        title=metric_title,
        xlabel="episode",
        ylabel=metric_ylabel,
    )
    axes[1, 0].legend()

    axes[1, 1].scatter(sb3_distances, sb3_returns, label="SAC/PPO", s=45)
    axes[1, 1].scatter(rc_distances, rc_returns, label="Reservoir", s=45)
    axes[1, 1].set(
        title="Reward versus boundary metric",
        xlabel=metric_title.lower(),
        ylabel="return (higher is better)",
    )
    axes[1, 1].legend()

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)

    animation_mean_difference = None
    if args.animate_output:
        animation_mean_difference = save_comparison_animation(
            sb3_runs[0],
            rc_runs[0],
            args.algorithm,
            args.animate_output,
            args.animation_resolution,
            args.animation_stride,
            args.animation_fps,
        )
        print(f"Saved side-by-side animation to {args.animate_output}")

    print(f"SAC/PPO mean return:          {sb3_returns.mean():.3f} +/- {sb3_returns.std():.3f}")
    print(f"Reservoir mean return:        {rc_returns.mean():.3f} +/- {rc_returns.std():.3f}")
    print(f"SAC/PPO mean {metric_label}:  {sb3_distances.mean():.4f}")
    print(f"Reservoir mean {metric_label}: {rc_distances.mean():.4f}")
    print(
        "Mean paired return difference (SB3 - Reservoir): "
        f"{np.mean(sb3_returns - rc_returns):.4f}"
    )
    print(
        f"Mean paired {metric_label} difference (SB3 - Reservoir): "
        f"{np.mean(sb3_distances - rc_distances):.4f}"
    )
    if animation_mean_difference is not None:
        print(
            "Animated episode mean phase-space difference: "
            f"{animation_mean_difference:.4f}"
        )
    print(f"Saved comparison to {args.output}")
    sb3_env.close()
    reservoir_env.close()


if __name__ == "__main__":
    main()
