"""Evaluate a Stable-Baselines3 agent and visualize boundary tracking."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

from agent import load_agent
from environment import DuffingBoundaryEnv
from indicators import basin_label


def estimate_basin_map(env: DuffingBoundaryEnv, resolution: int = 90):
    xs = np.linspace(-1.6, 1.6, resolution)
    vs = np.linspace(-1.8, 1.8, resolution)
    labels = np.empty((resolution, resolution))
    for row, v in enumerate(vs):
        for col, x in enumerate(xs):
            labels[row, col] = basin_label(
                (x, v),
                env.params,
                horizon=12.0,
                dt=0.04,
                start_time=env._forcing_time(),
            )
    return xs, vs, labels


def run_episode(
    model,
    env: DuffingBoundaryEnv,
    seed: int,
):
    obs, _ = env.reset(seed=seed)
    states, rewards, distances, stretching = [env.state.copy()], [], [], []
    params_history = [dict(env.params)]
    forcing_times = [env._forcing_time()]
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        states.append(env.state.copy())
        params_history.append(dict(env.params))
        forcing_times.append(env._forcing_time())
        rewards.append(reward)
        distances.append(info["boundary_distance"])
        stretching.append(info["local_stretching_rate"])
        done = terminated or truncated
    return (
        np.asarray(states),
        np.asarray(rewards),
        np.asarray(distances),
        np.asarray(stretching),
        params_history,
        np.asarray(forcing_times),
    )


def save_boundary_animation(
    episode,
    output: Path,
    resolution: int,
    stride: int,
    fps: int,
) -> None:
    """Animate the finite-time basin boundary and the agent tracking it."""
    states, _, _, _, params_history, times = episode
    frame_indices = list(range(0, len(states), stride))
    if frame_indices[-1] != len(states) - 1:
        frame_indices.append(len(states) - 1)
    xs = np.linspace(-1.6, 1.6, resolution)
    vs = np.linspace(-1.8, 1.8, resolution)
    cache: dict[int, np.ndarray] = {}
    fig, ax = plt.subplots(figsize=(7, 6))

    def draw(frame_number: int):
        index = frame_indices[frame_number]
        if index not in cache:
            labels = np.empty((resolution, resolution))
            params = params_history[index]
            for row, v in enumerate(vs):
                for col, x in enumerate(xs):
                    labels[row, col] = basin_label(
                        (x, v),
                        params,
                        horizon=6.0,
                        dt=0.08,
                        start_time=float(times[index]),
                    )
            cache[index] = labels
        params = params_history[index]
        ax.clear()
        ax.contourf(
            xs,
            vs,
            cache[index],
            levels=[-1.5, 0, 1.5],
            alpha=0.35,
            cmap="coolwarm",
        )
        ax.contour(xs, vs, cache[index], levels=[0], colors="black", linewidths=1.2)
        ax.plot(states[: index + 1, 0], states[: index + 1, 1], color="tab:green", lw=1.8)
        ax.scatter(states[index, 0], states[index, 1], color="tab:green", s=55, zorder=3)
        ax.set(
            xlim=(xs[0], xs[-1]),
            ylim=(vs[0], vs[-1]),
            xlabel="x",
            ylabel="v",
            title=(
                f"Boundary tracking, step {index}  "
                f"(alpha={params['alpha']:.3f}, beta={params['beta']:.3f})"
            ),
        )
        return ax.collections + ax.lines

    animation = FuncAnimation(fig, draw, frames=len(frame_indices), blit=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=fps) if output.suffix.lower() == ".mp4" else PillowWriter(fps=fps)
    animation.save(output, writer=writer, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=Path("models/duffing_agent"))
    parser.add_argument("--algorithm", choices=("sac", "ppo"), default="sac")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("evaluation.png"))
    parser.add_argument("--basin-resolution", type=int, default=90)
    parser.add_argument(
        "--evolve-dynamics", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--use-basin-mixing", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--animate-output",
        type=Path,
        help="Optional .gif or .mp4 showing the evolving basin boundary and agent path.",
    )
    parser.add_argument(
        "--evaluation-parameter-walk-interval",
        type=int,
        default=1,
        help="Change Duffing parameters every N control steps; use 0 to disable.",
    )
    parser.add_argument("--animation-stride", type=int, default=10)
    parser.add_argument("--animation-resolution", type=int, default=41)
    parser.add_argument("--animation-fps", type=int, default=8)
    args = parser.parse_args()
    if args.evaluation_parameter_walk_interval < 0:
        parser.error("--evaluation-parameter-walk-interval cannot be negative")
    if args.animation_stride < 1 or args.animation_resolution < 10 or args.animation_fps < 1:
        parser.error("animation stride/fps must be positive and resolution must be at least 10")

    env = DuffingBoundaryEnv(
        parameter_walk=args.evaluation_parameter_walk_interval > 0,
        parameter_walk_interval_steps=max(1, args.evaluation_parameter_walk_interval),
        evolve_dynamics=args.evolve_dynamics,
        use_basin_mixing=args.use_basin_mixing,
        boundary_update_interval=1,
    )
    model_file = args.model_path.with_suffix(".zip")
    if not model_file.exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_file}. Run `python src/train.py` first."
        )
    model = load_agent(args.model_path, env, args.algorithm)
    episodes = [
        run_episode(model, env, seed)
        for seed in range(args.episodes)
    ]
    xs, vs, basin = estimate_basin_map(env, args.basin_resolution)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].contourf(xs, vs, basin, levels=[-1.5, 0, 1.5], alpha=0.35, cmap="coolwarm")
    for index, (states, _, _, _, _, _) in enumerate(episodes):
        axes[0].plot(states[:, 0], states[:, 1], lw=1.5, label=f"episode {index + 1}")
    axes[0].set(title="Basin map and agent paths", xlabel="x", ylabel="v")
    axes[0].legend(fontsize=8)
    for _, _, distances, _, _, _ in episodes:
        axes[1].plot(distances, alpha=0.8)
    metric_title = (
        "Finite-time boundary score"
        if args.use_basin_mixing
        else "Global curved-boundary distance"
    )
    metric_ylabel = (
        "1 - local basin mixing (lower is better)"
        if args.use_basin_mixing
        else "nearest-curve distance (lower is better)"
    )
    axes[1].set(title=metric_title, xlabel="step", ylabel=metric_ylabel)
    for _, _, _, stretching, _, _ in episodes:
        axes[2].plot(stretching, alpha=0.8)
    axes[2].set(title="Local stretching rate", xlabel="step", ylabel="rate")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)

    if args.animate_output:
        save_boundary_animation(
            episodes[0],
            args.animate_output,
            args.animation_resolution,
            args.animation_stride,
            args.animation_fps,
        )
        print(f"Saved boundary animation to {args.animate_output}")

    returns = np.array([rewards.sum() for _, rewards, _, _, _, _ in episodes])
    proximity = np.array([distances.mean() for _, _, distances, _, _, _ in episodes])
    print(f"Mean return: {returns.mean():.3f} +/- {returns.std():.3f}")
    print(f"Mean {metric_title.lower()}: {proximity.mean():.4f} +/- {proximity.std():.4f}")
    print(f"Saved evaluation plot to {args.output}")
    env.close()


if __name__ == "__main__":
    main()
