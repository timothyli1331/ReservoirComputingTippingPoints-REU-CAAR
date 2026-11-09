"""Compare SAC, PPO, and ReservoirPy on one shared Duffing parameter walk."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from agent import load_agent
from agentrc import ReservoirPolicy
from environment import DuffingBoundaryEnv


def run_episode(model, params: dict, seed: int, recurrent: bool):
    env = DuffingBoundaryEnv(
        params=params,
        parameter_walk=True,
        parameter_walk_interval_steps=1,
        evolve_dynamics=True,
        use_basin_mixing=True,
        boundary_update_interval=1,
    )
    observation, _ = env.reset(seed=seed)
    if recurrent:
        model.reset()
    states = [env.state.copy()]
    rewards = []
    distances = []
    done = False
    while not done:
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        states.append(env.state.copy())
        rewards.append(float(reward))
        distances.append(float(info["boundary_distance"]))
        done = terminated or truncated
    env.close()
    return {
        "states": np.asarray(states),
        "return": float(np.sum(rewards)),
        "mean_distance": float(np.mean(distances)),
        "steps": len(rewards),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sac-model", type=Path, required=True)
    parser.add_argument("--ppo-model", type=Path, required=True)
    parser.add_argument("--reservoir-model", type=Path, required=True)
    parser.add_argument(
        "--schedule",
        type=Path,
        default=Path("test_data/duffing_parameter_walk.json"),
    )
    parser.add_argument("--seed", type=int, default=50_000)
    parser.add_argument("--output", type=Path, default=Path("three_model_walk.png"))
    parser.add_argument("--csv", type=Path, default=Path("three_model_walk.csv"))
    args = parser.parse_args()

    schedule = json.loads(args.schedule.read_text())
    episode_params = schedule["episodes"]
    if not episode_params:
        raise ValueError("the parameter schedule contains no episodes")

    probe_env = DuffingBoundaryEnv(
        parameter_walk=True,
        parameter_walk_interval_steps=1,
        evolve_dynamics=True,
        use_basin_mixing=True,
        boundary_update_interval=1,
    )
    models = {
        "SAC": (load_agent(args.sac_model, probe_env, "sac"), False),
        "PPO": (load_agent(args.ppo_model, probe_env, "ppo"), False),
        "Reservoir": (ReservoirPolicy.load(args.reservoir_model), True),
    }
    colors = {"SAC": "tab:blue", "PPO": "tab:green", "Reservoir": "tab:orange"}
    results = []
    first_runs = {}
    for episode, scheduled in enumerate(episode_params):
        params = {
            key: float(scheduled[key])
            for key in ("alpha", "beta", "damping", "gamma", "omega", "max_action")
        }
        for model_name, (model, recurrent) in models.items():
            run = run_episode(model, params, args.seed + episode, recurrent)
            results.append(
                {
                    "episode": episode,
                    "seed": args.seed + episode,
                    "model": model_name,
                    "alpha": params["alpha"],
                    "beta": params["beta"],
                    "damping": params["damping"],
                    "return": run["return"],
                    "mean_boundary_distance": run["mean_distance"],
                    "steps": run["steps"],
                }
            )
            if episode == 0:
                first_runs[model_name] = run

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    x_axis = np.arange(len(episode_params))
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes[0, 0].plot(x_axis, [row["alpha"] for row in episode_params], label="alpha")
    axes[0, 0].plot(x_axis, [row["beta"] for row in episode_params], label="beta")
    axes[0, 0].plot(
        x_axis, [row["damping"] for row in episode_params], label="damping"
    )
    axes[0, 0].set(title="Shared parameter walk", xlabel="episode", ylabel="value")
    axes[0, 0].legend()

    summaries = {}
    for model_name in models:
        rows = [row for row in results if row["model"] == model_name]
        returns = np.asarray([row["return"] for row in rows])
        distances = np.asarray([row["mean_boundary_distance"] for row in rows])
        summaries[model_name] = (returns, distances)
        axes[0, 1].plot(x_axis, returns, label=model_name, color=colors[model_name])
        axes[0, 2].plot(x_axis, distances, label=model_name, color=colors[model_name])
    axes[0, 1].set(title="Return by test episode", xlabel="episode", ylabel="return")
    axes[0, 2].set(
        title="Boundary distance by episode",
        xlabel="episode",
        ylabel="distance (lower is better)",
    )
    axes[0, 1].legend()
    axes[0, 2].legend()

    model_names = list(models)
    positions = np.arange(len(model_names))
    mean_returns = [summaries[name][0].mean() for name in model_names]
    std_returns = [summaries[name][0].std() for name in model_names]
    mean_distances = [summaries[name][1].mean() for name in model_names]
    std_distances = [summaries[name][1].std() for name in model_names]
    axes[1, 0].bar(
        positions,
        mean_returns,
        yerr=std_returns,
        color=[colors[name] for name in model_names],
    )
    axes[1, 0].set_xticks(positions, model_names)
    axes[1, 0].set(title="Mean return", ylabel="return (higher is better)")
    axes[1, 1].bar(
        positions,
        mean_distances,
        yerr=std_distances,
        color=[colors[name] for name in model_names],
    )
    axes[1, 1].set_xticks(positions, model_names)
    axes[1, 1].set(
        title="Mean boundary distance", ylabel="distance (lower is better)"
    )

    for model_name in model_names:
        states = first_runs[model_name]["states"]
        axes[1, 2].plot(
            states[:, 0], states[:, 1], label=model_name, color=colors[model_name]
        )
    first = episode_params[0]
    stable_lambda = (
        -first["damping"]
        - np.sqrt(first["damping"] ** 2 - 4.0 * first["alpha"])
    ) / 2.0
    boundary_x = np.linspace(-1.5, 1.5, 200)
    axes[1, 2].plot(
        boundary_x,
        stable_lambda * boundary_x,
        "k--",
        alpha=0.6,
        label="initial local tangent",
    )
    axes[1, 2].set(title="First paired trajectory", xlabel="x", ylabel="v")
    axes[1, 2].legend()

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    probe_env.close()

    print(f"Saved comparison plot to {args.output}")
    print(f"Saved episode-level results to {args.csv}")
    for model_name in model_names:
        returns, distances = summaries[model_name]
        print(
            f"{model_name:10s} return={returns.mean():8.3f} +/- {returns.std():7.3f}, "
            f"distance={distances.mean():.4f} +/- {distances.std():.4f}"
        )


if __name__ == "__main__":
    main()
