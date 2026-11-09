"""Train a fixed-reservoir actor with Soft Actor-Critic on multiple systems."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np

from agentrc import build_agent
from multi_environment import MultiDoubleWellEnv
from reservoir_sac import MEMORY_MODES, ReplayBuffer, ReservoirSACAgent
from systems import DEFAULT_SYSTEM_NAMES, parse_system_names


def make_env(args, selection: str) -> MultiDoubleWellEnv:
    return MultiDoubleWellEnv(
        system_names=args.systems,
        selection=selection,
        include_system_id=args.include_system_id,
        include_local_acceleration=args.include_local_acceleration,
        include_local_stretching=args.include_local_stretching,
        parameter_walk=args.parameter_walk,
        parameter_walk_interval_steps=args.parameter_walk_interval,
        parameter_walk_scale=args.parameter_walk_scale,
        boundary_start_probability=args.boundary_start_probability,
        evolve_dynamics=args.evolve_dynamics,
        dynamics_substeps=args.dynamics_substeps,
        use_basin_mixing=args.use_basin_mixing,
        boundary_update_interval=args.boundary_update_interval,
        global_boundary_resolution=args.boundary_resolution,
        max_steps=args.max_steps,
    )


def reservoir_kwargs(args) -> dict[str, float | int]:
    if args.reservoir_backend == "duffing":
        return {
            "dt": args.duffing_dt,
            "substeps": args.duffing_substeps,
            "alpha": args.duffing_alpha,
            "beta": args.duffing_beta,
            "damping": args.duffing_damping,
            "parameter_spread": args.duffing_parameter_spread,
            "input_scaling": args.duffing_input_scaling,
            "state_clip": args.duffing_state_clip,
        }
    return {
        "spectral_radius": args.spectral_radius,
        "leaking_rate": args.leaking_rate,
        "input_scaling": args.input_scaling,
    }


def evaluate_agent(
    agent: ReservoirSACAgent,
    env: MultiDoubleWellEnv,
    system_names: tuple[str, ...],
    episodes_per_system: int,
    seed: int,
    balance_penalty: float,
) -> dict[str, object]:
    """Evaluate without disturbing the encoder used by the training episode."""
    encoder = copy.deepcopy(agent.encoder)
    per_system: dict[str, dict[str, float]] = {}
    all_returns = []

    for system_index, system_name in enumerate(system_names):
        returns = []
        proximities = []
        saturations = []
        for episode in range(episodes_per_system):
            episode_seed = seed + 1000 * system_index + episode
            observation, _ = env.reset(
                seed=episode_seed, options={"system_name": system_name}
            )
            encoder.reset()
            features = agent.encode_with(encoder, observation)
            total_return = 0.0
            episode_proximities = []
            normalized_actions = []
            done = False
            while not done:
                action = agent.act_features(features, deterministic=True)
                normalized_actions.append(np.abs(action) / env.action_limit)
                observation, reward, terminated, truncated, info = env.step(action)
                features = agent.encode_with(encoder, observation)
                total_return += float(reward)
                if info["boundary_metric"] == "basin_mixing":
                    proximity = float(info["basin_mixing"])
                else:
                    proximity = float(
                        np.exp(-float(info["boundary_distance"]) / 0.10)
                    )
                episode_proximities.append(proximity)
                done = terminated or truncated
            returns.append(total_return)
            proximities.append(float(np.mean(episode_proximities)))
            saturations.append(
                float(np.mean(np.concatenate(normalized_actions) >= 0.90))
            )
        all_returns.extend(returns)
        per_system[system_name] = {
            "return": float(np.mean(returns)),
            "return_std": float(np.std(returns)),
            "boundary_proximity": float(np.mean(proximities)),
            "action_saturation": float(np.mean(saturations)),
        }

    system_means = np.asarray(
        [per_system[name]["return"] for name in system_names], dtype=float
    )
    robust_score = float(
        system_means.mean() - balance_penalty * system_means.std()
    )
    return {
        "mean_return": float(np.mean(all_returns)),
        "return_std": float(np.std(all_returns)),
        "worst_system_return": float(system_means.min()),
        "system_return_std": float(system_means.std()),
        "robust_score": robust_score,
        "per_system": per_system,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--systems", default=",".join(DEFAULT_SYSTEM_NAMES))
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument(
        "--boundary-start-probability",
        type=float,
        default=0.6,
        help="Probability of starting near the local boundary; lower is harder.",
    )
    parser.add_argument("--include-system-id", action="store_true")
    parser.add_argument(
        "--include-local-acceleration",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include normalized local acceleration in observations.",
    )
    parser.add_argument(
        "--include-local-stretching",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include local stretching rate in observations.",
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
    parser.add_argument(
        "--parameter-walk", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--parameter-walk-interval", type=int, default=1)
    parser.add_argument(
        "--parameter-walk-scale",
        type=float,
        default=1.0,
        help="Multiplier for every parameter random-walk standard deviation.",
    )
    parser.add_argument("--boundary-update-interval", type=int, default=5)
    parser.add_argument("--boundary-resolution", type=int, default=41)

    parser.add_argument(
        "--reservoir-backend",
        choices=("reservoirpy", "duffing"),
        default="reservoirpy",
    )
    parser.add_argument("--reservoir-units", type=int, default=100)
    parser.add_argument(
        "--memory-mode",
        choices=MEMORY_MODES,
        default="persistent",
        help=(
            "Carry reservoir state across an episode, or reset it before every "
            "observation for a controlled no-memory ablation."
        ),
    )
    parser.add_argument("--spectral-radius", type=float, default=0.9)
    parser.add_argument("--leaking-rate", type=float, default=0.3)
    parser.add_argument("--input-scaling", type=float, default=0.5)
    parser.add_argument("--duffing-dt", type=float, default=0.04)
    parser.add_argument("--duffing-substeps", type=int, default=2)
    parser.add_argument("--duffing-alpha", type=float, default=-1.0)
    parser.add_argument("--duffing-beta", type=float, default=1.0)
    parser.add_argument("--duffing-damping", type=float, default=0.25)
    parser.add_argument("--duffing-parameter-spread", type=float, default=0.18)
    parser.add_argument("--duffing-input-scaling", type=float, default=0.8)
    parser.add_argument("--duffing-state-clip", type=float, default=4.0)

    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--learning-starts", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--actor-learning-rate", type=float, default=3e-4)
    parser.add_argument("--critic-learning-rate", type=float, default=3e-4)
    parser.add_argument("--temperature-learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--initial-temperature", type=float, default=0.2)
    parser.add_argument("--critic-hidden-sizes", default="128,128")

    parser.add_argument("--validation-interval", type=int, default=10_000)
    parser.add_argument("--validation-episodes-per-system", type=int, default=2)
    parser.add_argument("--balance-penalty", type=float, default=0.5)
    parser.add_argument("--log-interval", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/multi_reservoir_sac.pt"),
    )
    args = parser.parse_args()
    args.systems = parse_system_names(args.systems)
    try:
        args.critic_hidden_sizes = tuple(
            int(value.strip())
            for value in args.critic_hidden_sizes.split(",")
            if value.strip()
        )
    except ValueError:
        parser.error("critic-hidden-sizes must be comma-separated integers")
    if not args.critic_hidden_sizes or min(args.critic_hidden_sizes) < 1:
        parser.error("critic-hidden-sizes must contain positive integers")
    positive_values = {
        "timesteps": args.timesteps,
        "max-steps": args.max_steps,
        "reservoir-units": args.reservoir_units,
        "buffer-size": args.buffer_size,
        "batch-size": args.batch_size,
        "validation-interval": args.validation_interval,
        "validation-episodes-per-system": args.validation_episodes_per_system,
        "log-interval": args.log_interval,
    }
    invalid = [name for name, value in positive_values.items() if value < 1]
    if invalid:
        parser.error("these values must be positive: " + ", ".join(invalid))
    if args.gradient_steps < 1 or args.learning_starts < 0:
        parser.error("gradient-steps must be positive and learning-starts nonnegative")
    if args.batch_size > args.buffer_size:
        parser.error("batch-size cannot exceed buffer-size")
    if args.parameter_walk_scale < 0:
        parser.error("parameter-walk-scale must be nonnegative")
    if args.dynamics_substeps < 1:
        parser.error("dynamics-substeps must be at least 1")
    if not 0.0 <= args.boundary_start_probability <= 1.0:
        parser.error("boundary-start-probability must be in [0, 1]")
    return args


def main() -> None:
    args = parse_args()
    training_env = make_env(args, selection="random")
    validation_env = make_env(args, selection="round_robin")
    encoder = build_agent(
        training_env,
        units=args.reservoir_units,
        seed=args.seed,
        backend=args.reservoir_backend,
        **reservoir_kwargs(args),
    )
    agent = ReservoirSACAgent(
        encoder=encoder,
        action_size=int(training_env.action_space.shape[0]),
        action_limit=float(training_env.action_space.high[0]),
        hidden_sizes=args.critic_hidden_sizes,
        actor_learning_rate=args.actor_learning_rate,
        critic_learning_rate=args.critic_learning_rate,
        temperature_learning_rate=args.temperature_learning_rate,
        gamma=args.gamma,
        tau=args.tau,
        initial_temperature=args.initial_temperature,
        memory_mode=args.memory_mode,
        seed=args.seed,
        device=args.device,
    )
    replay = ReplayBuffer(
        args.buffer_size,
        agent.feature_size,
        agent.action_size,
        seed=args.seed,
    )
    rng = np.random.default_rng(args.seed)
    history: list[dict[str, object]] = []
    best_score = -np.inf
    best_path = agent.save(args.model_path)
    latest_path = best_path.with_name(f"{best_path.stem}_latest.pt")

    observation, _ = training_env.reset(seed=args.seed)
    agent.reset()
    features = agent.encode(observation)
    episode_return = 0.0
    completed_episodes = 0
    recent_returns: list[float] = []
    last_losses: dict[str, float] = {}
    started = time.monotonic()
    interrupted = False
    step = 0

    try:
        for step in range(1, args.timesteps + 1):
            if step <= args.learning_starts:
                action = rng.uniform(
                    training_env.action_space.low,
                    training_env.action_space.high,
                ).astype(np.float32)
            else:
                action = agent.act_features(features, deterministic=False)

            next_observation, reward, terminated, truncated, _ = training_env.step(
                action
            )
            next_features = agent.encode(next_observation)
            replay.add(features, action, reward, next_features, terminated)
            features = next_features
            episode_return += float(reward)

            if step > args.learning_starts and len(replay) >= args.batch_size:
                for _ in range(args.gradient_steps):
                    last_losses = agent.update(replay.sample(args.batch_size))

            if terminated or truncated:
                completed_episodes += 1
                recent_returns.append(episode_return)
                recent_returns = recent_returns[-20:]
                observation, _ = training_env.reset()
                agent.reset()
                features = agent.encode(observation)
                episode_return = 0.0

            if step % args.log_interval == 0:
                elapsed = max(time.monotonic() - started, 1e-9)
                mean_recent = (
                    float(np.mean(recent_returns)) if recent_returns else float("nan")
                )
                loss_text = (
                    f" | actor {last_losses['actor_loss']:.3f} "
                    f"critic {last_losses['critic_loss']:.3f} "
                    f"alpha {last_losses['temperature']:.3f}"
                    if last_losses
                    else ""
                )
                print(
                    f"step {step:7d} | episodes {completed_episodes:5d} | "
                    f"return20 {mean_recent:8.3f} | {step / elapsed:6.1f} steps/s"
                    f"{loss_text}",
                    flush=True,
                )

            should_validate = (
                step % args.validation_interval == 0 or step == args.timesteps
            )
            if should_validate:
                metrics = evaluate_agent(
                    agent,
                    validation_env,
                    args.systems,
                    args.validation_episodes_per_system,
                    seed=args.seed + 1_000_000,
                    balance_penalty=args.balance_penalty,
                )
                record: dict[str, object] = {"step": step, **metrics}
                history.append(record)
                print(
                    f"validation | mean {metrics['mean_return']:8.3f} | "
                    f"worst {metrics['worst_system_return']:8.3f} | "
                    f"balanced {metrics['robust_score']:8.3f}",
                    flush=True,
                )
                if float(metrics["robust_score"]) > best_score:
                    best_score = float(metrics["robust_score"])
                    best_path = agent.save(args.model_path)
                    print(f"  checkpointed new balanced best to {best_path}")
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted safely; saving the current agent.")
    finally:
        latest_path = agent.save(latest_path)
        training_env.close()
        validation_env.close()

    metadata_path = best_path.with_suffix(".json")
    metadata = {
        "interrupted": interrupted,
        "completed_timesteps": step,
        "completed_episodes": completed_episodes,
        "best_balanced_score": best_score if np.isfinite(best_score) else None,
        "best_model": str(best_path),
        "latest_model": str(latest_path),
        "history": history,
        "configuration": {
            "systems": list(args.systems),
            "max_steps": args.max_steps,
            "boundary_start_probability": args.boundary_start_probability,
            "include_system_id": args.include_system_id,
            "include_local_acceleration": args.include_local_acceleration,
            "include_local_stretching": args.include_local_stretching,
            "evolve_dynamics": args.evolve_dynamics,
            "dynamics_substeps": args.dynamics_substeps,
            "use_basin_mixing": args.use_basin_mixing,
            "parameter_walk": args.parameter_walk,
            "parameter_walk_interval": args.parameter_walk_interval,
            "parameter_walk_scale": args.parameter_walk_scale,
            "boundary_update_interval": args.boundary_update_interval,
            "boundary_resolution": args.boundary_resolution,
            "reservoir_backend": args.reservoir_backend,
            "reservoir_units": args.reservoir_units,
            "memory_mode": args.memory_mode,
            "spectral_radius": args.spectral_radius,
            "leaking_rate": args.leaking_rate,
            "input_scaling": args.input_scaling,
            "buffer_size": args.buffer_size,
            "learning_starts": args.learning_starts,
            "batch_size": args.batch_size,
            "gradient_steps": args.gradient_steps,
            "actor_learning_rate": args.actor_learning_rate,
            "critic_learning_rate": args.critic_learning_rate,
            "temperature_learning_rate": args.temperature_learning_rate,
            "initial_temperature": args.initial_temperature,
            "critic_hidden_sizes": list(args.critic_hidden_sizes),
            "gamma": args.gamma,
            "tau": args.tau,
            "balance_penalty": args.balance_penalty,
            "seed": args.seed,
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved best checkpoint to {best_path}")
    print(f"Saved latest checkpoint to {latest_path}")
    print(f"Saved training metadata to {metadata_path}")


if __name__ == "__main__":
    main()
