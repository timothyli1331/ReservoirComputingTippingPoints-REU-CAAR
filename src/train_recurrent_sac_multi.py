"""Train matched LSTM-SAC or fixed-ESN-SAC on multiple double-well systems."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from statistics import mean

import numpy as np

from multi_environment import MultiDoubleWellEnv
from recurrent_sac import (
    ENCODER_TYPES,
    EpisodeSequenceReplayBuffer,
    RecurrentSACAgent,
    RecurrentSACConfig,
)
from systems import DEFAULT_SYSTEM_NAMES, parse_system_names


def parse_sizes(text: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("hidden sizes must be comma-separated integers") from exc
    if any(value < 1 for value in values):
        raise argparse.ArgumentTypeError("hidden sizes must be positive")
    return values


def make_env(args, selection: str) -> MultiDoubleWellEnv:
    return MultiDoubleWellEnv(
        system_names=args.systems,
        selection=selection,
        include_system_id=False,
        include_local_acceleration=args.include_local_acceleration,
        include_local_stretching=args.include_local_stretching,
        parameter_walk=True,
        parameter_walk_interval_steps=1,
        parameter_walk_scale=args.parameter_walk_scale,
        boundary_start_probability=args.boundary_start_probability,
        evolve_dynamics=True,
        dynamics_substeps=args.dynamics_substeps,
        use_basin_mixing=True,
        boundary_update_interval=args.boundary_update_interval,
        max_steps=args.max_steps,
    )


def evaluate(
    agent: RecurrentSACAgent,
    env: MultiDoubleWellEnv,
    episodes_per_system: int,
    seed: int,
) -> tuple[float, dict[str, float]]:
    scores: dict[str, list[float]] = {name: [] for name in env.system_names}
    for system_index, system_name in enumerate(env.system_names):
        for episode in range(episodes_per_system):
            episode_seed = seed + 10_000 * system_index + episode
            observation, _ = env.reset(
                seed=episode_seed, options={"system_name": system_name}
            )
            agent.reset()
            done = False
            total = 0.0
            while not done:
                action, _ = agent.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, _ = env.step(action)
                total += float(reward)
                done = bool(terminated or truncated)
            scores[system_name].append(total)
    means = {name: mean(values) for name, values in scores.items()}
    # Penalize system imbalance in the same spirit as the existing custom
    # reservoir-SAC validation selection.
    values = list(means.values())
    balanced_score = float(mean(values) - 0.5 * np.std(values))
    return balanced_score, means


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", choices=ENCODER_TYPES, required=True)
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--systems", default=",".join(DEFAULT_SYSTEM_NAMES))
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--include-local-acceleration", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-local-stretching", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--parameter-walk-scale", type=float, default=1.0)
    parser.add_argument("--boundary-start-probability", type=float, default=0.6)
    parser.add_argument("--dynamics-substeps", type=int, default=1)
    parser.add_argument("--boundary-update-interval", type=int, default=5)

    parser.add_argument("--recurrent-units", type=int, default=100)
    parser.add_argument("--actor-hidden-sizes", type=parse_sizes, default=parse_sizes("128"))
    parser.add_argument("--critic-hidden-sizes", type=parse_sizes, default=parse_sizes("128,128"))
    parser.add_argument("--spectral-radius", type=float, default=0.9)
    parser.add_argument("--leaking-rate", type=float, default=0.3)
    parser.add_argument("--input-scaling", type=float, default=0.5)

    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--learning-starts", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--burn-in", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--gradient-steps-per-transition", type=float, default=1.0)
    parser.add_argument("--actor-learning-rate", type=float, default=3e-4)
    parser.add_argument("--critic-learning-rate", type=float, default=3e-4)
    parser.add_argument("--temperature-learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--initial-temperature", type=float, default=0.2)

    parser.add_argument("--validation-interval", type=int, default=10_000)
    parser.add_argument("--validation-episodes-per-system", type=int, default=2)
    parser.add_argument("--log-interval", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-path", type=Path, required=True)
    args = parser.parse_args()
    args.systems = parse_system_names(args.systems)

    positive = {
        "timesteps": args.timesteps,
        "max-steps": args.max_steps,
        "recurrent-units": args.recurrent_units,
        "buffer-size": args.buffer_size,
        "batch-size": args.batch_size,
        "sequence-length": args.sequence_length,
        "validation-interval": args.validation_interval,
        "validation-episodes-per-system": args.validation_episodes_per_system,
        "log-interval": args.log_interval,
    }
    invalid = [name for name, value in positive.items() if value < 1]
    if invalid:
        parser.error("these arguments must be positive: " + ", ".join(invalid))
    if args.burn_in < 0 or args.learning_starts < 0:
        parser.error("burn-in and learning-starts cannot be negative")
    if args.burn_in + args.sequence_length > args.max_steps:
        parser.error("burn-in + sequence-length cannot exceed max-steps")
    if args.gradient_steps_per_transition <= 0:
        parser.error("gradient-steps-per-transition must be positive")
    if args.parameter_walk_scale < 0 or args.dynamics_substeps < 1:
        parser.error("walk scale must be nonnegative and substeps positive")
    if not 0.0 <= args.boundary_start_probability <= 1.0:
        parser.error("boundary-start-probability must be in [0, 1]")
    return args


def main() -> None:
    args = parse_args()
    training_env = make_env(args, "random")
    validation_env = make_env(args, "round_robin")
    config = RecurrentSACConfig(
        observation_size=int(training_env.observation_space.shape[0]),
        action_size=int(training_env.action_space.shape[0]),
        action_limit=float(training_env.action_space.high[0]),
        encoder_type=args.encoder,
        recurrent_units=args.recurrent_units,
        actor_hidden_sizes=args.actor_hidden_sizes,
        critic_hidden_sizes=args.critic_hidden_sizes,
        spectral_radius=args.spectral_radius,
        leaking_rate=args.leaking_rate,
        input_scaling=args.input_scaling,
        actor_learning_rate=args.actor_learning_rate,
        critic_learning_rate=args.critic_learning_rate,
        temperature_learning_rate=args.temperature_learning_rate,
        gamma=args.gamma,
        tau=args.tau,
        initial_temperature=args.initial_temperature,
        seed=args.seed,
    )
    agent = RecurrentSACAgent(config, device=args.device)
    replay = EpisodeSequenceReplayBuffer(args.buffer_size, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    started = time.monotonic()
    total_steps = 0
    total_updates = 0
    completed_episodes = 0
    next_log = args.log_interval
    next_validation = args.validation_interval
    recent_returns: list[float] = []
    best_validation = -np.inf
    latest_losses: dict[str, float] = {}
    history: list[dict[str, object]] = []
    best_path = Path(args.model_path)
    if best_path.suffix != ".pt":
        best_path = best_path.with_suffix(".pt")
    latest_path = best_path.with_name(f"{best_path.stem}_latest.pt")

    while total_steps < args.timesteps:
        observation, _ = training_env.reset()
        agent.reset()
        observations = [np.asarray(observation, dtype=np.float32)]
        actions: list[np.ndarray] = []
        rewards: list[float] = []
        terminals: list[bool] = []
        episode_return = 0.0
        done = False
        while not done and total_steps < args.timesteps:
            proposed_action, _ = agent.predict(observation, deterministic=False)
            if total_steps < args.learning_starts:
                action = rng.uniform(
                    training_env.action_space.low, training_env.action_space.high
                ).astype(np.float32)
            else:
                action = proposed_action
            next_observation, reward, terminated, truncated, _ = training_env.step(action)
            done = bool(terminated or truncated)
            observations.append(np.asarray(next_observation, dtype=np.float32))
            actions.append(np.asarray(action, dtype=np.float32))
            rewards.append(float(reward))
            terminals.append(bool(terminated))
            episode_return += float(reward)
            observation = next_observation
            total_steps += 1

        replay.add_episode(observations, actions, rewards, terminals)
        completed_episodes += 1
        recent_returns.append(episode_return)
        recent_returns = recent_returns[-20:]

        # Updating between episodes keeps the LSTM hidden state internally
        # consistent throughout each rollout.  Sequence replay still performs
        # off-policy updates with burn-in and BPTT.
        if total_steps >= args.learning_starts and replay.can_sample(
            args.burn_in, args.sequence_length
        ):
            desired_updates = len(actions) * args.gradient_steps_per_transition
            update_count = max(1, int(round(desired_updates)))
            for _ in range(update_count):
                batch = replay.sample(
                    args.batch_size, args.burn_in, args.sequence_length
                )
                latest_losses = agent.update(batch, args.burn_in)
                total_updates += 1

        while total_steps >= next_log:
            elapsed = max(time.monotonic() - started, 1e-9)
            losses = " ".join(
                f"{name} {value:.3f}" for name, value in latest_losses.items()
            )
            print(
                f"step {total_steps:7d} | episodes {completed_episodes:5d} | "
                f"updates {total_updates:7d} | return20 {mean(recent_returns):8.3f} | "
                f"{total_steps / elapsed:6.1f} steps/s"
                + (f" | {losses}" if losses else ""),
                flush=True,
            )
            history.append(
                {
                    "event": "training",
                    "step": total_steps,
                    "episodes": completed_episodes,
                    "updates": total_updates,
                    "elapsed_seconds": elapsed,
                    "return20": mean(recent_returns),
                    **latest_losses,
                }
            )
            next_log += args.log_interval

        while total_steps >= next_validation:
            score, system_scores = evaluate(
                agent,
                validation_env,
                args.validation_episodes_per_system,
                seed=1_000_000 + args.seed * 10_000 + next_validation,
            )
            print(
                f"validation step {total_steps}: balanced={score:.3f} "
                + " ".join(f"{name}={value:.3f}" for name, value in system_scores.items()),
                flush=True,
            )
            history.append(
                {
                    "event": "validation",
                    "step": total_steps,
                    "balanced_score": score,
                    "system_scores": system_scores,
                }
            )
            if score > best_validation:
                best_validation = score
                agent.save(best_path)
            next_validation += args.validation_interval

    agent.save(latest_path)
    if not best_path.is_file():
        agent.save(best_path)
    training_env.close()
    validation_env.close()
    training_seconds = time.monotonic() - started

    metadata = {
        "config": asdict(config),
        "timesteps": total_steps,
        "updates": total_updates,
        "episodes": completed_episodes,
        "best_validation_score": float(best_validation),
        "training_seconds": training_seconds,
        "environment_steps_per_second": total_steps / max(training_seconds, 1e-9),
        "sequence_replay": {
            "buffer_size": args.buffer_size,
            "batch_size": args.batch_size,
            "burn_in": args.burn_in,
            "sequence_length": args.sequence_length,
            "gradient_steps_per_transition": args.gradient_steps_per_transition,
            "updates_between_episodes": True,
        },
        "environment": {
            "systems": list(args.systems),
            "include_local_acceleration": args.include_local_acceleration,
            "include_local_stretching": args.include_local_stretching,
            "parameter_walk_scale": args.parameter_walk_scale,
            "boundary_start_probability": args.boundary_start_probability,
            "dynamics_substeps": args.dynamics_substeps,
        },
        "parameter_counts": agent.parameter_counts(),
        "learning_history": history,
        "best_checkpoint": str(best_path),
        "latest_checkpoint": str(latest_path),
    }
    best_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"saved best model to {best_path}", flush=True)
    print(f"saved latest model to {latest_path}", flush=True)


if __name__ == "__main__":
    main()
