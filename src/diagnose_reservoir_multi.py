"""Run a short, checkpointed learnability diagnostic for the multi-system reservoir."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from agentrc import ReservoirPolicy, build_agent, train_cem
from multi_environment import MultiDoubleWellEnv
from systems import DEFAULT_SYSTEM_NAMES, parse_system_names


def make_env(args) -> MultiDoubleWellEnv:
    """Build the normal environment with a shorter diagnostic horizon."""
    return MultiDoubleWellEnv(
        system_names=args.systems,
        selection="round_robin",
        include_system_id=args.include_system_id,
        parameter_walk=args.parameter_walk,
        parameter_walk_interval_steps=args.parameter_walk_interval,
        evolve_dynamics=args.evolve_dynamics,
        use_basin_mixing=args.use_basin_mixing,
        boundary_update_interval=args.boundary_update_interval,
        global_boundary_resolution=args.boundary_resolution,
        max_steps=args.max_steps,
    )


def rollout(policy, env, system_name: str, seed: int) -> dict[str, float]:
    observation, info = env.reset(
        seed=seed, options={"system_name": system_name}
    )
    policy.reset()
    total_return = 0.0
    normalized_actions: list[np.ndarray] = []
    boundary_proximities: list[float] = []
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action, _ = policy.predict(observation)
        normalized_actions.append(
            np.abs(np.asarray(action, dtype=float)) / env.action_limit
        )
        observation, reward, terminated, truncated, info = env.step(action)
        total_return += float(reward)
        if info["boundary_metric"] == "basin_mixing":
            proximity = float(info["basin_mixing"])
        else:
            proximity = float(np.exp(-float(info["boundary_distance"]) / 0.10))
        boundary_proximities.append(proximity)

    action_values = np.concatenate(normalized_actions)
    return {
        "return": total_return,
        "boundary_proximity": float(np.mean(boundary_proximities)),
        "action_rms": float(np.sqrt(np.mean(action_values**2))),
        "action_saturation": float(np.mean(action_values >= 0.90)),
        "steps": float(env.steps),
        "terminated": float(terminated),
    }


def evaluate_by_system(
    policy,
    env,
    system_names: tuple[str, ...],
    episodes: int,
    seed_offset: int,
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for system_index, system_name in enumerate(system_names):
        episodes_data = [
            rollout(
                policy,
                env,
                system_name,
                seed_offset + 1000 * system_index + episode,
            )
            for episode in range(episodes)
        ]
        results[system_name] = {
            key: float(np.mean([item[key] for item in episodes_data]))
            for key in episodes_data[0]
        }
        results[system_name]["return_std"] = float(
            np.std([item["return"] for item in episodes_data])
        )
    return results


def mean_return(
    policy,
    env,
    system_names: tuple[str, ...],
    seed_offset: int,
) -> float:
    values = [
        rollout(policy, env, name, seed_offset + index)["return"]
        for index, name in enumerate(system_names)
    ]
    return float(np.mean(values))


def rank_correlation(first: np.ndarray, second: np.ndarray) -> float:
    first_ranks = np.argsort(np.argsort(first)).astype(float)
    second_ranks = np.argsort(np.argsort(second)).astype(float)
    if np.std(first_ranks) == 0.0 or np.std(second_ranks) == 0.0:
        return float("nan")
    return float(np.corrcoef(first_ranks, second_ranks)[0, 1])


def candidate_stability(policy, env, args) -> dict[str, object]:
    """Check whether candidates that win on one seed set also win on another."""
    rng = np.random.default_rng(args.seed + 91_337)
    center = policy.readout_weights.copy()
    half = args.population // 2
    noise = rng.normal(size=(half, *center.shape))
    candidates = [center + args.initial_search_std * item for item in noise]
    candidates += [center - args.initial_search_std * item for item in noise]
    if args.population % 2:
        candidates.append(center.copy())

    training_scores = []
    fresh_scores = []
    for weights in candidates:
        policy.readout_weights = weights
        training_scores.append(
            mean_return(policy, env, args.systems, args.seed + 20_000)
        )
        fresh_scores.append(
            mean_return(policy, env, args.systems, args.seed + 40_000)
        )
    policy.readout_weights = center

    training = np.asarray(training_scores)
    fresh = np.asarray(fresh_scores)
    training_winner = int(np.argmax(training))
    fresh_winner = int(np.argmax(fresh))
    return {
        "rank_correlation": rank_correlation(training, fresh),
        "same_winner": training_winner == fresh_winner,
        "training_winner": training_winner,
        "fresh_winner": fresh_winner,
        "training_scores": training.tolist(),
        "fresh_scores": fresh.tolist(),
        "winner_fresh_regret": float(fresh[fresh_winner] - fresh[training_winner]),
    }


def print_system_table(
    baseline: dict[str, dict[str, float]],
    trained: dict[str, dict[str, float]],
) -> None:
    print("\nPer-system evaluation on identical seeds")
    print("system                 initial   trained    delta   boundary  saturation")
    for name in baseline:
        before = baseline[name]
        after = trained[name]
        print(
            f"{name:22s} {before['return']:8.3f} {after['return']:9.3f} "
            f"{after['return'] - before['return']:8.3f} "
            f"{after['boundary_proximity']:9.3f} "
            f"{after['action_saturation']:11.1%}"
        )


def conclusions(
    stability: dict[str, object],
    baseline: dict[str, dict[str, float]],
    trained: dict[str, dict[str, float]],
) -> list[str]:
    messages: list[str] = []
    correlation = float(stability["rank_correlation"])
    if not np.isfinite(correlation) or correlation < 0.30:
        messages.append(
            "Candidate ranking is unstable across seeds; CEM is mostly selecting rollout noise."
        )
    elif correlation < 0.65:
        messages.append(
            "Candidate ranking has only moderate seed-to-seed agreement; use more episodes per candidate."
        )
    else:
        messages.append("Candidate rankings generalize reasonably well to fresh seeds.")

    deltas = {
        name: trained[name]["return"] - baseline[name]["return"]
        for name in baseline
    }
    if float(np.mean(list(deltas.values()))) <= 0.0:
        messages.append("The short search did not improve mean held-out return.")
    weak = [name for name, delta in deltas.items() if delta <= 0.0]
    if weak:
        messages.append("No held-out improvement for: " + ", ".join(weak) + ".")

    saturated = [
        name for name, values in trained.items()
        if values["action_saturation"] >= 0.50
    ]
    if saturated:
        messages.append(
            "Readout actions are saturated for "
            + ", ".join(saturated)
            + "; the policy has little fine control there."
        )
    if not messages:
        messages.append("No obvious failure mode was detected in this short run.")
    return messages


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--systems", default=",".join(DEFAULT_SYSTEM_NAMES))
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--episodes-per-candidate", type=int, default=4)
    parser.add_argument("--validation-episodes", type=int, default=8)
    parser.add_argument("--evaluation-episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--reservoir-units", type=int, default=16)
    parser.add_argument(
        "--reservoir-backend",
        choices=("reservoirpy", "duffing"),
        default="reservoirpy",
    )
    parser.add_argument("--include-system-id", action="store_true")
    parser.add_argument(
        "--evolve-dynamics", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--parameter-walk", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--parameter-walk-interval", type=int, default=1)
    parser.add_argument(
        "--use-basin-mixing", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--boundary-update-interval", type=int, default=10)
    parser.add_argument("--boundary-resolution", type=int, default=25)
    parser.add_argument("--initial-search-std", type=float, default=0.04)
    parser.add_argument("--minimum-search-std", type=float, default=0.002)
    parser.add_argument("--maximum-search-std", type=float, default=0.06)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path("models/reservoir_diagnostic_best.pkl"),
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("results/reservoir_diagnostic.json"),
    )
    args = parser.parse_args()
    args.systems = parse_system_names(args.systems)
    if args.population < 2 or args.generations < 1:
        parser.error("population must be at least 2 and generations at least 1")
    if args.episodes_per_candidate < len(args.systems):
        parser.error("episodes-per-candidate must cover every selected system")
    if args.validation_episodes < len(args.systems):
        parser.error("validation-episodes must cover every selected system")
    return args


def main() -> None:
    args = parse_args()
    env = make_env(args)
    policy = build_agent(
        env,
        units=args.reservoir_units,
        seed=args.seed,
        backend=args.reservoir_backend,
    )
    args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)

    print("Evaluating the untrained readout...")
    baseline = evaluate_by_system(
        policy, env, args.systems, args.evaluation_episodes, args.seed + 60_000
    )
    print("Measuring whether candidate rankings survive fresh seeds...")
    stability = candidate_stability(policy, env, args)
    policy.save(args.checkpoint_path)

    best_seen = -np.inf

    def report_generation(record: dict[str, float]) -> None:
        nonlocal best_seen
        validation = record.get("validation_return")
        validation_text = "" if validation is None else f" | val {validation:8.3f}"
        print(
            f"generation {int(record['generation']):3d} | "
            f"best {record['best_return']:8.3f} | "
            f"mean {record['mean_return']:8.3f} | "
            f"std {record['search_std']:.4f}{validation_text}",
            flush=True,
        )
        if validation is not None and validation > best_seen:
            best_seen = validation
            policy.save(args.checkpoint_path)
            print(f"  checkpointed new validation best to {args.checkpoint_path}")

    interrupted = False
    history: list[dict[str, float]] = []
    try:
        history = train_cem(
            policy,
            env,
            generations=args.generations,
            population_size=args.population,
            elite_fraction=0.25,
            episodes_per_candidate=args.episodes_per_candidate,
            initial_std=args.initial_search_std,
            min_std=args.minimum_search_std,
            max_std=args.maximum_search_std,
            std_decay=0.98,
            smoothing=0.70,
            mirrored_sampling=True,
            validation_episodes=args.validation_episodes,
            validation_interval=1,
            early_stopping_patience=args.early_stopping_patience,
            seed=args.seed,
            callback=report_generation,
        )
        policy.save(args.checkpoint_path)
    except KeyboardInterrupt:
        interrupted = True
        print("\nStopped safely; using the most recent validation checkpoint.")
        policy = ReservoirPolicy.load(args.checkpoint_path)

    print("Evaluating the saved policy on held-out episodes...")
    trained = evaluate_by_system(
        policy, env, args.systems, args.evaluation_episodes, args.seed + 60_000
    )
    findings = conclusions(stability, baseline, trained)
    print_system_table(baseline, trained)
    print(f"\nCandidate rank correlation: {stability['rank_correlation']:.3f}")
    print("Findings")
    for finding in findings:
        print(f"- {finding}")

    report = {
        "interrupted": interrupted,
        "configuration": {
            "systems": list(args.systems),
            "generations": args.generations,
            "population": args.population,
            "episodes_per_candidate": args.episodes_per_candidate,
            "validation_episodes": args.validation_episodes,
            "evaluation_episodes": args.evaluation_episodes,
            "max_steps": args.max_steps,
            "reservoir_units": args.reservoir_units,
            "reservoir_backend": args.reservoir_backend,
            "initial_search_std": args.initial_search_std,
            "minimum_search_std": args.minimum_search_std,
            "maximum_search_std": args.maximum_search_std,
            "seed": args.seed,
        },
        "candidate_stability": stability,
        "baseline": baseline,
        "trained": trained,
        "history": history,
        "findings": findings,
        "checkpoint_path": str(args.checkpoint_path),
    }
    args.report_path.write_text(json.dumps(report, indent=2) + "\n")
    env.close()
    print(f"Saved diagnostic report to {args.report_path}")


if __name__ == "__main__":
    main()
