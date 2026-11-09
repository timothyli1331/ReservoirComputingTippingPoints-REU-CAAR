"""Train a stateful reservoir policy with continuous-action policy search."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentrc import build_agent, train_cem
from environment import DuffingBoundaryEnv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=int, default=80)
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--episodes-per-candidate", type=int, default=2)
    parser.add_argument("--reservoir-units", type=int, default=100)
    parser.add_argument(
        "--reservoir-backend",
        choices=("reservoirpy", "duffing"),
        default="reservoirpy",
        help="Use a conventional ESN or a stateful bank of Duffing oscillators.",
    )
    parser.add_argument("--duffing-dt", type=float, default=0.04)
    parser.add_argument("--duffing-substeps", type=int, default=2)
    parser.add_argument("--duffing-alpha", type=float, default=-1.0)
    parser.add_argument("--duffing-beta", type=float, default=1.0)
    parser.add_argument("--duffing-damping", type=float, default=0.25)
    parser.add_argument("--duffing-parameter-spread", type=float, default=0.18)
    parser.add_argument("--duffing-input-scaling", type=float, default=0.8)
    parser.add_argument("--duffing-state-clip", type=float, default=4.0)
    parser.add_argument("--elite-fraction", type=float, default=0.2)
    parser.add_argument("--initial-search-std", type=float, default=0.15)
    parser.add_argument("--minimum-search-std", type=float, default=0.001)
    parser.add_argument("--search-std-decay", type=float, default=0.95)
    parser.add_argument("--cem-smoothing", type=float, default=0.2)
    parser.add_argument(
        "--mirrored-sampling", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--validation-episodes", type=int, default=8)
    parser.add_argument("--validation-interval", type=int, default=1)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=20,
        help="Stop after this many validation checks without improvement; zero disables.",
    )
    parser.add_argument(
        "--evolve-dynamics", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--use-basin-mixing", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--parameter-walk", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--parameter-walk-interval", type=int, default=1)
    parser.add_argument("--boundary-update-interval", type=int, default=1)
    parser.add_argument("--boundary-resolution", type=int, default=41)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--model-path", type=Path, default=Path("models/duffing_reservoir.pkl")
    )
    args = parser.parse_args()
    if args.parameter_walk_interval < 1 or args.boundary_update_interval < 1:
        parser.error("walk and boundary update intervals must be at least 1")
    if args.validation_episodes < 1:
        parser.error("validation-episodes must be at least 1 for robust model selection")

    env = DuffingBoundaryEnv(
        parameter_walk=args.parameter_walk,
        parameter_walk_interval_steps=args.parameter_walk_interval,
        evolve_dynamics=args.evolve_dynamics,
        use_basin_mixing=args.use_basin_mixing,
        boundary_update_interval=args.boundary_update_interval,
        global_boundary_resolution=args.boundary_resolution,
    )
    reservoir_kwargs = {}
    if args.reservoir_backend == "duffing":
        reservoir_kwargs = {
            "dt": args.duffing_dt,
            "substeps": args.duffing_substeps,
            "alpha": args.duffing_alpha,
            "beta": args.duffing_beta,
            "damping": args.duffing_damping,
            "parameter_spread": args.duffing_parameter_spread,
            "input_scaling": args.duffing_input_scaling,
            "state_clip": args.duffing_state_clip,
        }
    policy = build_agent(
        env,
        units=args.reservoir_units,
        seed=args.seed,
        backend=args.reservoir_backend,
        **reservoir_kwargs,
    )

    def report(record: dict[str, float]) -> None:
        validation = (
            f" | val {record['validation_return']:8.3f}"
            if "validation_return" in record
            else ""
        )
        print(
            f"generation {int(record['generation']):3d} | "
            f"best {record['best_return']:8.3f} | "
            f"mean {record['mean_return']:8.3f} | "
            f"std {record['search_std']:.4f}"
            f"{validation}"
        )

    history = train_cem(
        policy,
        env,
        generations=args.generations,
        population_size=args.population,
        elite_fraction=args.elite_fraction,
        episodes_per_candidate=args.episodes_per_candidate,
        initial_std=args.initial_search_std,
        min_std=args.minimum_search_std,
        std_decay=args.search_std_decay,
        smoothing=args.cem_smoothing,
        mirrored_sampling=args.mirrored_sampling,
        validation_episodes=args.validation_episodes,
        validation_interval=args.validation_interval,
        early_stopping_patience=args.early_stopping_patience,
        seed=args.seed,
        callback=report,
    )
    saved_path = policy.save(args.model_path)
    history_path = saved_path.with_name(f"{saved_path.stem}_history.json")
    metadata_path = saved_path.with_name(f"{saved_path.stem}_metadata.json")
    history_path.write_text(json.dumps(history, indent=2))
    metadata_path.write_text(
        json.dumps(
            {
                "generations": args.generations,
                "population": args.population,
                "episodes_per_candidate": args.episodes_per_candidate,
                "reservoir_units": args.reservoir_units,
                "reservoir_backend": args.reservoir_backend,
                "duffing_reservoir": (
                    {
                        "dt": args.duffing_dt,
                        "substeps": args.duffing_substeps,
                        "alpha": args.duffing_alpha,
                        "beta": args.duffing_beta,
                        "damping": args.duffing_damping,
                        "parameter_spread": args.duffing_parameter_spread,
                        "input_scaling": args.duffing_input_scaling,
                        "state_clip": args.duffing_state_clip,
                    }
                    if args.reservoir_backend == "duffing"
                    else None
                ),
                "elite_fraction": args.elite_fraction,
                "initial_search_std": args.initial_search_std,
                "minimum_search_std": args.minimum_search_std,
                "search_std_decay": args.search_std_decay,
                "cem_smoothing": args.cem_smoothing,
                "mirrored_sampling": args.mirrored_sampling,
                "validation_episodes": args.validation_episodes,
                "validation_interval": args.validation_interval,
                "early_stopping_patience": args.early_stopping_patience,
                "completed_generations": len(history),
                "best_validation_return": history[-1].get(
                    "best_validation_return"
                ),
                "parameter_walk": args.parameter_walk,
                "parameter_walk_interval_steps": args.parameter_walk_interval,
                "evolve_dynamics": args.evolve_dynamics,
                "use_basin_mixing": args.use_basin_mixing,
                "boundary_update_interval": args.boundary_update_interval,
                "boundary_resolution": args.boundary_resolution,
                "observation": [
                    "x",
                    "velocity",
                    "sin_forcing_phase",
                    "cos_forcing_phase",
                    "local_acceleration",
                    "local_stretching_rate",
                ],
                "seed": args.seed,
            },
            indent=2,
        )
    )
    env.close()
    print(f"Saved policy to {saved_path}")
    print(f"Saved training history to {history_path}")
    print(f"Saved experiment metadata to {metadata_path}")


if __name__ == "__main__":
    main()
