"""Train one SAC or PPO policy across multiple double-well systems."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent import build_agent
from multi_environment import MultiDoubleWellEnv
from systems import DEFAULT_SYSTEM_NAMES, parse_system_names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("sac", "ppo"), default="sac")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--systems", default=",".join(DEFAULT_SYSTEM_NAMES))
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
    parser.add_argument(
        "--boundary-start-probability",
        type=float,
        default=0.6,
        help="Probability of starting near the local boundary; lower is harder.",
    )
    parser.add_argument("--parameter-walk-interval", type=int, default=1)
    parser.add_argument(
        "--parameter-walk-scale",
        type=float,
        default=1.0,
        help="Multiplier for every parameter random-walk standard deviation.",
    )
    parser.add_argument("--boundary-update-interval", type=int, default=1)
    parser.add_argument("--boundary-resolution", type=int, default=41)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-path", type=Path, default=Path("models/multi_sac"))
    args = parser.parse_args()
    if (
        args.parameter_walk_interval < 1
        or args.boundary_update_interval < 1
        or args.dynamics_substeps < 1
        or args.parameter_walk_scale < 0
        or not 0.0 <= args.boundary_start_probability <= 1.0
    ):
        parser.error(
            "start probability must be in [0, 1], walk scale nonnegative, "
            "and intervals/substeps at least 1"
        )

    system_names = parse_system_names(args.systems)
    env = MultiDoubleWellEnv(
        system_names=system_names,
        selection="random",
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
    )
    model = build_agent(env, algorithm=args.algorithm, seed=args.seed)
    model.learn(total_timesteps=args.timesteps)
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.model_path))
    metadata_path = args.model_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "algorithm": args.algorithm,
                "timesteps": args.timesteps,
                "systems": list(system_names),
                "include_system_id": args.include_system_id,
                "include_local_acceleration": args.include_local_acceleration,
                "include_local_stretching": args.include_local_stretching,
                "parameter_walk": args.parameter_walk,
                "parameter_walk_interval_steps": args.parameter_walk_interval,
                "parameter_walk_scale": args.parameter_walk_scale,
                "boundary_start_probability": args.boundary_start_probability,
                "evolve_dynamics": args.evolve_dynamics,
                "dynamics_substeps": args.dynamics_substeps,
                "use_basin_mixing": args.use_basin_mixing,
                "boundary_update_interval": args.boundary_update_interval,
                "boundary_resolution": args.boundary_resolution,
                "observation": [
                    "normalized_x",
                    "normalized_velocity",
                ]
                + (
                    ["normalized_local_acceleration"]
                    if args.include_local_acceleration
                    else []
                )
                + (["local_stretching_rate"] if args.include_local_stretching else []),
                "seed": args.seed,
            },
            indent=2,
        )
    )
    env.close()
    print(f"Saved model to {args.model_path}.zip")
    print(f"Saved experiment metadata to {metadata_path}")


if __name__ == "__main__":
    main()
