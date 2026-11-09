"""Train a Stable-Baselines3 Duffing boundary agent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent import build_agent
from environment import DuffingBoundaryEnv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--algorithm", choices=("sac", "ppo"), default="sac")
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
    parser.add_argument("--model-path", type=Path, default=Path("models/duffing_agent"))
    args = parser.parse_args()
    if args.parameter_walk_interval < 1 or args.boundary_update_interval < 1:
        parser.error("walk and boundary update intervals must be at least 1")

    env = DuffingBoundaryEnv(
        parameter_walk=args.parameter_walk,
        parameter_walk_interval_steps=args.parameter_walk_interval,
        evolve_dynamics=args.evolve_dynamics,
        use_basin_mixing=args.use_basin_mixing,
        boundary_update_interval=args.boundary_update_interval,
        global_boundary_resolution=args.boundary_resolution,
    )
    agent = build_agent(env, algorithm=args.algorithm, seed=args.seed)
    agent.learn(total_timesteps=args.timesteps)
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(str(args.model_path))
    metadata_path = args.model_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "algorithm": args.algorithm,
                "timesteps": args.timesteps,
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
    print(f"Saved trained {args.algorithm.upper()} model to {args.model_path}.zip")
    print(f"Saved experiment metadata to {metadata_path}")


if __name__ == "__main__":
    main()
