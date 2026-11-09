"""Generate a reproducible Duffing parameter random walk for fair evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dynamics import walk_duffing_params
from environment import DEFAULT_PARAMS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--alpha-step", type=float, default=0.025)
    parser.add_argument("--beta-step", type=float, default=0.025)
    parser.add_argument("--damping-step", type=float, default=0.010)
    parser.add_argument(
        "--output", type=Path, default=Path("test_data/duffing_parameter_walk.json")
    )
    args = parser.parse_args()
    if args.episodes < 1:
        raise ValueError("episodes must be positive")

    step_scales = {
        "alpha": args.alpha_step,
        "beta": args.beta_step,
        "damping": args.damping_step,
    }
    rng = np.random.default_rng(args.seed)
    params = dict(DEFAULT_PARAMS)
    episodes = []
    for episode in range(args.episodes):
        if episode > 0:
            walk_duffing_params(params, rng=rng, step_scales=step_scales)
        episodes.append(
            {
                "episode": episode,
                "alpha": float(params["alpha"]),
                "beta": float(params["beta"]),
                "damping": float(params["damping"]),
                "gamma": float(params["gamma"]),
                "omega": float(params["omega"]),
                "max_action": float(params["max_action"]),
            }
        )

    payload = {
        "description": "Bounded Duffing parameter random walk for paired model evaluation",
        "seed": args.seed,
        "step_scales": step_scales,
        "bounds": {
            "alpha": [-1.25, -0.75],
            "beta": [0.75, 1.25],
            "damping": [0.15, 0.40],
        },
        "episodes": episodes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"Saved {args.episodes} evaluation episodes to {args.output}")


if __name__ == "__main__":
    main()
