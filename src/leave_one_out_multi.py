"""Leave-one-system-out transfer test for the physical Duffing reservoir.

For every selected double-well system, this script trains one shared
``DuffingPhysicalReservoir`` on trajectories from all *other* systems and then
evaluates it on the held-out one.  Coordinates are normalized by each system's
natural well and velocity scales, so the model must learn transferable local
dynamics rather than raw coordinate ranges.

Example:

    .venv/bin/python src/leave_one_out_multi.py --trajectory-steps 400 \
        --train-trajectories 16 --test-trajectories 8

Results, one model per fold, and a CSV summary are written under
``results/leave_one_out_physical_reservoir`` by default.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from physics_informed_reservoir import DuffingPhysicalReservoir
from systems import (
    DEFAULT_SYSTEM_NAMES,
    SYSTEMS,
    DoubleWellSystem,
    get_system,
    integrate_step,
    parse_system_names,
    velocity_scale,
)


def _normalized_state(state: np.ndarray, system: DoubleWellSystem) -> np.ndarray:
    """Express a state in units that are comparable between system families."""
    return np.array(
        [state[0] / system.well_location, state[1] / velocity_scale(system)],
        dtype=float,
    )


def generate_sequences(
    system: DoubleWellSystem,
    trajectories: int,
    steps: int,
    dt: float,
    seed: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Generate normalized one-step input/target trajectory pairs."""
    if trajectories < 1 or steps < 2 or dt <= 0:
        raise ValueError("trajectories and steps must be positive, with steps >= 2")
    rng = np.random.default_rng(seed)
    inputs, targets = [], []
    x_extent = 1.35 * system.well_location
    v_extent = 1.35 * velocity_scale(system)
    for _ in range(trajectories):
        state = np.array(
            [rng.uniform(-x_extent, x_extent), rng.uniform(-v_extent, v_extent)],
            dtype=float,
        )
        input_sequence = np.empty((steps, 2), dtype=float)
        target_sequence = np.empty((steps, 2), dtype=float)
        time = 0.0
        for index in range(steps):
            input_sequence[index] = _normalized_state(state, system)
            next_state = integrate_step(state, system, dt, time, rng)
            target_sequence[index] = _normalized_state(next_state, system)
            state = next_state
            time += dt
        inputs.append(input_sequence)
        targets.append(target_sequence)
    return inputs, targets


def _rmse(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((prediction - truth) ** 2, axis=0))


def evaluate_held_out_system(
    model: DuffingPhysicalReservoir,
    inputs: list[np.ndarray],
    targets: list[np.ndarray],
    washout: int,
    forecast_steps: int,
) -> dict[str, float]:
    """Return teacher-forced and autonomous errors for a held-out system."""
    one_step_predictions = []
    one_step_targets = []
    free_run_predictions = []
    free_run_targets = []
    for input_sequence, target_sequence in zip(inputs, targets):
        one_step = model.predict(input_sequence, reset=True)
        one_step_predictions.append(one_step[washout:])
        one_step_targets.append(target_sequence[washout:])

        horizon = min(forecast_steps, len(input_sequence) - washout)
        free_run = model.forecast(
            input_sequence[washout],
            steps=horizon,
            feedback_indices=(0, 1),
            warmup_inputs=input_sequence[:washout],
        )
        free_run_predictions.append(free_run)
        free_run_targets.append(target_sequence[washout : washout + horizon])

    one_step_error = _rmse(
        np.vstack(one_step_predictions), np.vstack(one_step_targets)
    )
    free_run_error = _rmse(
        np.vstack(free_run_predictions), np.vstack(free_run_targets)
    )
    return {
        "one_step_rmse_x": float(one_step_error[0]),
        "one_step_rmse_v": float(one_step_error[1]),
        "free_run_rmse_x": float(free_run_error[0]),
        "free_run_rmse_v": float(free_run_error[1]),
        "free_run_horizon": int(min(forecast_steps, len(inputs[0]) - washout)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--systems", default=",".join(DEFAULT_SYSTEM_NAMES))
    parser.add_argument(
        "--held-out",
        help="Comma-separated subset to hold out; defaults to every selected system.",
    )
    parser.add_argument("--train-trajectories", type=int, default=12)
    parser.add_argument("--test-trajectories", type=int, default=6)
    parser.add_argument("--trajectory-steps", type=int, default=300)
    parser.add_argument("--dt", type=float, default=0.03)
    parser.add_argument("--washout", type=int, default=50)
    parser.add_argument("--forecast-steps", type=int, default=150)
    parser.add_argument("--units", type=int, default=48)
    parser.add_argument("--ridge", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/leave_one_out_physical_reservoir")
    )
    args = parser.parse_args()

    system_names = parse_system_names(args.systems)
    held_out_names = (
        parse_system_names(args.held_out) if args.held_out else system_names
    )
    unknown_held_out = sorted(set(held_out_names) - set(system_names))
    if unknown_held_out:
        parser.error(f"held-out systems are not selected: {unknown_held_out}")
    if len(system_names) < 2:
        parser.error("leave-one-out learning requires at least two systems")
    if args.train_trajectories < 1 or args.test_trajectories < 1:
        parser.error("trajectory counts must be positive")
    if args.trajectory_steps < 2 or not 0 <= args.washout < args.trajectory_steps:
        parser.error("washout must be nonnegative and smaller than --trajectory-steps")
    if args.forecast_steps < 1 or args.dt <= 0 or args.units < 1 or args.ridge < 0:
        parser.error("dt, units, and forecast steps must be positive; ridge cannot be negative")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_data: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {}
    test_data: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {}
    for index, name in enumerate(system_names):
        system = get_system(name)
        train_data[name] = generate_sequences(
            system,
            args.train_trajectories,
            args.trajectory_steps,
            args.dt,
            args.seed + 1_000 * index,
        )
        test_data[name] = generate_sequences(
            system,
            args.test_trajectories,
            args.trajectory_steps,
            args.dt,
            args.seed + 100_000 + 1_000 * index,
        )

    records: list[dict[str, object]] = []
    for held_out in held_out_names:
        training_names = tuple(name for name in system_names if name != held_out)
        train_inputs = [
            sequence
            for name in training_names
            for sequence in train_data[name][0]
        ]
        train_targets = [
            sequence
            for name in training_names
            for sequence in train_data[name][1]
        ]
        model = DuffingPhysicalReservoir(
            input_size=2,
            output_size=2,
            units=args.units,
            dt=args.dt,
            ridge=args.ridge,
            residual_indices=(0, 1),
        ).fit_sequences(train_inputs, train_targets, washout=args.washout)

        metrics = evaluate_held_out_system(
            model,
            *test_data[held_out],
            washout=args.washout,
            forecast_steps=args.forecast_steps,
        )
        model_path = model.save(args.output_dir / f"without_{held_out}.pkl")
        record: dict[str, object] = {
            "held_out": held_out,
            "held_out_label": SYSTEMS[held_out].label,
            "training_systems": list(training_names),
            "model_path": str(model_path),
            **metrics,
        }
        records.append(record)
        print(
            f"held out {held_out:18s} | "
            f"one-step RMSE (x, v) = ({metrics['one_step_rmse_x']:.5f}, "
            f"{metrics['one_step_rmse_v']:.5f}) | "
            f"free-run {metrics['free_run_horizon']:3d}-step RMSE = "
            f"({metrics['free_run_rmse_x']:.5f}, {metrics['free_run_rmse_v']:.5f})"
        )

    config = {
        "systems": list(system_names),
        "held_out": list(held_out_names),
        "train_trajectories": args.train_trajectories,
        "test_trajectories": args.test_trajectories,
        "trajectory_steps": args.trajectory_steps,
        "dt": args.dt,
        "washout": args.washout,
        "forecast_steps": args.forecast_steps,
        "units": args.units,
        "ridge": args.ridge,
        "seed": args.seed,
        "normalization": "x / well_location, velocity / velocity_scale",
    }
    results_path = args.output_dir / "results.json"
    results_path.write_text(json.dumps({"config": config, "folds": records}, indent=2))
    csv_path = args.output_dir / "results.csv"
    with csv_path.open("w", newline="") as file:
        columns = [
            "held_out",
            "held_out_label",
            "training_systems",
            "model_path",
            "one_step_rmse_x",
            "one_step_rmse_v",
            "free_run_horizon",
            "free_run_rmse_x",
            "free_run_rmse_v",
        ]
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["training_systems"] = ",".join(record["training_systems"])
            writer.writerow(row)
    print(f"Saved leave-one-out results to {results_path}")
    print(f"Saved CSV summary to {csv_path}")


if __name__ == "__main__":
    main()
