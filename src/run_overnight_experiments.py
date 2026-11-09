"""Run the three memory-sensitive SAC experiments sequentially and quietly.

This runner adds targeted conditions to the already-completed baseline:
  far_start    -- mostly start away from the boundary
  high_drift   -- parameter random walk has 4x larger increments
  no_stretch   -- remove local stretching rate from the observation
  no_accel_no_stretch -- observe only position and velocity
  state_only_strong_drift -- position/velocity with 3x physical drift

Every condition trains matched regular SAC, persistent Reservoir-SAC, and
reset-each-step Reservoir-SAC models for every seed, then evaluates them.
Subprocess output is retained in logs instead of printing continuously.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev


@dataclass(frozen=True)
class Condition:
    name: str
    boundary_start_probability: float
    parameter_walk_scale: float
    include_local_acceleration: bool
    include_local_stretching: bool
    dynamics_substeps: int
    description: str


CONDITIONS = (
    Condition("far_start", 0.15, 1.0, True, True, 1, "85% of resets begin away from the boundary"),
    Condition("high_drift", 0.60, 4.0, True, True, 1, "four-times-larger parameter random-walk increments"),
    Condition("no_stretch", 0.60, 1.0, True, False, 1, "local stretching rate removed from observations"),
    Condition(
        "no_accel_no_stretch",
        0.60,
        1.0,
        False,
        False,
        1,
        "observations contain only normalized position and velocity",
    ),
    Condition(
        "state_only_strong_drift",
        0.60,
        1.0,
        False,
        False,
        3,
        "position/velocity observations with three dynamics intervals per action",
    ),
)


def parse_seeds(text: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in text.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--seeds must be comma-separated integers") from exc
    if not values:
        raise argparse.ArgumentTypeError("--seeds cannot be empty")
    return values


def parse_conditions(text: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in text.split(",") if item.strip())
    valid = {condition.name for condition in CONDITIONS}
    unknown = sorted(set(values) - valid)
    if not values:
        raise argparse.ArgumentTypeError("--conditions cannot be empty")
    if unknown:
        raise argparse.ArgumentTypeError(
            "unknown conditions: "
            + ", ".join(unknown)
            + "; choose from "
            + ", ".join(sorted(valid))
        )
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds("0,1,2,3,4"))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--leak-rate", type=float, default=0.3)
    parser.add_argument(
        "--conditions",
        type=parse_conditions,
        default=tuple(condition.name for condition in CONDITIONS),
        help="Comma-separated experiment sections to run.",
    )
    parser.add_argument("--model-root", type=Path, default=Path("models/overnight_memory_tests"))
    parser.add_argument("--result-root", type=Path, default=Path("results/overnight_memory_tests"))
    parser.add_argument("--log-root", type=Path, default=Path("logs/overnight_memory_tests"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true", help="Print planned stages without running them.")
    args = parser.parse_args()
    if args.timesteps < 1 or args.episodes < 1 or not 0.0 < args.leak_rate <= 1.0:
        parser.error("timesteps/episodes must be positive and leak rate must be in (0, 1]")
    return args


def flag(name: str, value: bool) -> str:
    return f"--{name}" if value else f"--no-{name}"


def status(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def resolve_under(root: Path, path: Path) -> Path:
    """Keep default relative output paths anchored to the project directory."""
    return path if path.is_absolute() else root / path


def preflight(script_dir: Path, python_executable: str) -> None:
    """Fail quickly before an overnight run if a script or package is missing."""
    script_names = (
        "train_memory_ablation.py",
        "train_reservoir_sac_multi.py",
        "train_multi.py",
        "evaluate_transfer.py",
    )
    missing_scripts = [name for name in script_names if not (script_dir / name).is_file()]
    if missing_scripts:
        raise FileNotFoundError(
            "Missing required scripts: " + ", ".join(missing_scripts)
        )
    dependency_check = (
        "import gymnasium, matplotlib, numpy, reservoirpy, "
        "stable_baselines3, torch"
    )
    subprocess.run(
        [python_executable, "-c", dependency_check],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def run(command: list[str], log_path: Path, dry_run: bool) -> None:
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    matplotlib_cache = log_path.parent / ".matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["MPLCONFIGDIR"] = str(matplotlib_cache)
    with log_path.open("w") as log_file:
        try:
            subprocess.run(
                command,
                check=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=environment,
            )
        except subprocess.CalledProcessError:
            status(f"FAILED: see {log_path}")
            raise


def condition_flags(condition: Condition) -> list[str]:
    return [
        "--parameter-walk", "--parameter-walk-interval", "1",
        "--parameter-walk-scale", str(condition.parameter_walk_scale),
        "--boundary-start-probability", str(condition.boundary_start_probability),
        flag("include-local-acceleration", condition.include_local_acceleration),
        flag("include-local-stretching", condition.include_local_stretching),
        "--dynamics-substeps", str(condition.dynamics_substeps),
        "--evolve-dynamics", "--use-basin-mixing",
    ]


def aggregate(condition: Condition, result_dir: Path, seeds: list[int]) -> None:
    """Summarize episode data by training seed, then across training seeds."""
    values: dict[tuple[str, str, int], list[tuple[float, float]]] = {}
    for seed in seeds:
        for comparison, policy_names in (("persistent", {"SAC": "SAC", "Reservoir-SAC": "persistent"}),
                                         ("reset_each_step", {"Reservoir-SAC": "reset_each_step"})):
            path = result_dir / f"sac_vs_{comparison}_seed_{seed}.csv"
            with path.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    if row["model"] not in policy_names:
                        continue
                    key = (policy_names[row["model"]], row["system"], seed)
                    values.setdefault(key, []).append((float(row["return"]), float(row["mean_boundary_score"])))

    grouped: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for (policy, system, _seed), episodes in values.items():
        returns, scores = zip(*episodes)
        grouped.setdefault((policy, system), []).append((mean(returns), mean(scores)))
    output = result_dir / "summary.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("condition", "policy", "system", "training_seeds", "mean_return", "std_return", "mean_boundary_score", "std_boundary_score"))
        writer.writeheader()
        for (policy, system), seed_values in sorted(grouped.items()):
            returns, scores = zip(*seed_values)
            writer.writerow({"condition": condition.name, "policy": policy, "system": system,
                             "training_seeds": len(returns), "mean_return": mean(returns),
                             "std_return": stdev(returns) if len(returns) > 1 else 0.0,
                             "mean_boundary_score": mean(scores),
                             "std_boundary_score": stdev(scores) if len(scores) > 1 else 0.0})


def run_condition(
    args: argparse.Namespace,
    condition: Condition,
    script_dir: Path,
    python_executable: str,
) -> None:
    scripts = {name: script_dir / f"{name}.py" for name in ("train_memory_ablation", "train_multi", "evaluate_transfer")}
    model_dir, result_dir, log_dir = (args.model_root / condition.name, args.result_root / condition.name, args.log_root / condition.name)
    shared = condition_flags(condition)
    result_dir.mkdir(parents=True, exist_ok=True)
    status(f"START {condition.name}: {condition.description}")

    reservoir_command = [python_executable, str(scripts["train_memory_ablation"]), "--timesteps", str(args.timesteps), "--seeds", ",".join(map(str, args.seeds)), "--leak-rates", str(args.leak_rate), "--output-dir", str(model_dir), "--prefix", "reservoir_sac", "--", *shared, "--boundary-update-interval", "5", "--validation-interval", "10000", "--validation-episodes-per-system", "5", "--device", args.device]
    run(reservoir_command, log_dir / "reservoir_training.log", args.dry_run)
    status(f"{condition.name}: reservoir pair training complete")

    for seed in args.seeds:
        regular_command = [python_executable, str(scripts["train_multi"]), "--algorithm", "sac", "--timesteps", str(args.timesteps), "--seed", str(seed), "--model-path", str(model_dir / f"regular_sac_seed_{seed}"), *shared, "--boundary-update-interval", "5"]
        run(regular_command, log_dir / f"regular_sac_seed_{seed}.log", args.dry_run)
    status(f"{condition.name}: regular SAC training complete")

    leak_label = f"{args.leak_rate:g}".replace(".", "p")
    for seed in args.seeds:
        for memory_mode in ("persistent", "reset_each_step"):
            reservoir_model = model_dir / f"reservoir_sac_{memory_mode}_leak_{leak_label}_seed_{seed}.pt"
            eval_command = [python_executable, str(scripts["evaluate_transfer"]), "--algorithm", "sac", "--sb3-model", str(model_dir / f"regular_sac_seed_{seed}"), "--reservoir-sac-model", str(reservoir_model), "--episodes", str(args.episodes), "--seed", str(2_000_000 + seed * 100), "--output", str(result_dir / f"sac_vs_{memory_mode}_seed_{seed}.png"), "--csv", str(result_dir / f"sac_vs_{memory_mode}_seed_{seed}.csv"), "--evaluation-parameter-walk-interval", "1", "--evaluation-parameter-walk-scale", str(condition.parameter_walk_scale), "--boundary-start-probability", str(condition.boundary_start_probability), flag("include-local-acceleration", condition.include_local_acceleration), flag("include-local-stretching", condition.include_local_stretching), "--dynamics-substeps", str(condition.dynamics_substeps), "--evolve-dynamics", "--use-basin-mixing"]
            run(eval_command, log_dir / f"evaluate_{memory_mode}_seed_{seed}.log", args.dry_run)
    if not args.dry_run:
        aggregate(condition, result_dir, args.seeds)
    status(f"DONE {condition.name}: models, episode CSVs, plots, and summary are saved")


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    root = script_dir.parent
    project_python = root / ".venv" / "bin" / "python"
    python_executable = (
        str(project_python) if project_python.is_file() else sys.executable
    )
    status(f"Using Python interpreter: {python_executable}")
    preflight(script_dir, python_executable)
    args.model_root = resolve_under(root, args.model_root)
    args.result_root = resolve_under(root, args.result_root)
    args.log_root = resolve_under(root, args.log_root)
    status("Preflight passed: scripts, paths, and Python packages are available")
    selected_conditions = tuple(
        condition for condition in CONDITIONS if condition.name in args.conditions
    )
    manifest = {
        "python_executable": python_executable,
        "timesteps": args.timesteps,
        "seeds": args.seeds,
        "episodes": args.episodes,
        "leak_rate": args.leak_rate,
        "model_root": str(args.model_root),
        "result_root": str(args.result_root),
        "log_root": str(args.log_root),
        "conditions": [asdict(item) for item in selected_conditions],
    }
    args.result_root.mkdir(parents=True, exist_ok=True)
    (args.result_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for condition in selected_conditions:
        run_condition(args, condition, script_dir, python_executable)
    status("ALL OVERNIGHT EXPERIMENTS COMPLETE")


if __name__ == "__main__":
    main()
