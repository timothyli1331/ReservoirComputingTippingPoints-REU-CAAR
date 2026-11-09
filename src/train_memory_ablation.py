"""Train matched persistent-memory and reset-each-step Reservoir-SAC models."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_csv(text: str, cast, label: str):
    try:
        values = [cast(value.strip()) for value in text.split(",") if value.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid {label}: {text!r}") from exc
    if not values:
        raise argparse.ArgumentTypeError(f"{label} cannot be empty")
    return values


def leak_label(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Pass additional shared trainer options after '--'. Example: "
            "-- --boundary-update-interval 5 --validation-interval 10000"
        ),
    )
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument(
        "--leak-rates",
        default="0.3",
        help="Comma-separated ReservoirPy leak rates (default: 0.3).",
    )
    parser.add_argument(
        "--seeds",
        default="0",
        help="Comma-separated matched training seeds (default: 0).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/memory_ablation"),
    )
    parser.add_argument("--prefix", default="reservoir_sac")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "training_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments forwarded to train_reservoir_sac_multi.py.",
    )
    args = parser.parse_args()
    if args.timesteps < 1:
        parser.error("--timesteps must be positive")
    try:
        args.leak_rates = parse_csv(args.leak_rates, float, "leak rates")
        args.seeds = parse_csv(args.seeds, int, "seeds")
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    if any(not 0.0 < rate <= 1.0 for rate in args.leak_rates):
        parser.error("every leak rate must be in (0, 1]")
    if args.training_args[:1] == ["--"]:
        args.training_args = args.training_args[1:]
    owned_options = (
        "--timesteps",
        "--reservoir-backend",
        "--memory-mode",
        "--leaking-rate",
        "--seed",
        "--model-path",
    )
    conflicts = [
        value
        for value in args.training_args
        if any(value == option or value.startswith(f"{option}=") for option in owned_options)
    ]
    if conflicts:
        parser.error(
            "the ablation runner controls these forwarded options: "
            + ", ".join(conflicts)
        )
    return args


def write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    trainer = Path(__file__).with_name("train_reservoir_sac_multi.py")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / f"{args.prefix}_manifest.json"
    manifest = {
        "purpose": "persistent versus reset-each-step reservoir memory ablation",
        "timesteps": args.timesteps,
        "leak_rates": args.leak_rates,
        "seeds": args.seeds,
        "shared_training_args": args.training_args,
        "runs": [],
    }

    for seed in args.seeds:
        for leak_rate in args.leak_rates:
            for memory_mode in ("persistent", "reset_each_step"):
                stem = (
                    f"{args.prefix}_{memory_mode}_"
                    f"leak_{leak_label(leak_rate)}_seed_{seed}"
                )
                model_path = args.output_dir / f"{stem}.pt"
                command = [
                    sys.executable,
                    str(trainer),
                    "--timesteps",
                    str(args.timesteps),
                    "--reservoir-backend",
                    "reservoirpy",
                    "--memory-mode",
                    memory_mode,
                    "--leaking-rate",
                    str(leak_rate),
                    "--seed",
                    str(seed),
                    "--model-path",
                    str(model_path),
                    *args.training_args,
                ]
                record = {
                    "memory_mode": memory_mode,
                    "leak_rate": leak_rate,
                    "seed": seed,
                    "model_path": str(model_path),
                    "metadata_path": str(model_path.with_suffix(".json")),
                    "command": command,
                    "status": "planned" if args.dry_run else "running",
                }
                manifest["runs"].append(record)
                write_manifest(manifest_path, manifest)
                print("\n" + " ".join(command), flush=True)
                if args.dry_run:
                    continue
                try:
                    subprocess.run(command, check=True)
                except subprocess.CalledProcessError as exc:
                    record["status"] = "failed"
                    record["return_code"] = exc.returncode
                    write_manifest(manifest_path, manifest)
                    raise
                else:
                    record["status"] = "completed"
                    write_manifest(manifest_path, manifest)

    print(f"\nSaved ablation manifest to {manifest_path}")


if __name__ == "__main__":
    main()
