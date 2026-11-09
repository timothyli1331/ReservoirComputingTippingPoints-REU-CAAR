import pandas as pd
from pathlib import Path

# Change this to the directory containing the five CSVs
RESULT_DIR = Path("results/recurrent_controller_tests_latest/no_accel_no_stretch")

files = sorted(RESULT_DIR.glob("controller_comparison_seed_*.csv"))

if len(files) != 5:
    raise ValueError(f"Expected 5 seed files, found {len(files)}")

# Load all episode-level results
dfs = [pd.read_csv(path) for path in files]
data = pd.concat(dfs, ignore_index=True)

# First average the 20 evaluation episodes within each
# training seed / model / system combination.
per_seed = (
    data
    .groupby(["model", "system", "training_seed"])
    .agg(
        mean_return=("return", "mean"),
        mean_boundary_score=("mean_boundary_score", "mean"),
        mean_survived_steps=("survived_steps", "mean"),
    )
    .reset_index()
)

# Then calculate mean and SAMPLE SD across the 5 training seeds.
summary = (
    per_seed
    .groupby(["model", "system"])
    .agg(
        mean_return=("mean_return", "mean"),
        std_return=("mean_return", "std"),

        mean_boundary_score=("mean_boundary_score", "mean"),
        std_boundary_score=("mean_boundary_score", "std"),

        mean_survived_steps=("mean_survived_steps", "mean"),
        std_survived_steps=("mean_survived_steps", "std"),

        training_seeds=("training_seed", "nunique"),
    )
    .reset_index()
)

print("\nPER-SEED RESULTS:")
print(per_seed.to_string(index=False))

print("\nFINAL SUMMARY:")
print(summary.to_string(index=False))

# Save the results
output = RESULT_DIR / "summary_recalculated.csv"
summary.to_csv(output, index=False)

print(f"\nSaved to: {output}")