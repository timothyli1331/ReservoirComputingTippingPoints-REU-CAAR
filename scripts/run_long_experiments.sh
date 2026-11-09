#!/bin/zsh

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
RUN_ROOT="results/long_walk_experiments"
LOG_ROOT="logs/long_walk_experiments"
START_AT="${START_AT:-}"
START_REACHED=0
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-$LOG_ROOT/matplotlib_cache}"

mkdir -p "$RUN_ROOT" "$LOG_ROOT" "$MPLCONFIGDIR" models
FAILED_STAGES=()
STATUS_LOG="$LOG_ROOT/run_status.log"

record_status() {
    local message="$1"
    print "$message"
    print "$message" >> "$STATUS_LOG"
}

run_stage() {
    local stage="$1"
    shift
    if [[ -n "$START_AT" && "$START_REACHED" -eq 0 ]]; then
        if [[ "$stage" == "$START_AT" ]]; then
            START_REACHED=1
            record_status "Resuming at stage: $stage"
        else
            record_status "Skipping completed stage before resume point: $stage"
            return 0
        fi
    fi
    record_status "[$(date '+%Y-%m-%d %H:%M:%S')] START $stage"
    if "$@" </dev/null >/dev/null 2>&1; then
        record_status "[$(date '+%Y-%m-%d %H:%M:%S')] DONE  $stage"
    else
        local exit_code=$?
        FAILED_STAGES+=("$stage:$exit_code")
        record_status "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED $stage (exit $exit_code)"
    fi
}

# The physical reservoir is a dynamics forecaster rather than a control policy.
# Its CLI fits the readout, evaluates teacher-forced/free-running forecasts,
# saves the trained model, and writes the evaluation chart in one run.
run_stage physical_duffing_forecast \
    "$PYTHON_BIN" src/physics_informed_reservoir.py \
    --samples 40000 \
    --units 96 \
    --washout 300 \
    --forecast-steps 2000 \
    --ridge 1e-5 \
    --model-path models/physical_duffing_forecaster.pkl \
    --plot "$RUN_ROOT/physical_duffing_forecast.png"

run_stage physical_multi_leave_one_out \
    "$PYTHON_BIN" src/leave_one_out_multi.py \
    --systems duffing,langevin,sextic,smooth_biquadratic \
    --train-trajectories 32 \
    --test-trajectories 12 \
    --trajectory-steps 600 \
    --washout 100 \
    --forecast-steps 300 \
    --units 96 \
    --output-dir "$RUN_ROOT/physical_multi_leave_one_out"

# Duffing-only, parameters walking after every control action.
run_stage duffing_walk1_sac_train \
    "$PYTHON_BIN" src/train.py \
    --algorithm sac \
    --timesteps 200000 \
    --parameter-walk-interval 1 \
    --boundary-update-interval 1 \
    --seed 101 \
    --model-path models/duffing_walk1_long_sac

run_stage duffing_walk1_reservoir_train \
    "$PYTHON_BIN" src/trainrc.py \
    --generations 80 \
    --population 32 \
    --episodes-per-candidate 2 \
    --reservoir-units 50 \
    --parameter-walk-interval 1 \
    --boundary-update-interval 1 \
    --validation-episodes 8 \
    --early-stopping-patience 20 \
    --seed 101 \
    --model-path models/duffing_walk1_long_reservoir.pkl

run_stage duffing_walk1_compare \
    "$PYTHON_BIN" src/compare_models.py \
    --algorithm sac \
    --sb3-model models/duffing_walk1_long_sac \
    --reservoir-model models/duffing_walk1_long_reservoir.pkl \
    --episodes 20 \
    --seed 20001 \
    --parameter-walk-interval 1 \
    --output "$RUN_ROOT/duffing_walk1_comparison.png" \
    --animate-output "$RUN_ROOT/duffing_walk1_comparison.gif" \
    --animation-stride 5 \
    --animation-resolution 31 \
    --animation-fps 5

# Duffing-only, parameters walking after every fifth control action.
run_stage duffing_walk5_sac_train \
    "$PYTHON_BIN" src/train.py \
    --algorithm sac \
    --timesteps 200000 \
    --parameter-walk-interval 5 \
    --boundary-update-interval 1 \
    --seed 105 \
    --model-path models/duffing_walk5_long_sac

run_stage duffing_walk5_reservoir_train \
    "$PYTHON_BIN" src/trainrc.py \
    --generations 80 \
    --population 32 \
    --episodes-per-candidate 2 \
    --reservoir-units 50 \
    --parameter-walk-interval 5 \
    --boundary-update-interval 1 \
    --validation-episodes 8 \
    --early-stopping-patience 20 \
    --seed 105 \
    --model-path models/duffing_walk5_long_reservoir.pkl

run_stage duffing_walk5_compare \
    "$PYTHON_BIN" src/compare_models.py \
    --algorithm sac \
    --sb3-model models/duffing_walk5_long_sac \
    --reservoir-model models/duffing_walk5_long_reservoir.pkl \
    --episodes 20 \
    --seed 20005 \
    --parameter-walk-interval 5 \
    --output "$RUN_ROOT/duffing_walk5_comparison.png" \
    --animate-output "$RUN_ROOT/duffing_walk5_comparison.gif" \
    --animation-stride 5 \
    --animation-resolution 31 \
    --animation-fps 5

# Multi-system, parameters walking after every control action.
run_stage multi_walk1_sac_train \
    "$PYTHON_BIN" src/train_multi.py \
    --algorithm sac \
    --timesteps 200000 \
    --systems duffing,langevin,sextic,smooth_biquadratic \
    --parameter-walk-interval 1 \
    --boundary-update-interval 1 \
    --seed 201 \
    --model-path models/multi_walk1_long_sac

run_stage multi_walk1_reservoir_train \
    "$PYTHON_BIN" src/trainrc_multi.py \
    --generations 60 \
    --population 24 \
    --episodes-per-candidate 4 \
    --reservoir-units 50 \
    --systems duffing,langevin,sextic,smooth_biquadratic \
    --parameter-walk-interval 1 \
    --boundary-update-interval 1 \
    --validation-episodes 8 \
    --early-stopping-patience 20 \
    --seed 201 \
    --model-path models/multi_walk1_long_reservoir.pkl

run_stage multi_walk1_evaluate \
    "$PYTHON_BIN" src/evaluate_transfer.py \
    --algorithm sac \
    --sb3-model models/multi_walk1_long_sac \
    --reservoir-model models/multi_walk1_long_reservoir.pkl \
    --systems duffing,langevin,sextic,smooth_biquadratic \
    --episodes 20 \
    --seed 30001 \
    --evaluation-parameter-walk-interval 1 \
    --output "$RUN_ROOT/multi_walk1_comparison.png" \
    --csv "$RUN_ROOT/multi_walk1_results.csv" \
    --animate-output "$RUN_ROOT/multi_walk1_comparison.gif" \
    --animation-stride 5 \
    --animation-resolution 25 \
    --animation-fps 5

# Multi-system, parameters walking after every fifth control action.
run_stage multi_walk5_sac_train \
    "$PYTHON_BIN" src/train_multi.py \
    --algorithm sac \
    --timesteps 200000 \
    --systems duffing,langevin,sextic,smooth_biquadratic \
    --parameter-walk-interval 5 \
    --boundary-update-interval 1 \
    --seed 205 \
    --model-path models/multi_walk5_long_sac

run_stage multi_walk5_reservoir_train \
    "$PYTHON_BIN" src/trainrc_multi.py \
    --generations 60 \
    --population 24 \
    --episodes-per-candidate 4 \
    --reservoir-units 50 \
    --systems duffing,langevin,sextic,smooth_biquadratic \
    --parameter-walk-interval 5 \
    --boundary-update-interval 1 \
    --validation-episodes 8 \
    --early-stopping-patience 20 \
    --seed 205 \
    --model-path models/multi_walk5_long_reservoir.pkl

run_stage multi_walk5_evaluate \
    "$PYTHON_BIN" src/evaluate_transfer.py \
    --algorithm sac \
    --sb3-model models/multi_walk5_long_sac \
    --reservoir-model models/multi_walk5_long_reservoir.pkl \
    --systems duffing,langevin,sextic,smooth_biquadratic \
    --episodes 20 \
    --seed 30005 \
    --evaluation-parameter-walk-interval 5 \
    --output "$RUN_ROOT/multi_walk5_comparison.png" \
    --csv "$RUN_ROOT/multi_walk5_results.csv" \
    --animate-output "$RUN_ROOT/multi_walk5_comparison.gif" \
    --animation-stride 5 \
    --animation-resolution 25 \
    --animation-fps 5

if [[ -n "$START_AT" && "$START_REACHED" -eq 0 ]]; then
    record_status "Unknown START_AT stage: $START_AT"
    exit 2
fi

if (( ${#FAILED_STAGES[@]} )); then
    record_status "[$(date '+%Y-%m-%d %H:%M:%S')] RUN FINISHED WITH FAILURES"
    for failed_stage in "${FAILED_STAGES[@]}"; do
        record_status "$failed_stage"
    done
    exit 1
fi

record_status "[$(date '+%Y-%m-%d %H:%M:%S')] ALL EXPERIMENTS COMPLETE"
