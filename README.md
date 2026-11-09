# ReservoirComputingTippingPoints-REU-CAAR
Testing Reservoir Computing methods on simulations involving tipping points with modified reinforcement learning techniques; Research project made at REU CAAR

## Duffing oscillator research

This archive follows the REU work from single-system Duffing basin-boundary
tracking to control and transfer across Duffing, Langevin, sextic, and smooth
biquadratic double-well systems. It includes PPO and SAC baselines, classical
and physical reservoir policies, Reservoir-SAC memory ablations, and matched
feedforward, LSTM, and echo state network SAC comparisons.

## Setup

The archived environment used Python 3.13. Run the commands from the repository root.

```sh
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
PYTHONPATH=src python -m unittest discover -s tests -v
```

FFmpeg is needed for MP4 exports. GIF exports use Pillow.

## Research notes and workflows

| Workflow | Where to start |
| --- | --- |
| Adaptive Duffing boundary tracking | [Training paradigm](docs/TRAINING_PARADIGM.md), [baseline trainer](src/train.py), and [reservoir trainer](src/trainrc.py) |
| Multi-system training and held-out transfer | [Multi-system guide](docs/MULTI_SYSTEM_README.md) and [transfer evaluator](src/evaluate_transfer.py) |
| Physical reservoir forecasting | [Duffing forecaster](src/physics_informed_reservoir.py) and [leave-one-system-out evaluation](src/leave_one_out_multi.py) |
| Reservoir-SAC and memory ablation | [Architecture proposal](docs/RESERVOIR_SAC_PROPOSAL.md), [ablation guide](docs/MEMORY_ABLATION.md), and [overnight runner](src/run_overnight_experiments.py) |
| Matched recurrent SAC comparisons | [Comparison runner](src/run_sb3_reservoir_input_overnight.py) and [trajectory plotting](src/plot_recurrent_paths.py) |
| Long parameter walks | [Experiment script](scripts/run_long_experiments.sh) and [fixed walk schedule](test_data/duffing_parameter_walk.json) |
| Alternative computing methods | [Research notes (PDF)](docs/ALTERNATIVE_COMPUTING_METHODS.pdf) and [LaTeX source](docs/ALTERNATIVE_COMPUTING_METHODS.tex) |

Run Python entry points as `python src/<script>.py` and the long-walk
workflow as `zsh scripts/run_long_experiments.sh`. Check each entry point's
`--help` for available options. Several runners write into the archived
experiment directories by default, so pass separate output paths for new runs
when the CLI supports them.

## Archived material

- [Results](results/) contains episode CSVs, run manifests, summaries, plots,
  and animations. It includes the [long-walk experiments](results/long_walk_experiments/),
  [memory experiments](results/overnight_memory_tests/), and both the
  [earlier](results/recurrent_controller_tests/) and
  [latest](results/recurrent_controller_tests_latest/) recurrent comparisons.
- [Models](models/) contains trained checkpoints and run metadata. The archive
  includes about 380 MB of model files. Some checkpoints use pickle-based
  serialization and should only be loaded from a trusted checkout.
- [Figures](figures/phase_space/) contains phase-space diagrams for all four
  systems. Regenerate them with `python src/phase_space_diagrams.py`.
- [Logs](logs/) contains historical training and evaluation output. The
  archived logs and metadata use project-relative paths in place of the
  original local checkout prefix. Historical commands reflect the layout at
  the time of the run.
