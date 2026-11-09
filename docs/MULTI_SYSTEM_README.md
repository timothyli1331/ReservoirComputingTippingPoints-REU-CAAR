# Multi-system double-well transfer experiment

The Duffing-only and multi-system experiments now use the same adaptive-control
paradigm. On every control step, the policy makes a bounded phase-space move,
the physical parameters take a small bounded random-walk step, and the system
then evolves under the new parameters.

## Included systems

- `duffing`: deterministic quartic Duffing baseline.
- `langevin`: the same quartic potential with stochastic acceleration.
- `sextic`: a sixth-order potential with a quintic restoring force.
- `smooth_biquadratic`: a smooth approximation to two joined harmonic wells.

All systems have state `[x, v]`, action `[delta_x, delta_v]`, two stable wells,
and a saddle at the origin. This shared interface makes policy transfer
meaningful. Allen-Cahn and phi-four fields need spatial observations and
actions; Chua is three-dimensional and is not a potential system; the
Hamiltonian quartic system has centers rather than attracting basins when
damping is zero. Those should be separate follow-up experiments.

The policy observes only local quantities: state, local acceleration, and local
stretching (plus forcing phase for the forced Duffing system). It never receives
the physical parameters, basin label, boundary distance, or reward components
as input. Training rewards finite-time basin mixing in a small neighborhood of
the current state. For Langevin dynamics, this uses repeated stochastic
rollouts. The reported boundary score is `1 - basin_mixing`, so lower is better.

Episode starts are mixed: 60% are near the saddle's local stable direction and
40% cover a broader normalized phase-space region. This reduces specialization
to the origin. The local boundary reward is refreshed on every control step.

## Train on every system

```bash
.venv/bin/python src/train_multi.py --algorithm sac --timesteps 500000 \
  --model-path models/multi_sac

.venv/bin/python src/train_multi.py --algorithm ppo --timesteps 500000 \
  --model-path models/multi_ppo

.venv/bin/python src/trainrc_multi.py --generations 200 --population 32 \
  --episodes-per-candidate 4 --model-path models/multi_reservoir.pkl

# Stateful physical reservoir made from driven Duffing oscillators.
.venv/bin/python src/trainrc_multi.py --reservoir-backend duffing \
  --reservoir-units 48 --generations 200 --population 32 \
  --episodes-per-candidate 4 --model-path models/multi_duffing_reservoir.pkl
```

## Train the reservoir with SAC

This hybrid keeps the recurrent reservoir fixed, trains a linear Gaussian
actor readout with SAC, and uses small twin critics. Reservoir features are
stored directly in replay, so random replay batches do not lose recurrent
history.

Start with a bounded 20,000-transition experiment:

```bash
.venv/bin/python src/train_reservoir_sac_multi.py \
  --timesteps 20000 \
  --reservoir-units 100 \
  --boundary-update-interval 10 \
  --validation-interval 2000 \
  --validation-episodes-per-system 2 \
  --model-path models/multi_reservoir_sac.pt
```

The validation checkpoint uses the mean per-system return minus a penalty for
uneven system performance. This prevents a large Duffing or sextic gain from
hiding a collapse on Langevin or smooth biquadratic. `Control+C` is safe: the
best checkpoint and current latest checkpoint are saved separately.

Evaluate the best balanced checkpoint with ten fresh episodes per system:

```bash
.venv/bin/python src/evaluate_reservoir_sac_multi.py \
  --model-path models/multi_reservoir_sac.pt \
  --episodes-per-system 10
```

The evaluator automatically reads the training metadata beside the checkpoint
and reproduces its environment settings. It reports mean, worst-system, and
balanced returns along with boundary proximity and action saturation. It also
creates these files by default:

```text
results/multi_reservoir_sac_evaluation.json
results/multi_reservoir_sac_evaluation.png
results/multi_reservoir_sac_evaluation.gif
```

Use `--animation-stride`, `--animation-resolution`, and `--animation-fps` to
control GIF cost and playback. Pass `--no-generate-gif` when only the JSON and
static plot are needed.

To use the driven physical Duffing reservoir instead of ReservoirPy:

```bash
.venv/bin/python src/train_reservoir_sac_multi.py \
  --reservoir-backend duffing --reservoir-units 48 \
  --timesteps 20000 --boundary-update-interval 10 \
  --validation-interval 2000 \
  --model-path models/multi_duffing_reservoir_sac.pt
```

## Held-out transfer experiment

Train without `smooth_biquadratic`:

```bash
.venv/bin/python src/train_multi.py --algorithm sac --timesteps 500000 \
  --systems duffing,langevin,sextic \
  --model-path models/transfer_sac

.venv/bin/python src/trainrc_multi.py --generations 200 --population 32 \
  --episodes-per-candidate 3 --systems duffing,langevin,sextic \
  --model-path models/transfer_reservoir.pkl
```

Then evaluate on the training systems and the unseen target:

```bash
.venv/bin/python src/evaluate_transfer.py --algorithm sac \
  --sb3-model models/transfer_sac \
  --reservoir-model models/transfer_reservoir.pkl \
  --systems duffing,langevin,sextic,smooth_biquadratic \
  --episodes 20 --output transfer_comparison.png \
  --csv transfer_results.csv \
  --animate-output transfer_trajectories.gif
```

The animation synchronizes the first SAC/PPO and reservoir evaluation runs
across all selected systems. Use `--animation-stride` to control how many
simulation steps each frame advances and `--animation-fps` to control playback
speed. An `.mp4` output requires FFmpeg; `.gif` works through Pillow.

Parameters walk on every control step during both training and evaluation. The
animation reconstructs the instantaneous numerical boundary for each displayed
frame, and current parameter values appear above each panel. Increase
`--evaluation-parameter-walk-interval` for less frequent changes, or pass `0`
to perform a fixed-parameter comparison.

To show only the Duffing system with evolving parameters and physical drift:

```bash
.venv/bin/python src/evaluate_transfer.py \
  --systems duffing --episodes 1 --evolve-dynamics \
  --animate-output results/duffing_parameter_walk.gif \
  --evaluation-parameter-walk-interval 1 \
  --animation-stride 2 --animation-fps 3
```

The previous files in `models/` were trained with different observations and
transition rules. Retrain them before using this adaptive evaluation setup.

The default observation excludes the system identity. Add `--include-system-id`
to both training and evaluation only when testing model-conditioned multi-task
control rather than zero-shot transfer.
