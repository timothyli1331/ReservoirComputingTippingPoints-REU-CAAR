# Adaptive boundary-tracking paradigm

Both the Duffing-only and multi-system trainers use the same step order:

1. The policy observes local state, acceleration, and stretching information.
2. It selects a small action in `[delta_x, delta_v]`, bounded by `0.05`.
3. The physical parameters take a small bounded random-walk step.
4. The action is applied and the physical system evolves under the new values.
5. A local basin-mixing probe supplies the training reward.

The policy never observes the basin-mixing score, numerical boundary, reward
components, or parameter values. Those quantities are available only to the
training environment. Consequently, the policy must react to changes visible
in the local vector field rather than memorize a boundary curve.

The parameter walk uses reflected bounds, so Duffing remains in its double-well
regime and every multi-system oscillator retains two wells and a saddle. The
per-action standard deviations are scaled so their accumulated diffusion over
50 steps is comparable to the former coarser parameter walk.

## Train Duffing policies

```bash
.venv/bin/python src/train.py --algorithm sac --timesteps 500000 \
  --model-path models/adaptive_duffing_sac

.venv/bin/python src/trainrc.py --generations 200 --population 32 \
  --episodes-per-candidate 4 \
  --model-path models/adaptive_duffing_reservoir.pkl

# Use driven Duffing oscillators as the stateful policy reservoir.
.venv/bin/python src/trainrc.py --reservoir-backend duffing \
  --reservoir-units 48 --generations 200 --population 32 \
  --episodes-per-candidate 4 \
  --model-path models/adaptive_duffing_physical_reservoir.pkl
```

The Duffing backend retains every oscillator's position and velocity between
control steps and clears them only at episode boundaries. Its default memory
timescale is controlled by `--duffing-damping 0.25` and
`--duffing-dt 0.04`. Lower damping generally retains input history longer;
input scaling and damping should be tuned together to avoid persistent
well-switching or unstable trajectories.

Reservoir policy search uses mirrored CEM samples and smoothed distribution
updates. A fixed held-out seed suite is evaluated throughout training, and the
saved readout is the best validation checkpoint rather than whichever elite
mean happened to be produced by the last generation. The search standard
deviation is allowed to converge below `0.01`, but it cannot collapse by more
than five percent in one generation. Validation patience can stop a run that
has plateaued; set `--early-stopping-patience 0` to disable that behavior.

For a noisy or strongly walking environment, increase
`--validation-episodes` before increasing the reservoir size. A larger fixed
reservoir also increases the dimension of the derivative-free readout search;
it is not automatically a more capable trained controller.

## Train multi-system policies

```bash
.venv/bin/python src/train_multi.py --algorithm sac --timesteps 500000 \
  --model-path models/adaptive_multi_sac

.venv/bin/python src/trainrc_multi.py --generations 200 --population 32 \
  --episodes-per-candidate 4 \
  --model-path models/adaptive_multi_reservoir.pkl

.venv/bin/python src/trainrc_multi.py --reservoir-backend duffing \
  --reservoir-units 48 --generations 200 --population 32 \
  --episodes-per-candidate 4 \
  --model-path models/adaptive_multi_physical_reservoir.pkl
```

## Compare Duffing policies side by side

After both Duffing models have been trained, animate them on the same initial
condition and the same parameter random walk:

```bash
.venv/bin/python src/compare_models.py \
  --sb3-model models/adaptive_duffing_sac \
  --reservoir-model models/adaptive_duffing_reservoir.pkl \
  --episodes 10 \
  --output results/duffing_sac_vs_reservoir.png \
  --animate-output results/duffing_sac_vs_reservoir.gif \
  --parameter-walk-interval 1 \
  --animation-stride 2 \
  --animation-resolution 31 \
  --animation-fps 3
```

Both panels reconstruct the changing finite-time boundary at every displayed
frame. The header shows the policies' current phase-space distance, its running
mean, and the full animated-episode mean. The console also reports paired mean
differences in return and boundary score over all requested episodes.

Physical evolution, per-action parameter walking, and local basin-mixing reward
are enabled by default. Their Boolean flags have corresponding `--no-*`
variants for controlled ablations. Models trained under the previous
observation and transition definitions are incompatible and must be retrained.
