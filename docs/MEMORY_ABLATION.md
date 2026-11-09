# Reservoir-SAC memory ablation

The controlled comparison uses the same reservoir, actor, critics, exploration
settings, environment, and seed in both conditions:

- `persistent`: reservoir state is retained throughout each episode.
- `reset_each_step`: reservoir state is cleared before every observation.

Only the first condition contains temporal memory. Both conditions retain the
same current-observation skip connection and nonlinear reservoir feature map.

## Train one matched pair

```bash
.venv/bin/python src/train_memory_ablation.py \
  --timesteps 200000 \
  --leak-rates 0.3 \
  --seeds 0 \
  --output-dir models/memory_ablation
```

This creates best-validation checkpoints and final/latest checkpoints for both
memory modes. The JSON manifest records every exact training command.

The primary paper comparison is between these two matched reservoir models.
Train conventional SAC separately as a secondary architecture baseline:

```bash
.venv/bin/python src/train_multi.py \
  --algorithm sac \
  --timesteps 200000 \
  --boundary-update-interval 5 \
  --seed 0 \
  --model-path models/memory_ablation/regular_sac_seed_0
```

## Train a leak-rate and seed sweep

```bash
.venv/bin/python src/train_memory_ablation.py \
  --timesteps 200000 \
  --leak-rates 0.05,0.1,0.3,0.6,1.0 \
  --seeds 0,1,2,3,4 \
  --output-dir models/memory_ablation \
  -- --boundary-update-interval 5 \
     --validation-interval 10000 \
     --validation-episodes-per-system 5
```

Arguments after `--` are forwarded unchanged to both matched conditions.
Use `--dry-run` to inspect the commands and manifest without training.

Keep the parameter walk enabled while making it harder with, for example,
`--parameter-walk-scale 2.0` or `--parameter-walk-scale 4.0`. This multiplies
the standard deviation of every per-step parameter increment while retaining
the existing reflected physical bounds.

The default starts near the boundary 60% of the time. Use
`--boundary-start-probability 0.15` in both matched training conditions to
make boundary acquisition, rather than only boundary tracking, the task.

## Train either condition directly

```bash
.venv/bin/python src/train_reservoir_sac_multi.py \
  --memory-mode persistent \
  --leaking-rate 0.3 \
  --timesteps 200000 \
  --model-path models/persistent_reservoir_sac.pt

.venv/bin/python src/train_reservoir_sac_multi.py \
  --memory-mode reset_each_step \
  --leaking-rate 0.3 \
  --timesteps 200000 \
  --model-path models/reset_reservoir_sac.pt
```

For a paper comparison, use multiple matched seeds and reserve separate seeds
for final evaluation. Tune the leak rate on validation results, not test data.

## Evaluate the trained pair

```bash
.venv/bin/python src/evaluate_reservoir_sac_multi.py \
  --model-path models/memory_ablation/reservoir_sac_persistent_leak_0p3_seed_0.pt \
  --episodes-per-system 20 \
  --seed 2000000 \
  --no-generate-gif \
  --output-path results/persistent_leak_0p3_seed_0.json \
  --plot-path results/persistent_leak_0p3_seed_0.png

.venv/bin/python src/evaluate_reservoir_sac_multi.py \
  --model-path models/memory_ablation/reservoir_sac_reset_each_step_leak_0p3_seed_0.pt \
  --episodes-per-system 20 \
  --seed 2000000 \
  --no-generate-gif \
  --output-path results/reset_each_step_leak_0p3_seed_0.json \
  --plot-path results/reset_each_step_leak_0p3_seed_0.png
```

Use the same evaluation seed for a paired comparison.
