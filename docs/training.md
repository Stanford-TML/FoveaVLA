# Training

Prepare the [data](data.md), then run these steps from the repository root in the Linux/Python 3.11 environment.

## 1. Set up the environment and paths

```bash
uv sync --locked --group rlds
MODEL=foveavla_single  # Or foveavla_dual.
export FOVEAVLA_DATA_ROOT=/path/to/data/processed/$MODEL
export FOVEAVLA_RLDS_DIR=/path/to/data/rlds
```

Set the two data paths to absolute paths for the model's prepared output and the shared RLDS dataset.

## 2. Train

Single and Dual use the π₀.₅ base checkpoint's DROID normalization statistics from `assets/droid/norm_stats.json`. Training downloads the statistics and base weights on the first run and saves the statistics in each training checkpoint.

W&B logging is enabled by default. Set your [W&B API key](https://wandb.ai/authorize) before training:

```bash
export WANDB_API_KEY="YOUR_WANDB_API_KEY"
uv run --no-sync scripts/train.py "$MODEL" \
  --exp-name=my_run --batch-size=256 --fsdp-devices=8
```

Single and Dual use the same defaults:

| Setting | Default |
|---|---|
| GPUs | 8 |
| Global batch size | 256 |
| Training steps | 100,000 |
| Learning rate | 5e-5 |

Set `--batch-size` and `--fsdp-devices` for your GPU setup. Use a batch size divisible by the total GPU count. For one GPU, set `--fsdp-devices=1`.

Checkpoints are saved every 5,000 steps under `checkpoints/<model>/<exp-name>/<step>/`.

## 3. Resume a run

Add `--resume` to the original training command, keeping the experiment name and training settings:

```bash
uv run --no-sync scripts/train.py "$MODEL" --exp-name=my_run --resume
```

## 4. Serve a trained checkpoint

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.70 uv run --no-sync scripts/serve.py \
  --model="$MODEL" --checkpoint="checkpoints/$MODEL/my_run/99999"
```

Follow [deployment](deployment.md) to start the robot client.

## Data and weight paths

Set individual variables to change the default data and weight locations:

| Variable | Default |
|---|---|
| `FOVEAVLA_RLDS_DIR` | `<data root>/rlds` |
| `FOVEAVLA_FOVEAL_DIR` | `<data root>/foveal` |
| `FOVEAVLA_DEPTH_LEFT_DIR` | `<data root>/depth/left` |
| `FOVEAVLA_DEPTH_RIGHT_DIR` | `<data root>/depth/right` (Dual only) |
| `FOVEAVLA_DEPTH_FILTER` | `<data root>/depth/paired_rlds_sample_ranges.json` |
| `FOVEAVLA_BASE_CHECKPOINT` | `gs://openpi-assets/checkpoints/pi05_base` |

Store crop and depth LMDBs on a local filesystem. Set `FOVEAVLA_RLDS_DIR` to a local TFDS directory or GCS path. To use a downloaded base checkpoint, set `FOVEAVLA_BASE_CHECKPOINT` to its directory containing `params/` and `assets/droid/norm_stats.json`.
