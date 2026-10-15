# Real-robot deployment

Use a Franka/DROID robot, one external ZED camera for Single or two for Dual, and an NVIDIA GPU server. Run commands from the repository root.

## 1. Start the GPU server

Use the Python 3.11 environment from [installation](../README.md#installation).

```bash
uv sync --locked
scripts/setup_fast_foundation_depth.sh
MODEL=foveavla_single  # Or foveavla_dual.
uv run scripts/download_checkpoint.py --model="$MODEL"
XLA_PYTHON_CLIENT_MEM_FRACTION=0.70 uv run scripts/serve.py --model="$MODEL"
```

The checkpoints are downloaded to:

| Model | Default checkpoint directory |
|---|---|
| FoveaVLA-Single | `checkpoints/foveavla_single` |
| FoveaVLA-Dual | `checkpoints/foveavla_dual` |

The server listens on port 8000. Set `--port` to use another port and adjust `XLA_PYTHON_CLIENT_MEM_FRACTION` for your GPU.

To use trained weights, add `--checkpoint=/path/to/checkpoint`. See [training](training.md) for checkpoint paths.

## 2. Set up the robot client

The client runs in its own Python 3.10 environment on the camera workstation, separate from the server's Python 3.11 environment. The workstation can be the GPU server itself.

Follow DROID's [host setup guide](https://droid-dataset.github.io/droid/software-setup/host-installation.html) completely for the Franka, NUC, and workstation, including setting the IPs, robot type, and serial number in `droid/misc/parameters.py`. On the workstation, make two changes to that guide: clone DROID at the tested commit into `.runtime/droid`, and create the environment with Python 3.10 instead of 3.7:

```bash
git clone https://github.com/droid-dataset/droid.git .runtime/droid
git -C .runtime/droid checkout 33ae6a67274f36d2e29525b86f23a56616ef43a7
git -C .runtime/droid submodule update --init --recursive
conda create -n foveavla_droid python=3.10
conda activate foveavla_droid
```

In this environment, install [ZED SDK **5.1.2**](https://www.stereolabs.com/developers/release/5.1) and its [Python API](https://www.stereolabs.com/docs/development/python/install), and finish the guide's remaining `pip install` steps. Then install the FoveaVLA client:

```bash
scripts/setup_droid_client_env.sh
```

The script installs the client, pins `numpy<2` for DROID's OpenCV, and patches the DROID checkout in `.runtime/droid` (set `DROID_ROOT` to use another path). It refuses a checkout at a different commit or one that the active environment does not import.

## 3. Run the robot

Replace the camera serials, instruction, and `SERVER_IP` (the GPU server's address). For Dual, assign the two cameras in physical-left then physical-right order.

```bash
# FoveaVLA-Single
LD_LIBRARY_PATH=/usr/local/zed/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} \
  python examples/droid/main.py \
  --left-camera-id=YOUR_LEFT_ZED_SERIAL \
  --instruction="pick up the object" \
  --remote-host=SERVER_IP --remote-port=8000

# FoveaVLA-Dual
LD_LIBRARY_PATH=/usr/local/zed/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} \
  python examples/droid/main.py \
  --left-camera-id=YOUR_LEFT_ZED_SERIAL --right-camera-id=YOUR_RIGHT_ZED_SERIAL \
  --instruction="pick up the object" \
  --remote-host=SERVER_IP --remote-port=8000
```

The client resets the robot and executes 8 actions per model query. Press Ctrl+C to stop.

Common client options:

| Option | Description |
|---|---|
| `--left-camera-id` | Physical-left external ZED serial; required for both models |
| `--right-camera-id` | Physical-right external ZED serial; required for Dual |
| `--instruction` | Task instruction |
| `--remote-host` | GPU policy server address |
| `--remote-port` | Policy server port; default 8000 |
| `--max-timesteps` | Maximum rollout length; default 900 |
| `--gripper-timeout-s` | Abort if the gripper is not open after reset, or does not follow an open/close command for this long; default 2.0 |
| `--debug` | Save a debug rollout, also when stopped with Ctrl+C or on error: `rollout.mp4` (one frame per policy query with camera views, crop box, foveal crops, full and foveal metric depth as the policy sees it, gripper state and commands, timing), `crops/` (each foveal crop sent to the policy), and `queries.jsonl`; off by default |
| `--debug-dir` | Where `--debug` writes one timestamped folder per rollout; default `outputs/rollouts` |
