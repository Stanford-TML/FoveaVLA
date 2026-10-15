# Data preparation

## Overview

FoveaVLA is post-trained on the external cameras of [DROID](https://droid-dataset.github.io/). Foveation is supervised with [RoboInter](https://github.com/InternRobotics/RoboInter) annotations: the target box encloses the components of the annotated skill (gripper, object, target) 8 steps ahead. The steps below build this box, a 224×224 foveal crop cut from the original-resolution frame, and metric depth from the ZED stereo pair with Fast-FoundationStereo.

Run these steps from the repository root in the Linux/Python 3.11 environment from [installation](../README.md#installation). Download the source data once and prepare each model in its own output directory.

## 1. Download data

| Input | Source / version | Files |
|---|---|---|
| DROID RLDS | `gs://gresearch/robotics/droid/1.0.1` | Dataset metadata and training TFRecord shards |
| Raw DROID | `gs://gresearch/robotics/droid_raw/1.0.1` | `trajectory.h5`, `metadata_*.json`, external-camera MP4s and SVOs |
| RoboInter | [InternRobotics/RoboInter-Data](https://huggingface.co/datasets/InternRobotics/RoboInter-Data/tree/0208b79c34eca4d214cdb5d07f5ae0f7cc634340) | `RoboInter_Data_Qsheet.json` and `Annotation_pure/annotations/data.mdb` |
| Non-idle filter | OpenPI DROID filter, downloaded below | `droid_sample_ranges_v1_0_1.json` |
| ZED calibration | [StereoLabs factory calibration](https://support.stereolabs.com/articles/3095519603-what-is-the-calibration-file) | `SN<exterior_serial>.conf` for each camera, downloaded in step 4 |

Install [gsutil](https://cloud.google.com/storage/docs/gsutil_install), then download the public data:

```bash
mkdir -p data/rlds/droid/1.0.1 data/raw data/RoboInter-Data

gsutil -m rsync -r gs://gresearch/robotics/droid/1.0.1 data/rlds/droid/1.0.1
# Download only episode metadata and trajectory.h5 first.
gsutil -m rsync -r -x '^(?!.*(?:^|/)(?:trajectory\.h5|metadata[^/]*\.json)$).*' \
  gs://gresearch/robotics/droid_raw/1.0.1 data/raw
# Metadata identifies each episode's two external cameras; download their MP4/SVO pairs.
uv run --no-sync scripts/download_droid_exterior.py --raw-droid-root=data/raw --workers=8

uv run --no-sync huggingface-cli download InternRobotics/RoboInter-Data \
  --repo-type dataset --revision 0208b79c34eca4d214cdb5d07f5ae0f7cc634340 \
  --include 'RoboInter_Data_Qsheet.json' 'Annotation_pure/annotations/*' \
  --local-dir data/RoboInter-Data

curl -fL --retry 3 \
  https://storage.googleapis.com/openpi-assets/droid/droid_sample_ranges_v1_0_1.json \
  -o data/droid_sample_ranges.json
printf '%s\n' '5046049ab62a2df2f802df89cf0888b720f852ce2557849417d40899c9a38bc8  data/droid_sample_ranges.json' \
  | sha256sum --check
```

Allocate several TB for the source data and additional space for crops and depth. Single and Dual share the downloaded files.

Set `--raw-droid-root` to the directory containing the lab folders, such as `data/raw`:

```text
data/raw/TRI/success/2024-01-29/Mon_Jan_29_13:38:42_2024/
  trajectory.h5
  metadata_TRI+52ca9b6a+2024-01-29-13h-38m-42s.json
  recordings/MP4/28451778.mp4
  recordings/SVO/28451778.svo
  recordings/MP4/28813166.mp4
  recordings/SVO/28813166.svo
```

## 2. Select complete downloaded episodes

Choose the model and select its downloaded episodes. Single uses external view 1; Dual uses both external views. Run steps 2–5 for each model.

```bash
MODEL=foveavla_single  # Or foveavla_dual.
PREPARED_DIR="data/processed/${MODEL}"
uv run --no-sync python - "$MODEL" <<'PY'
from pathlib import Path
import sys
from droid_foveation.sources import EpisodeResolver, iter_qsheet

views = {"exterior_image_1_left"}
if sys.argv[1] == "foveavla_dual":
    views.add("exterior_image_2_left")
resolver = EpisodeResolver("data/raw")
episodes = {}
for key, entry in iter_qsheet("data/RoboInter-Data/RoboInter_Data_Qsheet.json"):
    view = str(entry.get("view", ""))
    if view not in views:
        continue
    resolved = resolver.resolve(entry)
    if resolved.status != "ok" or (resolved.video_width, resolved.video_height) != (1280, 720):
        continue
    svo = Path(resolved.local_episode_path) / "recordings/SVO" / f"{resolved.camera_serial}.svo"
    if svo.is_file():
        episodes.setdefault(str(entry["ori_path"]), {})[view] = key
keys = sorted(key for episode in episodes.values() if views <= episode.keys() for key in episode.values())
if not keys:
    raise SystemExit("No complete episodes found; check raw directory layout and downloads")
path = Path(f"data/keys_{sys.argv[1]}.txt")
path.write_text("\n".join(keys) + "\n")
print(f"Selected {len(keys)} episode views: {path}")
PY
```

## 3. Build interaction boxes and crops

Generate current and t+8 interaction boxes and 224×224 foveal crops in a new `$PREPARED_DIR` output directory:

```bash
uv run --no-sync scripts/prepare_data.py \
  --model="$MODEL" \
  --qsheet=data/RoboInter-Data/RoboInter_Data_Qsheet.json \
  --annotation-lmdb=data/RoboInter-Data/Annotation_pure/annotations \
  --raw-droid-root=data/raw --keys-file="data/keys_${MODEL}.txt" \
  --rlds-filter=data/droid_sample_ranges.json \
  --output-dir="$PREPARED_DIR" --workers=8
```

Annotations are saved under `$PREPARED_DIR/annotations`. Crops, manifests, and the training-frame filter are saved under `$PREPARED_DIR/foveal`.

## 4. Generate metric-depth sidecars

On the NVIDIA GPU machine, install **ZED SDK 5.1.2** and its Python 3.11 API using the SDK installer and `get_python_api.py`. Download the selected cameras' calibration files:

```bash
uv run --no-sync python - "${PREPARED_DIR}/foveal" <<'PY'
import configparser
from pathlib import Path
import urllib.request
import sys
import pyarrow.parquet as pq

settings = Path("data/zed_settings")
settings.mkdir(parents=True, exist_ok=True)
serials = set(pq.read_table(Path(sys.argv[1]) / "episodes.parquet", columns=["camera_serial"])["camera_serial"].to_pylist())
for serial in sorted(serials):
    path = settings / f"SN{serial}.conf"
    if not path.is_file():
        urllib.request.urlretrieve(f"https://calib.stereolabs.com/?SN={serial}", path)
    calibration = configparser.ConfigParser()
    calibration.read(path)
    baseline = calibration.getfloat("STEREO", "Baseline") / 1000.0
    if not 0.05 <= baseline <= 0.2:
        raise ValueError(f"Invalid metric baseline: {path}")
print(f"Calibration ready for {len(serials)} external cameras")
PY
```

Generate full-view and foveal depth from a clean Git checkout:

```bash
scripts/setup_fast_foundation_depth.sh
uv run --no-sync scripts/prepare_depth.py \
  --model="$MODEL" --foveal-repack-dir="${PREPARED_DIR}/foveal" --output-dir="${PREPARED_DIR}/depth" \
  --zed-settings=data/zed_settings
```

Depth maps and the combined training-frame filter are saved under `$PREPARED_DIR/depth`. Adjust `--batch-size` for GPU memory. Rerun the same command to resume depth preparation.

## 5. Train

```text
data/
  rlds/droid/1.0.1/             Public RLDS dataset
  raw/                         Shared external-camera MP4/SVO, metadata, and trajectory.h5
  RoboInter-Data/               Shared Qsheet and annotations
  zed_settings/                Shared calibration files
  processed/
    foveavla_single/            Single processing outputs
      annotations/
      foveal/
      depth/left/
      depth/paired_rlds_sample_ranges.json
    foveavla_dual/              Dual processing outputs
      annotations/
      foveal/
      depth/left/
      depth/right/
      depth/paired_rlds_sample_ranges.json
```

Training uses the π₀.₅ base checkpoint's DROID normalization statistics. Start training with your [W&B API key](https://wandb.ai/authorize):

```bash
uv sync --locked --group rlds
export FOVEAVLA_DATA_ROOT="$(pwd)/${PREPARED_DIR}"
export FOVEAVLA_RLDS_DIR="$(pwd)/data/rlds"
export WANDB_API_KEY="YOUR_WANDB_API_KEY"
uv run --no-sync scripts/train.py "$MODEL" \
  --exp-name=public_data_run --batch-size=256 --fsdp-devices=8
```

See [training](training.md) for GPU settings and checkpoints, and [deployment](deployment.md) to run the robot.
