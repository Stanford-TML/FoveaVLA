#!/usr/bin/env python3
"""Generate one resumable LMDB shard of left-aligned Fast-FoundationStereo depth."""

from __future__ import annotations

import argparse
import configparser
import hashlib
import json
from pathlib import Path
import struct
import sys
import time
import types
import zlib

import lmdb
import numpy as np
from pyzed import sl
import torch

from foveavla.shared import fast_foundation_depth

# Fast-FoundationStereo imports Open3D only for optional visualization.
sys.modules["open3d"] = types.ModuleType("open3d")

from core.utils.utils import InputPadder  # noqa: E402
from Utils import AMP_DTYPE  # noqa: E402

PLAN_FORMAT = "droid-fast-foundation-plan-v1"
REPORT_FORMAT = "droid-fast-foundation-worker-v1"
VALUE_HEADER = struct.Struct("<b3x4i")
OUTPUT_SIZE = 224
RAW_VALUE_BYTES = VALUE_HEADER.size + 2 * (OUTPUT_SIZE * OUTPUT_SIZE * 2 + OUTPUT_SIZE * OUTPUT_SIZE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--zed-settings", type=Path, required=True)
    parser.add_argument("--seek-gap", type=int, default=32)
    parser.add_argument("--max-episodes", type=int)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def require_success(status: object, operation: str) -> None:
    if status is not None and status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"ZED {operation} failed: {status}")


def baseline_m(settings_dir: Path, serial: str) -> float:
    parser = configparser.ConfigParser()
    path = settings_dir / f"SN{serial}.conf"
    if not parser.read(path):
        raise FileNotFoundError(path)
    baseline = parser.getfloat("STEREO", "Baseline") / 1000.0
    if not 0.05 <= baseline <= 0.2:
        raise ValueError(f"Implausible baseline in {path}: {baseline}")
    return baseline


def open_svo(path: str, settings_dir: Path) -> sl.Camera:
    source = sl.InputType()
    source.set_from_svo_file(path)
    init = sl.InitParameters(input_t=source, svo_real_time_mode=False)
    init.depth_mode = sl.DEPTH_MODE.NONE
    init.optional_settings_path = str(settings_dir.resolve())
    camera = sl.Camera()
    require_success(camera.open(init), f"open {path}")
    return camera


def encode_depth_pair(
    depth_full: np.ndarray,
    depth_crop: np.ndarray,
    valid_full: np.ndarray,
    valid_crop: np.ndarray,
    crop_mode: int,
    crop_bbox: np.ndarray,
) -> bytes:
    raw = VALUE_HEADER.pack(int(crop_mode), *(int(value) for value in crop_bbox))
    raw += np.stack((depth_full, depth_crop)).astype("<f2", copy=False).tobytes(order="C")
    raw += (np.stack((valid_full, valid_crop)).astype(np.uint8) * 255).tobytes(order="C")
    if len(raw) != RAW_VALUE_BYTES:
        raise ValueError(f"Invalid encoded value size: {len(raw)}")
    return zlib.compress(raw, level=1)


def load_model(plan: dict[str, object]) -> torch.nn.Module:
    checkpoint = Path(str(plan["checkpoint"]))
    if sha256_file(checkpoint) != plan["checkpoint_sha256"]:
        raise ValueError("Checkpoint hash does not match the plan")
    model = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.args.valid_iters = int(plan["valid_iters"])
    model.args.max_disp = int(plan["max_disparity"])
    if "normalize" not in model.args:
        model.args.normalize = True
    return model.cuda().eval()


def infer_disparity(
    model: torch.nn.Module,
    left_images: list[np.ndarray],
    right_images: list[np.ndarray],
    plan: dict[str, object],
) -> tuple[np.ndarray, float]:
    count = len(left_images)
    batch_size = int(plan["batch_size"])
    if not 0 < count <= batch_size:
        raise ValueError(f"Invalid inference batch size: {count}")
    if count < batch_size:
        left_images.extend([left_images[-1]] * (batch_size - count))
        right_images.extend([right_images[-1]] * (batch_size - count))
    left = torch.as_tensor(np.stack(left_images), device="cuda").float().permute(0, 3, 1, 2)
    right = torch.as_tensor(np.stack(right_images), device="cuda").float().permute(0, 3, 1, 2)
    padder = InputPadder(left.shape, divis_by=32, force_square=False)
    left, right = padder.pad(left, right)
    torch.cuda.synchronize()
    started = time.monotonic()
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=AMP_DTYPE):
        disparity = model.forward(
            left,
            right,
            iters=int(plan["valid_iters"]),
            test_mode=True,
            low_memory=bool(plan["low_memory"]),
            optimize_build_volume="pytorch1",
        )
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    output = padder.unpad(disparity.float())[:count].cpu().numpy()[:, 0]
    return output, elapsed


def encode_batch(
    disparities: np.ndarray,
    metadata: list[tuple[int, int, np.ndarray]],
    *,
    source_width: int,
    source_height: int,
    input_width: int,
    fx: float,
    baseline: float,
    depth_range: tuple[float, float],
) -> tuple[list[tuple[bytes, bytes]], int, int, int]:
    encoded = []
    full_valid_pixels = 0
    crop_valid_pixels = 0
    compressed_bytes = 0
    for disparity, (frame_idx, crop_mode, crop_bbox) in zip(disparities, metadata, strict=True):
        full_depth, crop_depth, full_valid, crop_valid = fast_foundation_depth.disparity_to_policy_depth(
            disparity,
            fx=fx,
            baseline_m=baseline,
            input_width=input_width,
            source_width=source_width,
            source_height=source_height,
            crop_mode=crop_mode,
            crop_bbox=tuple(int(value) for value in crop_bbox),
            output_size=OUTPUT_SIZE,
            depth_range_m=depth_range,
        )
        value = encode_depth_pair(
            full_depth.astype(np.float16),
            crop_depth.astype(np.float16),
            full_valid,
            crop_valid,
            crop_mode,
            crop_bbox,
        )
        encoded.append((str(frame_idx).encode(), value))
        full_valid_pixels += int(full_valid.sum())
        crop_valid_pixels += int(crop_valid.sum())
        compressed_bytes += len(value)
    return encoded, full_valid_pixels, crop_valid_pixels, compressed_bytes


def process_episode(
    model: torch.nn.Module,
    annotation_key: str,
    serial: str,
    svo_path: str,
    frame_indices: np.ndarray,
    crop_modes: np.ndarray,
    crop_bboxes: np.ndarray,
    plan: dict[str, object],
    zed_settings: Path,
    seek_gap: int,
) -> tuple[list[tuple[bytes, bytes]], dict[str, int | float]]:
    source_width, source_height = (int(value) for value in plan["source_resolution"])
    input_width, input_height = (int(value) for value in plan["input_resolution"])
    depth_range = tuple(float(value) for value in plan["depth_range_m"])
    camera = open_svo(svo_path, zed_settings)
    left_image = sl.Mat()
    right_image = sl.Mat()
    runtime = sl.RuntimeParameters()
    resolution = sl.Resolution(input_width, input_height)
    values: list[tuple[bytes, bytes]] = []
    full_valid_pixels = 0
    crop_valid_pixels = 0
    compressed_bytes = 0
    inference_seconds = 0.0
    grab_count = 0
    try:
        configuration = camera.get_camera_information().camera_configuration
        if (configuration.resolution.width, configuration.resolution.height) != (source_width, source_height):
            raise ValueError(f"Unexpected SVO resolution for {annotation_key}")
        fx = float(configuration.calibration_parameters.left_cam.fx)
        baseline = baseline_m(zed_settings, serial)
        require_success(camera.set_svo_position(int(frame_indices[0])), f"seek {annotation_key}")
        next_frame = int(frame_indices[0])
        left_batch: list[np.ndarray] = []
        right_batch: list[np.ndarray] = []
        metadata: list[tuple[int, int, np.ndarray]] = []

        def flush() -> None:
            nonlocal full_valid_pixels, crop_valid_pixels, compressed_bytes, inference_seconds
            if not metadata:
                return
            disparities, elapsed = infer_disparity(model, left_batch, right_batch, plan)
            batch_values, batch_full_valid, batch_crop_valid, batch_bytes = encode_batch(
                disparities,
                metadata,
                source_width=source_width,
                source_height=source_height,
                input_width=input_width,
                fx=fx,
                baseline=baseline,
                depth_range=depth_range,
            )
            values.extend(batch_values)
            full_valid_pixels += batch_full_valid
            crop_valid_pixels += batch_crop_valid
            compressed_bytes += batch_bytes
            inference_seconds += elapsed
            left_batch.clear()
            right_batch.clear()
            metadata.clear()

        for frame_value, crop_mode_value, crop_bbox_value in zip(
            frame_indices, crop_modes, crop_bboxes, strict=True
        ):
            frame_idx = int(frame_value)
            gap = frame_idx - next_frame
            if gap < 0:
                raise ValueError(f"Non-monotonic frames in {annotation_key}")
            if gap > seek_gap:
                require_success(camera.set_svo_position(frame_idx), f"seek {annotation_key}:{frame_idx}")
                next_frame = frame_idx
            while next_frame < frame_idx:
                require_success(camera.grab(runtime), f"skip {annotation_key}:{next_frame}")
                next_frame += 1
                grab_count += 1
            require_success(camera.grab(runtime), f"grab {annotation_key}:{frame_idx}")
            next_frame += 1
            grab_count += 1
            require_success(camera.retrieve_image(left_image, sl.VIEW.LEFT, sl.MEM.CPU, resolution), "left retrieval")
            require_success(camera.retrieve_image(right_image, sl.VIEW.RIGHT, sl.MEM.CPU, resolution), "right retrieval")
            left_batch.append(np.asarray(left_image.get_data())[..., :3][:, :, ::-1].copy())
            right_batch.append(np.asarray(right_image.get_data())[..., :3][:, :, ::-1].copy())
            metadata.append((frame_idx, int(crop_mode_value), np.asarray(crop_bbox_value, dtype=np.int32)))
            if len(metadata) == int(plan["batch_size"]):
                flush()
        flush()
    finally:
        camera.close()
    return values, {
        "samples": len(frame_indices),
        "full_valid_pixels": full_valid_pixels,
        "crop_valid_pixels": crop_valid_pixels,
        "compressed_bytes": compressed_bytes,
        "inference_seconds": inference_seconds,
        "grabs": grab_count,
    }


def aggregate_progress(environment: lmdb.Environment, progress: lmdb._Database) -> dict[str, int | float]:
    totals: dict[str, int | float] = {
        "episodes": 0,
        "samples": 0,
        "full_valid_pixels": 0,
        "crop_valid_pixels": 0,
        "compressed_bytes": 0,
        "inference_seconds": 0.0,
        "grabs": 0,
    }
    with environment.begin(db=progress) as transaction:
        for value in transaction.cursor().iternext(keys=False, values=True):
            episode = json.loads(value)
            totals["episodes"] += 1
            for name in totals.keys() - {"episodes"}:
                totals[name] += episode[name]
    return totals


def main() -> None:
    args = parse_args()
    plan = json.loads(args.plan.read_text())
    if plan.get("format_version") != PLAN_FORMAT:
        raise ValueError("Unsupported plan")
    if sha256_file(Path(__file__)) != plan.get("extractor_source_sha256"):
        raise ValueError("Extractor source hash does not match the plan")
    if sha256_file(Path(fast_foundation_depth.__file__)) != plan.get("postprocess_source_sha256"):
        raise ValueError("Shared postprocess source hash does not match the plan")
    worker_count = int(plan["worker_count"])
    if not 0 <= args.worker_index < worker_count:
        raise ValueError("worker-index is out of range")
    if sl.Camera.get_sdk_version() != "5.1.2":
        raise RuntimeError(f"Expected ZED SDK 5.1.2, got {sl.Camera.get_sdk_version()}")
    if not torch.cuda.is_available():
        raise RuntimeError("Depth extraction requires a CUDA GPU")

    output_dir = Path(str(plan["output_dir"]))
    report_path = output_dir / "worker_reports" / f"worker-{args.worker_index:03d}.json"
    success_path = output_dir / "worker_status" / f"worker-{args.worker_index:03d}.success"
    plan_sha256 = sha256_file(args.plan)
    if success_path.is_file():
        report = json.loads(report_path.read_text())
        if report.get("plan_sha256") != plan_sha256:
            raise ValueError("Completed worker belongs to a different plan")
        print(json.dumps(report, indent=2))
        return

    selected = frozenset(json.loads((output_dir / "selected_annotation_keys.json").read_text()))
    expected_episodes = int(plan["worker_episode_counts"][args.worker_index])
    lmdb_path = output_dir / "pointmaps" / f"shard-{args.worker_index:03d}.lmdb"
    environment = lmdb.open(
        str(lmdb_path),
        map_size=int(plan["map_size_bytes"]),
        max_dbs=2,
        subdir=True,
        sync=True,
        metasync=True,
    )
    progress = environment.open_db(b"progress", create=True)
    model = load_model(plan)
    torch.backends.cudnn.benchmark = True
    started = time.monotonic()
    processed_this_run = 0
    seen = set()
    try:
        for manifest_path in sorted(Path(str(plan["manifest_root"])).glob("manifests/shard-*.npz")):
            with np.load(manifest_path, allow_pickle=False) as manifest:
                offsets = manifest["episode_offsets"]
                for episode_index, key_value in enumerate(manifest["annotation_key"]):
                    annotation_key = str(key_value)
                    if annotation_key not in selected:
                        continue
                    if zlib.crc32(annotation_key.encode()) % worker_count != args.worker_index:
                        continue
                    if annotation_key in seen:
                        raise ValueError(f"Duplicate manifest episode: {annotation_key}")
                    seen.add(annotation_key)
                    with environment.begin(db=progress) as transaction:
                        if transaction.get(annotation_key.encode()) is not None:
                            continue
                    if args.max_episodes is not None and processed_this_run >= args.max_episodes:
                        continue
                    start, end = int(offsets[episode_index]), int(offsets[episode_index + 1])
                    values, episode_stats = process_episode(
                        model,
                        annotation_key,
                        str(manifest["camera_serial"][episode_index]),
                        str(manifest["svo_path"][episode_index]),
                        np.asarray(manifest["frame_idx"][start:end], dtype=np.int32),
                        np.asarray(manifest["crop_mode"][start:end], dtype=np.int8),
                        np.asarray(manifest["crop_bbox"][start:end], dtype=np.int32),
                        plan,
                        args.zed_settings,
                        args.seek_gap,
                    )
                    with environment.begin(write=True) as transaction:
                        for frame_key, value in values:
                            sample_key = annotation_key.encode() + b"--" + frame_key
                            if not transaction.put(sample_key, value, overwrite=False):
                                raise ValueError(f"Duplicate sample key: {sample_key.decode()}")
                        transaction.put(
                            annotation_key.encode(),
                            json.dumps(episode_stats, sort_keys=True).encode(),
                            db=progress,
                            overwrite=False,
                        )
                    processed_this_run += 1
                    print(
                        json.dumps(
                            {
                                "annotation_key": annotation_key,
                                "worker_index": args.worker_index,
                                **episode_stats,
                            }
                        ),
                        flush=True,
                    )
        environment.sync()
        totals = aggregate_progress(environment, progress)
        if args.max_episodes is not None:
            print(json.dumps({"partial": True, **totals}, indent=2))
            return
        if len(seen) != expected_episodes or int(totals["episodes"]) != expected_episodes:
            raise ValueError(
                f"Worker episode coverage mismatch: manifest={len(seen)}, completed={totals['episodes']}, "
                f"expected={expected_episodes}"
            )
        # The named progress database itself occupies one entry in LMDB's main database.
        if environment.stat()["entries"] - 1 != int(totals["samples"]):
            raise ValueError("LMDB sample count does not match progress metadata")
        report = {
            "format_version": REPORT_FORMAT,
            "worker_index": args.worker_index,
            "worker_count": worker_count,
            "plan_sha256": plan_sha256,
            "elapsed_seconds_this_run": time.monotonic() - started,
            **totals,
        }
        atomic_json(report_path, report)
        success_path.write_text("complete\n")
        print(json.dumps(report, indent=2))
    finally:
        environment.close()


if __name__ == "__main__":
    main()
