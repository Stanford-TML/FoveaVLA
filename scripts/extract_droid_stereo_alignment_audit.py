#!/usr/bin/env python3
"""Extract sparse SVO RGB frames for an independent SVO/MP4 alignment audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from pyzed import sl

from foveavla.shared import stereo_alignment
from foveavla.shared import stereo_geometry_manifest

_FORMAT_VERSION = "droid-stereo-alignment-snapshot-v1"
_REPORT_FORMAT_VERSION = "droid-stereo-alignment-extraction-v1"
_EXPECTED_SDK_VERSION = "5.1.2"
_STRIDE = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--frames-per-episode", type=int, default=3)
    parser.add_argument("--zed-settings", type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frames_per_episode < 1:
        raise ValueError("frames-per-episode must be positive")
    manifest = stereo_geometry_manifest.load_geometry_manifest(args.manifest)
    manifest_sha256 = stereo_geometry_manifest.file_sha256(args.manifest)
    extractor_sha256 = stereo_geometry_manifest.file_sha256(__file__)
    alignment_source_sha256 = stereo_geometry_manifest.file_sha256(stereo_alignment.__file__)
    sdk_version = sl.Camera.get_sdk_version()
    if sdk_version != _EXPECTED_SDK_VERSION:
        raise RuntimeError(f"Expected ZED SDK {_EXPECTED_SDK_VERSION}, got {sdk_version}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    rows = []
    for episode in manifest.episodes:
        frame_indices = stereo_alignment.select_internal_frames(episode.frame_idx, args.frames_per_episode)
        output_path = args.output_dir / f"{episode.annotation_key}.npz"
        if output_path.exists():
            if not args.resume:
                raise FileExistsError(f"Alignment snapshot already exists: {output_path}")
            _validate_snapshot(
                output_path,
                episode.annotation_key,
                frame_indices,
                manifest_sha256,
                extractor_sha256,
                alignment_source_sha256,
            )
            rows.append({"annotation_key": episode.annotation_key, "status": "skipped_valid"})
            continue

        camera = _open_svo(episode.svo_path, args.zed_settings)
        try:
            runtime = sl.RuntimeParameters()
            left = sl.Mat()
            frames = []
            for frame_idx in frame_indices:
                _require_success(
                    camera.set_svo_position(int(frame_idx)),
                    f"seek for {episode.annotation_key} frame {frame_idx}",
                )
                _require_success(camera.grab(runtime), f"grab for {episode.annotation_key} frame {frame_idx}")
                _require_success(
                    camera.retrieve_image(left, sl.VIEW.LEFT),
                    f"left image retrieval for {episode.annotation_key} frame {frame_idx}",
                )
                frames.append(np.asarray(left.get_data())[::_STRIDE, ::_STRIDE, :3].copy())
        finally:
            camera.close()

        temporary_path = output_path.with_suffix(".npz.partial")
        with temporary_path.open("wb") as output_file:
            np.savez_compressed(
                output_file,
                format_version=np.asarray(_FORMAT_VERSION),
                annotation_key=np.asarray(episode.annotation_key),
                frame_idx=frame_indices,
                left_bgr=np.stack(frames),
                stride=np.asarray(_STRIDE, dtype=np.int32),
                manifest_sha256=np.asarray(manifest_sha256),
                extractor_source_sha256=np.asarray(extractor_sha256),
                alignment_source_sha256=np.asarray(alignment_source_sha256),
            )
        temporary_path.replace(output_path)
        rows.append({"annotation_key": episode.annotation_key, "status": "written"})

    report = {
        "format_version": _REPORT_FORMAT_VERSION,
        "manifest": str(args.manifest),
        "manifest_sha256": manifest_sha256,
        "manifest_shard": manifest.shard_index,
        "manifest_shards": manifest.shard_count,
        "sdk_version": sdk_version,
        "frames_per_episode": args.frames_per_episode,
        "stride": _STRIDE,
        "extractor_source_sha256": extractor_sha256,
        "alignment_source_sha256": alignment_source_sha256,
        "summary": {
            "episodes": len(rows),
            "written": sum(row["status"] == "written" for row in rows),
            "skipped_valid": sum(row["status"] == "skipped_valid" for row in rows),
            "elapsed_seconds": time.monotonic() - started,
        },
    }
    temporary_report = args.report.with_suffix(".json.partial")
    temporary_report.write_text(json.dumps(report, indent=2) + "\n")
    temporary_report.replace(args.report)
    print(json.dumps(report["summary"], indent=2))


def _open_svo(path: str, settings_dir: Path | None = None) -> sl.Camera:
    input_type = sl.InputType()
    input_type.set_from_svo_file(path)
    init = sl.InitParameters(input_t=input_type, svo_real_time_mode=False)
    init.depth_mode = sl.DEPTH_MODE.NONE
    if settings_dir is not None:
        init.optional_settings_path = str(settings_dir.resolve())
    camera = sl.Camera()
    status = camera.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Could not open {path}: {status}")
    return camera


def _require_success(status: object, operation: str) -> None:
    if status is not None and status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"ZED {operation} failed: {status}")


def _validate_snapshot(
    path: Path,
    annotation_key: str,
    frame_indices: np.ndarray,
    manifest_sha256: str,
    extractor_sha256: str,
    alignment_source_sha256: str,
) -> None:
    with np.load(path, allow_pickle=False) as snapshot:
        if (
            str(snapshot["format_version"].item()) != _FORMAT_VERSION
            or str(snapshot["annotation_key"].item()) != annotation_key
            or str(snapshot["manifest_sha256"].item()) != manifest_sha256
            or str(snapshot["extractor_source_sha256"].item()) != extractor_sha256
            or str(snapshot["alignment_source_sha256"].item()) != alignment_source_sha256
            or int(snapshot["stride"]) != _STRIDE
        ):
            raise ValueError(f"Incompatible alignment snapshot: {path}")
        np.testing.assert_array_equal(snapshot["frame_idx"], frame_indices)
        images = snapshot["left_bgr"]
        if images.ndim != 4 or images.shape[0] != len(frame_indices) or images.shape[-1] != 3:
            raise ValueError(f"Invalid alignment snapshot image shape: {path}: {images.shape}")


if __name__ == "__main__":
    main()
