#!/usr/bin/env python3
"""Validate sparse SVO RGB snapshots against their DROID MP4 frames."""

from __future__ import annotations

import argparse
import functools
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np

from foveavla.shared import stereo_alignment
from foveavla.shared import stereo_geometry_manifest
from foveavla.training import droid_stereo_geometry_store

_SNAPSHOT_FORMAT_VERSION = "droid-stereo-alignment-snapshot-v1"
_REPORT_FORMAT_VERSION = "droid-stereo-alignment-extraction-v1"
_VALIDATION_FORMAT_VERSION = "droid-stereo-alignment-validation-v2"
_MIN_DECISIVE_OFFSET_ADVANTAGE = 1.0
_MAX_EPISODES_PER_WORKER = 128
_NO_USABLE_VIDEO_BOUNDS_PREFIX = "No alignment frames are inside usable video bounds:"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--frames-per-episode", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frames_per_episode < 1:
        raise ValueError("frames-per-episode must be positive")
    if args.workers < 1:
        raise ValueError("workers must be positive")
    started = time.monotonic()
    manifest_index = json.loads((args.manifest_root / "index.json").read_text())
    if manifest_index.get("index_format_version") != "droid-stereo-geometry-production-manifests-v2":
        raise ValueError("Production manifests do not use the required v2 index format")
    shard_count = int(manifest_index["shards"])
    manifest_fingerprint = droid_stereo_geometry_store.production_manifest_fingerprint(args.manifest_root)
    extractor_path = Path(__file__).with_name("extract_droid_stereo_alignment_audit.py")
    extractor_sha256 = stereo_geometry_manifest.file_sha256(extractor_path)
    alignment_source_sha256 = stereo_geometry_manifest.file_sha256(stereo_alignment.__file__)

    expected_keys: set[str] = set()
    alignment_failures: list[dict[str, object]] = []
    alignment_ambiguities: list[dict[str, object]] = []
    validation_errors: list[str] = []
    best_offset_distribution = dict.fromkeys(stereo_alignment.OFFSETS, 0)
    zero_offset_maes = []
    validated_episodes = 0
    validated_frames = 0
    validation_items: list[tuple[Path, str, str, np.ndarray, str]] = []

    for shard in range(shard_count):
        manifest_path = args.manifest_root / "manifests" / f"shard-{shard:05d}.npz"
        try:
            manifest = stereo_geometry_manifest.load_geometry_manifest(manifest_path)
            manifest_sha256 = stereo_geometry_manifest.file_sha256(manifest_path)
            if manifest.shard_index != shard or manifest.shard_count != shard_count:
                raise ValueError(f"manifest identifies itself as shard {manifest.shard_index}/{manifest.shard_count}")
            _validate_extraction_report(
                args.audit_root / "reports" / f"shard-{shard:05d}.json",
                shard,
                shard_count,
                len(manifest.episodes),
                args.frames_per_episode,
                manifest_sha256,
                extractor_sha256,
                alignment_source_sha256,
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            validation_errors.append(f"{manifest_path}: {error}")
            continue

        for episode in manifest.episodes:
            if episode.annotation_key in expected_keys:
                validation_errors.append(f"duplicate annotation key: {episode.annotation_key}")
                continue
            expected_keys.add(episode.annotation_key)
            snapshot_path = args.audit_root / "snapshots" / f"{episode.annotation_key}.npz"
            validation_items.append(
                (
                    snapshot_path,
                    episode.annotation_key,
                    episode.video_path,
                    stereo_alignment.select_internal_frames(episode.frame_idx, args.frames_per_episode),
                    manifest_sha256,
                )
            )

    audit_episode = functools.partial(
        _audit_episode,
        extractor_sha256=extractor_sha256,
        alignment_source_sha256=alignment_source_sha256,
    )
    # OpenCV/FFmpeg retains allocations after VideoCapture.release() on this workload.
    # Recycle spawned workers so validating tens of thousands of videos has bounded memory.
    with multiprocessing.get_context("spawn").Pool(
        processes=args.workers,
        maxtasksperchild=_MAX_EPISODES_PER_WORKER,
    ) as worker_pool:
        outcomes = worker_pool.imap(
            audit_episode,
            validation_items,
            chunksize=1,
        )
        for item, (result, error) in zip(validation_items, outcomes, strict=True):
            snapshot_path, annotation_key, video_path, _, _ = item
            if error is not None:
                if is_unverifiable_video_bounds(error):
                    alignment_failures.append(
                        {
                            "annotation_key": annotation_key,
                            "video_path": video_path,
                            "reason": "no_alignment_frames_in_usable_video_bounds",
                        }
                    )
                    validated_episodes += 1
                    continue
                validation_errors.append(f"{snapshot_path}: {error}")
                continue
            assert result is not None
            validated_episodes += 1
            validated_frames += result.frames_compared
            best_offset_distribution[result.best_offset] += 1
            zero_offset_maes.append(result.mae_by_offset[0])
            classification = classify_alignment(result)
            if classification == "mismatch":
                alignment_failures.append(_alignment_row(annotation_key, video_path, result))
            elif classification == "ambiguous":
                alignment_ambiguities.append(_alignment_row(annotation_key, video_path, result))

    snapshot_dir = args.audit_root / "snapshots"
    actual_keys = {path.stem for path in snapshot_dir.glob("*.npz")}
    missing_keys = sorted(expected_keys - actual_keys)
    extra_keys = sorted(actual_keys - expected_keys)
    partial_paths = sorted(snapshot_dir.glob("*.partial"))
    if missing_keys:
        validation_errors.append(f"missing {len(missing_keys)} snapshots; first: {missing_keys[:10]}")
    if extra_keys:
        validation_errors.append(f"unexpected {len(extra_keys)} snapshots; first: {extra_keys[:10]}")
    if partial_paths:
        validation_errors.append(f"found {len(partial_paths)} partial snapshots; first: {partial_paths[:10]}")

    expected_episode_count = int(manifest_index["episodes"])
    if len(expected_keys) != expected_episode_count:
        validation_errors.append(f"manifest episode count: expected {expected_episode_count}, got {len(expected_keys)}")
    if validated_episodes != expected_episode_count:
        validation_errors.append(
            f"validated episode count: expected {expected_episode_count}, got {validated_episodes}"
        )

    # A decoded, content-mismatched episode is safe to carry as an explicit exclusion.
    # Structural validation errors still invalidate the entire audit.
    complete = not validation_errors
    summary = {
        "complete": complete,
        "episodes": validated_episodes,
        "alignment_frames": validated_frames,
        "alignment_failures": len(alignment_failures),
        "alignment_ambiguities": len(alignment_ambiguities),
        "validation_errors": len(validation_errors),
        "best_offset_distribution": {str(offset): count for offset, count in best_offset_distribution.items()},
        "mean_zero_offset_mae": float(np.mean(zero_offset_maes)) if zero_offset_maes else None,
        "thresholds": {
            "mean_mae": stereo_alignment.MAX_MEAN_MAE,
            "max_frame_mae": stereo_alignment.MAX_FRAME_MAE,
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    validation = {
        "format_version": _VALIDATION_FORMAT_VERSION,
        "manifest_root": str(args.manifest_root),
        "manifest_fingerprint": manifest_fingerprint,
        "annotation_keys_sha256": droid_stereo_geometry_store.annotation_keys_sha256(expected_keys),
        "frames_per_episode": args.frames_per_episode,
        "validator_source_sha256": stereo_geometry_manifest.file_sha256(__file__),
        "extractor_source_sha256": extractor_sha256,
        "alignment_source_sha256": alignment_source_sha256,
        "summary": summary,
        "alignment_failures": alignment_failures,
        "alignment_ambiguities": alignment_ambiguities,
        "validation_errors": validation_errors,
    }
    validation_path = args.audit_root / "validation.json"
    temporary_path = validation_path.with_suffix(".json.partial")
    temporary_path.write_text(json.dumps(validation, indent=2) + "\n")
    temporary_path.replace(validation_path)
    if not complete:
        raise RuntimeError(f"Alignment audit failed with {len(validation_errors)} validation errors")
    (args.audit_root / "_INCOMPLETE").unlink(missing_ok=True)
    (args.audit_root / "_SUCCESS").write_text("complete\n")
    print(json.dumps(summary, indent=2))


def _validate_extraction_report(
    path: Path,
    shard: int,
    shard_count: int,
    episode_count: int,
    frames_per_episode: int,
    manifest_sha256: str,
    extractor_sha256: str,
    alignment_source_sha256: str,
) -> None:
    report = json.loads(path.read_text())
    if (
        report.get("format_version") != _REPORT_FORMAT_VERSION
        or int(report.get("manifest_shard", -1)) != shard
        or int(report.get("manifest_shards", -1)) != shard_count
        or int(report.get("frames_per_episode", -1)) != frames_per_episode
        or report.get("manifest_sha256") != manifest_sha256
        or report.get("extractor_source_sha256") != extractor_sha256
        or report.get("alignment_source_sha256") != alignment_source_sha256
        or int(report.get("summary", {}).get("episodes", -1)) != episode_count
    ):
        raise ValueError(f"Incompatible alignment extraction report: {path}")


def classify_alignment(result: stereo_alignment.AlignmentResult) -> str:
    if (
        result.mae_by_offset[0] > stereo_alignment.MAX_MEAN_MAE
        or result.max_mae_by_offset[0] > stereo_alignment.MAX_FRAME_MAE
    ):
        return "mismatch"
    if result.best_offset == 0:
        return "matched"
    best_mae = result.mae_by_offset[result.best_offset]
    if result.mae_by_offset[0] - best_mae >= _MIN_DECISIVE_OFFSET_ADVANTAGE:
        return "mismatch"
    return "ambiguous"


def is_unverifiable_video_bounds(error: str) -> bool:
    return error.startswith(_NO_USABLE_VIDEO_BOUNDS_PREFIX)


def _alignment_row(
    annotation_key: str,
    video_path: str,
    result: stereo_alignment.AlignmentResult,
) -> dict[str, object]:
    return {
        "annotation_key": annotation_key,
        "video_path": video_path,
        "best_offset": result.best_offset,
        "mae_by_offset": {str(key): value for key, value in result.mae_by_offset.items()},
        "max_mae_by_offset": {str(key): value for key, value in result.max_mae_by_offset.items()},
    }


def _load_snapshot(
    path: Path,
    annotation_key: str,
    expected_frame_indices: np.ndarray,
    manifest_sha256: str,
    extractor_sha256: str,
    alignment_source_sha256: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    with np.load(path, allow_pickle=False) as snapshot:
        if (
            str(snapshot["format_version"].item()) != _SNAPSHOT_FORMAT_VERSION
            or str(snapshot["annotation_key"].item()) != annotation_key
            or str(snapshot["manifest_sha256"].item()) != manifest_sha256
            or str(snapshot["extractor_source_sha256"].item()) != extractor_sha256
            or str(snapshot["alignment_source_sha256"].item()) != alignment_source_sha256
        ):
            raise ValueError(f"Incompatible alignment snapshot: {path}")
        frame_indices = snapshot["frame_idx"].astype(np.int32, copy=True)
        np.testing.assert_array_equal(frame_indices, expected_frame_indices)
        images = snapshot["left_bgr"].copy()
        stride = int(snapshot["stride"])
    return frame_indices, images, stride


def _audit_episode(
    item: tuple[Path, str, str, np.ndarray, str],
    extractor_sha256: str,
    alignment_source_sha256: str,
) -> tuple[stereo_alignment.AlignmentResult | None, str | None]:
    snapshot_path, annotation_key, video_path, expected_frame_indices, manifest_sha256 = item
    try:
        frame_indices, svo_frames, stride = _load_snapshot(
            snapshot_path,
            annotation_key,
            expected_frame_indices,
            manifest_sha256,
            extractor_sha256,
            alignment_source_sha256,
        )
        return (
            stereo_alignment.compare_video_frames(
                video_path,
                svo_frames,
                frame_indices,
                stride=stride,
            ),
            None,
        )
    except (OSError, ValueError, KeyError, AssertionError) as error:
        return None, str(error)


if __name__ == "__main__":
    main()
