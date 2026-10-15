#!/usr/bin/env python3
"""Build an exact RLDS range filter for the camera-1 pointmap sample set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from foveavla.training import droid_pointmap_store
from foveavla.training import droid_stereo_geometry_store


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--geometry-sidecar", type=Path)
    parser.add_argument("--base-filter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def consecutive_ranges(frame_indices: np.ndarray) -> list[list[int]]:
    frames = np.asarray(frame_indices, dtype=np.int64)
    if frames.ndim != 1 or not len(frames):
        raise ValueError("Pointmap frame indices must be a non-empty vector")
    if np.any(np.diff(frames) <= 0):
        raise ValueError("Pointmap frame indices must be strictly increasing")
    boundaries = np.flatnonzero(np.diff(frames) > 1) + 1
    starts = np.concatenate((frames[:1], frames[boundaries]))
    ends = np.concatenate((frames[boundaries - 1] + 1, frames[-1:] + 1))
    return [[int(start), int(end)] for start, end in zip(starts, ends, strict=True)]


def select_frames_in_ranges(frames: np.ndarray, ranges: list[list[int]]) -> np.ndarray:
    range_index = 0
    selected = []
    for frame in frames:
        while range_index < len(ranges) and frame >= ranges[range_index][1]:
            range_index += 1
        if range_index < len(ranges) and ranges[range_index][0] <= frame < ranges[range_index][1]:
            selected.append(frame)
    return np.asarray(selected, dtype=np.int64)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    manifest_index_path = args.manifest_root / "index.json"
    manifest_index = json.loads(manifest_index_path.read_text())
    pointmap_sidecar = args.episodes.parent
    pointmap_index_path = pointmap_sidecar / "index.json"
    pointmap_summary_path = pointmap_sidecar / "summary.json"
    pointmap_index = json.loads(pointmap_index_path.read_text())
    pointmap_summary = json.loads(pointmap_summary_path.read_text())
    if pointmap_index.get("format_version") != droid_pointmap_store.FORMAT_VERSION:
        raise ValueError("--episodes must belong to a completed camera pointmap sidecar")
    episode_table = pq.read_table(args.episodes, columns=["annotation_key", "ori_path"])
    annotation_to_path = {
        str(annotation_key): str(ori_path)
        for annotation_key, ori_path in zip(
            episode_table["annotation_key"].to_pylist(),
            episode_table["ori_path"].to_pylist(),
            strict=True,
        )
    }
    annotation_keys_sha256 = droid_stereo_geometry_store.annotation_keys_sha256(annotation_to_path)
    pointmap_source = pointmap_index.get("source", {})
    if len(annotation_to_path) != int(pointmap_summary["episodes"]) or annotation_keys_sha256 != pointmap_source.get(
        "selected_annotation_keys_sha256"
    ):
        raise ValueError("Pointmap episode table does not match its sidecar index and summary")
    base_filter = json.loads(args.base_filter.read_text())
    excluded_annotation_keys = frozenset()
    if args.geometry_sidecar is not None:
        geometry_index = json.loads((args.geometry_sidecar / "index.json").read_text())
        alignment_audit = geometry_index.get("source", {}).get("alignment_audit", {})
        excluded_annotation_keys = frozenset(alignment_audit.get("excluded_annotation_keys", ())) | frozenset(
            alignment_audit.get("ambiguous_annotation_keys", ())
        )
    rlds_key_by_path = {}
    for rlds_key in base_filter:
        _, separator, episode_path = rlds_key.rpartition("--")
        if not separator or episode_path in rlds_key_by_path:
            raise ValueError(f"Invalid or duplicate RLDS filter episode path: {rlds_key}")
        rlds_key_by_path[episode_path] = rlds_key

    result = {}
    sample_count = 0
    episodes_absent_from_base_filter = 0
    episodes_without_retained_frames = 0
    episodes_excluded_by_alignment_audit = 0
    episodes_absent_from_pointmap_sidecar = 0
    for manifest_path in sorted((args.manifest_root / "manifests").glob("shard-*.npz")):
        with np.load(manifest_path, allow_pickle=False) as manifest:
            offsets = manifest["episode_offsets"]
            for episode_index, annotation_key_value in enumerate(manifest["annotation_key"]):
                annotation_key = str(annotation_key_value)
                if annotation_key in excluded_annotation_keys:
                    episodes_excluded_by_alignment_audit += 1
                    continue
                episode_path = annotation_to_path.get(annotation_key)
                if episode_path is None:
                    episodes_absent_from_pointmap_sidecar += 1
                    continue
                rlds_key = rlds_key_by_path.get(episode_path)
                if rlds_key is None:
                    episodes_absent_from_base_filter += 1
                    continue
                start, end = int(offsets[episode_index]), int(offsets[episode_index + 1])
                frames = np.asarray(manifest["frame_idx"][start:end], dtype=np.int64)
                frames = select_frames_in_ranges(frames, base_filter[rlds_key])
                if not len(frames):
                    episodes_without_retained_frames += 1
                    continue
                if rlds_key in result:
                    raise ValueError(f"Duplicate pointmap RLDS episode key: {rlds_key}")
                result[rlds_key] = consecutive_ranges(frames)
                sample_count += len(frames)

    if not result or sample_count > int(manifest_index["samples"]):
        raise ValueError("Pointmap RLDS filter is empty or exceeds the production manifest")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_bytes = (json.dumps(result, separators=(",", ":")) + "\n").encode()
    if args.output.exists():
        if args.output.read_bytes() != output_bytes:
            raise FileExistsError(f"Output exists with different content: {args.output}")
    else:
        args.output.write_bytes(output_bytes)
    geometry_index_path = args.geometry_sidecar / "index.json" if args.geometry_sidecar is not None else None
    summary = {
        "episodes": len(result),
        "samples": sample_count,
        "manifest_episodes": int(manifest_index["episodes"]),
        "manifest_samples": int(manifest_index["samples"]),
        "retained_sample_fraction": sample_count / int(manifest_index["samples"]),
        "episodes_absent_from_base_filter": episodes_absent_from_base_filter,
        "episodes_without_retained_frames": episodes_without_retained_frames,
        "episodes_excluded_by_alignment_audit": episodes_excluded_by_alignment_audit,
        "episodes_absent_from_pointmap_sidecar": episodes_absent_from_pointmap_sidecar,
        "ranges": sum(len(ranges) for ranges in result.values()),
        "output": str(args.output),
        "bytes": args.output.stat().st_size,
        "filter_sha256": hashlib.sha256(output_bytes).hexdigest(),
        "builder_source_sha256": _sha256_file(Path(__file__)),
        "base_filter": str(args.base_filter.resolve()),
        "base_filter_sha256": _sha256_file(args.base_filter),
        "manifest_root": str(args.manifest_root.resolve()),
        "manifest_index_sha256": _sha256_file(manifest_index_path),
        "pointmap_sidecar": str(pointmap_sidecar.resolve()),
        "pointmap_index_sha256": _sha256_file(pointmap_index_path),
        "pointmap_summary_sha256": _sha256_file(pointmap_summary_path),
        "pointmap_episodes_parquet_sha256": _sha256_file(args.episodes),
        "pointmap_selected_annotation_keys_sha256": annotation_keys_sha256,
        "pointmap_episodes": int(pointmap_summary["episodes"]),
        "pointmap_samples": int(pointmap_summary["samples"]),
        "geometry_sidecar": str(args.geometry_sidecar.resolve()) if args.geometry_sidecar is not None else None,
        "geometry_index_sha256": _sha256_file(geometry_index_path) if geometry_index_path is not None else None,
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
