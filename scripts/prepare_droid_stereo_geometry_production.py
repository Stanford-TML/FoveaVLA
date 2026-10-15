#!/usr/bin/env python3
"""Build compact, load-balanced manifests for production DROID geometry extraction."""

from __future__ import annotations

import argparse
import heapq
import itertools
import json
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from foveavla.shared import stereo_geometry_manifest

_EPISODE_OVERHEAD_SECONDS = 1.225
_SECONDS_PER_SAMPLE = 0.01844


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--foveal-repack-dir", "--roi-repack-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=256)
    parser.add_argument(
        "--exterior-view",
        choices=stereo_geometry_manifest.EXTERIOR_VIEWS,
        default=stereo_geometry_manifest.EXTERIOR_VIEW,
    )
    parser.add_argument(
        "--exclusions",
        type=Path,
        default=Path(__file__).with_name("droid_stereo_geometry_exclusions.json"),
    )
    return parser.parse_args()


def load_exclusions(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if payload.get("format_version") != "droid-stereo-geometry-exclusions-v1":
        raise ValueError(f"Unsupported exclusion file: {path}")
    episodes = payload.get("episodes")
    rules = payload.get("rules")
    if not isinstance(episodes, dict) or not all(
        isinstance(key, str) and isinstance(reason, str) and reason for key, reason in episodes.items()
    ):
        raise ValueError("exclusions.episodes must map annotation keys to nonempty reasons")
    if not isinstance(rules, list):
        raise ValueError("exclusions.rules must be a list")
    accepted_rule_keys = {"camera_serial", "local_episode_path_contains", "reason"}
    for rule in rules:
        if (
            not isinstance(rule, dict)
            or set(rule) - accepted_rule_keys
            or not isinstance(rule.get("reason"), str)
            or not rule["reason"]
            or not any(key in rule for key in ("camera_serial", "local_episode_path_contains"))
        ):
            raise ValueError(f"Invalid exclusion rule: {rule}")
    return payload


def exclusion_reason(row: dict[str, object], exclusions: dict[str, object]) -> str | None:
    episodes = exclusions["episodes"]
    assert isinstance(episodes, dict)
    explicit_reason = episodes.get(str(row["annotation_key"]))
    if explicit_reason is not None:
        return str(explicit_reason)
    rules = exclusions["rules"]
    assert isinstance(rules, list)
    for rule in rules:
        assert isinstance(rule, dict)
        if "camera_serial" in rule and str(row["camera_serial"]) != str(rule["camera_serial"]):
            continue
        if "local_episode_path_contains" in rule and str(rule["local_episode_path_contains"]) not in str(
            row["local_episode_path"]
        ):
            continue
        return str(rule["reason"])
    return None


def estimated_episode_seconds(row: dict[str, object]) -> float:
    return _EPISODE_OVERHEAD_SECONDS + _SECONDS_PER_SAMPLE * int(row["manifest_samples"])


def assign_episode_shards(
    episode_rows: list[dict[str, object]],
    shard_count: int,
) -> tuple[list[list[dict[str, object]]], list[float]]:
    if shard_count < 1 or shard_count > len(episode_rows):
        raise ValueError("shards must be positive and no greater than the episode count")
    assignments: list[list[dict[str, object]]] = [[] for _ in range(shard_count)]
    heap = [(0.0, shard) for shard in range(shard_count)]
    heapq.heapify(heap)
    ordered_rows = sorted(
        episode_rows,
        key=lambda row: (-estimated_episode_seconds(row), str(row["annotation_key"])),
    )
    for row in ordered_rows:
        load, shard = heapq.heappop(heap)
        assignments[shard].append(row)
        heapq.heappush(heap, (load + estimated_episode_seconds(row), shard))
    loads = [sum(estimated_episode_seconds(row) for row in rows) for rows in assignments]
    return assignments, loads


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    exclusions = load_exclusions(args.exclusions)

    episode_columns = [
        "annotation_key",
        "view",
        "local_episode_path",
        "camera_serial",
        "video_path",
        "future_valid_frames",
        "status",
    ]
    episode_rows = pq.read_table(
        args.foveal_repack_dir / "episodes.parquet",
        columns=episode_columns,
    ).to_pylist()
    episode_by_key = {}
    for row in episode_rows:
        annotation_key = str(row["annotation_key"])
        if annotation_key in episode_by_key:
            raise ValueError(f"Duplicate foveal episode metadata: {annotation_key}")
        episode_by_key[annotation_key] = row

    manifest_dataset = ds.dataset(args.foveal_repack_dir / "manifest", format="parquet")
    manifest_episode_counts: dict[str, int] = {}
    for parquet_path in sorted(manifest_dataset.files):
        table = pq.read_table(parquet_path, columns=["annotation_key", "view"])
        table = table.filter(pc.equal(table["view"], args.exterior_view))
        for value in pc.value_counts(table["annotation_key"]).to_pylist():
            annotation_key = str(value["values"])
            manifest_episode_counts[annotation_key] = manifest_episode_counts.get(annotation_key, 0) + int(
                value["counts"]
            )
    if not manifest_episode_counts:
        raise RuntimeError(f"foveal manifest has no rows for {args.exterior_view}")

    candidates = []
    invalid_manifest_episodes = []
    for annotation_key, manifest_samples in manifest_episode_counts.items():
        row = episode_by_key.get(annotation_key)
        if row is None:
            invalid_manifest_episodes.append({"annotation_key": annotation_key, "reason": "missing_episode_metadata"})
            continue
        svo_path = Path(str(row["local_episode_path"])) / "recordings" / "SVO" / f"{row['camera_serial']}.svo"
        if (
            row["status"] == "ok"
            and row["view"] == args.exterior_view
            and int(row["future_valid_frames"]) > 0
            and svo_path.is_file()
        ):
            row["svo_path"] = str(svo_path)
            row["manifest_samples"] = manifest_samples
            candidates.append(row)
        else:
            invalid_manifest_episodes.append(
                {
                    "annotation_key": annotation_key,
                    "status": row["status"],
                    "view": row["view"],
                    "future_valid_frames": int(row["future_valid_frames"]),
                    "svo_exists": svo_path.is_file(),
                }
            )
    if invalid_manifest_episodes:
        raise ValueError(f"foveal manifest references invalid episode metadata: {invalid_manifest_episodes[:5]}")
    if not candidates:
        raise RuntimeError(f"No {args.exterior_view} episodes with valid foveal samples and SVO files")

    selected_candidates = []
    excluded_candidates = []
    for row in candidates:
        reason = exclusion_reason(row, exclusions)
        if reason is None:
            selected_candidates.append(row)
        else:
            excluded_candidates.append(
                {
                    "annotation_key": str(row["annotation_key"]),
                    "camera_serial": str(row["camera_serial"]),
                    "local_episode_path": str(row["local_episode_path"]),
                    "samples": int(row["manifest_samples"]),
                    "reason": reason,
                }
            )
    if not selected_candidates:
        raise RuntimeError("All candidate episodes are excluded")

    candidate_sample_count = sum(int(row["manifest_samples"]) for row in candidates)
    manifest_sample_count = manifest_dataset.count_rows(filter=ds.field("view") == args.exterior_view)
    if manifest_sample_count != candidate_sample_count:
        raise ValueError(
            f"foveal manifest has {manifest_sample_count} rows but candidate episodes expect {candidate_sample_count}"
        )
    expected_sample_count = sum(int(row["manifest_samples"]) for row in selected_candidates)
    assignments, estimated_loads = assign_episode_shards(selected_candidates, args.shards)
    assignment_by_key = {str(row["annotation_key"]): shard for shard, rows in enumerate(assignments) for row in rows}

    args.output_dir.mkdir(parents=True)
    incomplete_path = args.output_dir / "_INCOMPLETE"
    incomplete_path.write_text("preparing\n")
    manifests_dir = args.output_dir / "manifests"
    manifests_dir.mkdir()
    (args.output_dir / "logs").mkdir()

    sample_chunks: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {
        annotation_key: [] for annotation_key in assignment_by_key
    }
    excluded_keys = {str(row["annotation_key"]) for row in excluded_candidates}
    for parquet_path in sorted(manifest_dataset.files):
        table = pq.read_table(
            parquet_path,
            columns=["annotation_key", "view", "frame_idx", "crop_mode", "crop_bbox"],
        )
        table = table.filter(pc.equal(table["view"], args.exterior_view))
        if not len(table):
            continue
        keys = np.asarray(table["annotation_key"].to_pylist(), dtype=object)
        frame_idx = table["frame_idx"].combine_chunks().to_numpy(zero_copy_only=False)
        crop_mode = table["crop_mode"].combine_chunks().to_numpy(zero_copy_only=False)
        crop_bbox_list = table["crop_bbox"].combine_chunks()
        offsets = crop_bbox_list.offsets.to_numpy(zero_copy_only=False)
        if not np.all(np.diff(offsets) == 4):
            raise ValueError(f"crop_bbox must contain four values per row in {parquet_path}")
        crop_bbox = crop_bbox_list.values.to_numpy(zero_copy_only=False).reshape(-1, 4)

        boundaries = np.concatenate(
            (
                np.asarray((0,), dtype=np.int64),
                np.flatnonzero(keys[1:] != keys[:-1]) + 1,
                np.asarray((len(keys),), dtype=np.int64),
            )
        )
        for start, end in itertools.pairwise(boundaries):
            annotation_key = str(keys[start])
            if annotation_key in excluded_keys:
                continue
            if annotation_key not in sample_chunks:
                raise ValueError(f"Unexpected annotation key in foveal manifest: {annotation_key}")
            sample_chunks[annotation_key].append(
                (
                    frame_idx[start:end].astype(np.int32, copy=True),
                    crop_mode[start:end].astype(np.int8, copy=True),
                    crop_bbox[start:end].astype(np.int32, copy=True),
                )
            )

    shard_summaries = []
    for shard, rows in enumerate(assignments):
        episodes = []
        for row in sorted(rows, key=lambda item: str(item["annotation_key"])):
            annotation_key = str(row["annotation_key"])
            chunks = sample_chunks.pop(annotation_key)
            if not chunks:
                raise ValueError(f"No manifest samples for {annotation_key}")
            frame_idx = np.concatenate([chunk[0] for chunk in chunks])
            crop_mode = np.concatenate([chunk[1] for chunk in chunks])
            crop_bbox = np.concatenate([chunk[2] for chunk in chunks])
            order = np.argsort(frame_idx)
            frame_idx = frame_idx[order]
            crop_mode = crop_mode[order]
            crop_bbox = crop_bbox[order]
            expected_count = int(row["manifest_samples"])
            if len(frame_idx) != expected_count:
                raise ValueError(f"Expected {expected_count} samples for {annotation_key}, found {len(frame_idx)}")
            episodes.append(
                stereo_geometry_manifest.GeometryManifestEpisode(
                    annotation_key=annotation_key,
                    camera_serial=str(row["camera_serial"]),
                    svo_path=str(row["svo_path"]),
                    video_path=str(row["video_path"]),
                    frame_idx=frame_idx,
                    crop_mode=crop_mode,
                    crop_bbox=crop_bbox,
                )
            )

        manifest = stereo_geometry_manifest.GeometryManifest(
            format_version=stereo_geometry_manifest.MANIFEST_FORMAT_VERSION,
            exterior_view=args.exterior_view,
            foveal_repack_dir=str(args.foveal_repack_dir),
            shard_index=shard,
            shard_count=args.shards,
            episodes=tuple(episodes),
        )
        manifest_path = manifests_dir / f"shard-{shard:05d}.npz"
        stereo_geometry_manifest.write_geometry_manifest(manifest_path, manifest)
        shard_summaries.append(
            {
                "shard": shard,
                "manifest": str(manifest_path),
                "episodes": len(episodes),
                "samples": manifest.sample_count,
                "estimated_seconds": estimated_loads[shard],
            }
        )
    if sample_chunks:
        raise AssertionError(f"Unwritten sample chunks for {len(sample_chunks)} episodes")

    estimated_load_array = np.asarray(estimated_loads, dtype=np.float64)
    excluded_by_reason: dict[str, dict[str, int]] = {}
    for row in excluded_candidates:
        counts = excluded_by_reason.setdefault(str(row["reason"]), {"episodes": 0, "samples": 0})
        counts["episodes"] += 1
        counts["samples"] += int(row["samples"])
    index = {
        "index_format_version": "droid-stereo-geometry-production-manifests-v2",
        "format_version": stereo_geometry_manifest.MANIFEST_FORMAT_VERSION,
        "foveal_repack_dir": str(args.foveal_repack_dir),
        "exterior_view": args.exterior_view,
        "shards": args.shards,
        "episodes": len(selected_candidates),
        "samples": expected_sample_count,
        "candidates": {
            "episodes": len(candidates),
            "samples": candidate_sample_count,
        },
        "exclusions": {
            "source": str(args.exclusions),
            "source_sha256": stereo_geometry_manifest.file_sha256(args.exclusions),
            "episodes": len(excluded_candidates),
            "samples": sum(int(row["samples"]) for row in excluded_candidates),
            "by_reason": excluded_by_reason,
            "expanded": sorted(excluded_candidates, key=lambda row: str(row["annotation_key"])),
        },
        "manifest_source_sha256": stereo_geometry_manifest.file_sha256(stereo_geometry_manifest.__file__),
        "preparer_source_sha256": stereo_geometry_manifest.file_sha256(__file__),
        "load_model": {
            "episode_overhead_seconds": _EPISODE_OVERHEAD_SECONDS,
            "seconds_per_sample": _SECONDS_PER_SAMPLE,
        },
        "estimated_seconds_per_shard": {
            "min": float(estimated_load_array.min()),
            "mean": float(estimated_load_array.mean()),
            "max": float(estimated_load_array.max()),
            "max_over_mean": float(estimated_load_array.max() / estimated_load_array.mean()),
        },
        "manifests": shard_summaries,
    }
    (args.output_dir / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    incomplete_path.unlink()
    (args.output_dir / "_SUCCESS").write_text("complete\n")
    print(json.dumps({key: value for key, value in index.items() if key != "manifests"}, indent=2))


if __name__ == "__main__":
    main()
