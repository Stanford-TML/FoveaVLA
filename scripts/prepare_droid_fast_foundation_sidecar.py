#!/usr/bin/env python3
"""Initialize or finalize a sharded Fast-FoundationStereo depth sidecar."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zlib

import lmdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from foveavla.training import droid_pointmap_store
from foveavla.training import droid_stereo_geometry_store

PLAN_FORMAT = "droid-fast-foundation-plan-v1"
REPORT_FORMAT = "droid-fast-foundation-worker-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("initialize", "finalize"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-sidecar", type=Path)
    parser.add_argument("--manifest-root", type=Path)
    parser.add_argument("--foveal-repack-dir", "--roi-repack-dir", type=Path)
    parser.add_argument("--alignment-audit-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--fast-repo", type=Path)
    parser.add_argument("--worker-count", type=int, default=64)
    parser.add_argument("--map-size-gb", type=float, default=64.0)
    parser.add_argument("--input-width", type=int, default=640)
    parser.add_argument("--input-height", type=int, default=360)
    parser.add_argument("--max-disparity", type=int, default=192)
    parser.add_argument("--valid-iters", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
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


def git_commit(path: Path) -> str:
    return subprocess.run(
        ("git", "-C", str(path), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def require_clean_git(path: Path) -> str:
    status = subprocess.run(
        ("git", "-C", str(path), "status", "--porcelain", "--untracked-files=normal"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RuntimeError(f"Refusing to initialize from a dirty worktree: {path}")
    return git_commit(path)


def initialize(args: argparse.Namespace) -> None:
    required = {
        "manifest-root": args.manifest_root,
        "checkpoint": args.checkpoint,
        "fast-repo": args.fast_repo,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"initialize requires: {', '.join(missing)}")
    assert args.manifest_root is not None
    assert args.checkpoint is not None
    assert args.fast_repo is not None
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.worker_count < 1 or args.map_size_gb <= 0 or args.batch_size < 1:
        raise ValueError("worker-count and map-size-gb must be positive")
    if not (args.manifest_root / "_SUCCESS").is_file():
        raise RuntimeError("Manifest root is incomplete")
    if not args.checkpoint.is_file() or not args.fast_repo.is_dir():
        raise FileNotFoundError("Fast-FoundationStereo checkpoint or repository is missing")

    if args.source_sidecar is not None:
        source_index, source_summary, episode_table, source_hashes = _load_source_sidecar(
            args.source_sidecar,
            args.worker_count,
        )
        source_sidecar = str(args.source_sidecar.resolve())
    else:
        if args.foveal_repack_dir is None or args.alignment_audit_root is None:
            raise ValueError("Direct manifest initialization requires --foveal-repack-dir and --alignment-audit-root")
        source_index, source_summary, episode_table, source_hashes = _build_manifest_source(
            args.manifest_root,
            args.foveal_repack_dir,
            args.alignment_audit_root,
            args.worker_count,
        )
        source_sidecar = None

    annotation_keys = sorted(str(value) for value in episode_table.column("annotation_key").to_pylist())
    worker_episode_counts = [0] * args.worker_count
    for annotation_key in annotation_keys:
        worker_episode_counts[zlib.crc32(annotation_key.encode()) % args.worker_count] += 1

    foveavla_repo = Path(__file__).resolve().parents[1]
    foveavla_commit = require_clean_git(foveavla_repo)
    extractor_path = foveavla_repo / "scripts" / "extract_droid_fast_foundation_depth.py"
    postprocess_path = foveavla_repo / "src" / "foveavla" / "shared" / "fast_foundation_depth.py"

    args.output_dir.mkdir(parents=True)
    (args.output_dir / "pointmaps").mkdir()
    (args.output_dir / "worker_reports").mkdir()
    (args.output_dir / "worker_status").mkdir()
    (args.output_dir / "_INCOMPLETE").write_text("incomplete\n")
    pq.write_table(episode_table, args.output_dir / "episodes.parquet")
    atomic_json(args.output_dir / "selected_annotation_keys.json", annotation_keys)

    plan = {
        "format_version": PLAN_FORMAT,
        "worker_count": args.worker_count,
        "worker_episode_counts": worker_episode_counts,
        "map_size_bytes": int(args.map_size_gb * 1024**3),
        "output_dir": str(args.output_dir.resolve()),
        "source_sidecar": source_sidecar,
        "direct_source_index": source_index if source_sidecar is None else None,
        "source": {
            **source_hashes,
            "episodes": int(source_summary["episodes"]),
            "samples": int(source_summary["samples"]),
        },
        "manifest_root": str(args.manifest_root.resolve()),
        "manifest_index_sha256": sha256_file(args.manifest_root / "index.json"),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "fast_repo": str(args.fast_repo.resolve()),
        "fast_repo_commit": git_commit(args.fast_repo),
        # Retain historical plan keys so existing depth preparation plans remain readable.
        "openpi_repo": str(foveavla_repo),
        "openpi_commit": foveavla_commit,
        "extractor_source_sha256": sha256_file(extractor_path),
        "postprocess_source_sha256": sha256_file(postprocess_path),
        "input_resolution": [args.input_width, args.input_height],
        "source_resolution": [1280, 720],
        "max_disparity": args.max_disparity,
        "valid_iters": args.valid_iters,
        "batch_size": args.batch_size,
        "low_memory": True,
        "depth_range_m": [0.2, 3.0],
        "output_shape": [224, 224],
        "shard_assignment": "crc32(annotation_key) % worker_count",
    }
    atomic_json(args.output_dir / "plan.json", plan)
    print(json.dumps(plan, indent=2))


def _load_source_sidecar(
    source_sidecar: Path,
    worker_count: int,
) -> tuple[dict[str, object], dict[str, object], pa.Table, dict[str, str]]:
    if not (source_sidecar / "_SUCCESS").is_file():
        raise RuntimeError("Source sidecar is incomplete")
    index_path = source_sidecar / "index.json"
    summary_path = source_sidecar / "summary.json"
    episodes_path = source_sidecar / "episodes.parquet"
    index = json.loads(index_path.read_text())
    summary = json.loads(summary_path.read_text())
    if index.get("format_version") != droid_pointmap_store.FORMAT_VERSION:
        raise ValueError("Unsupported source sidecar format")
    if int(index.get("store_shards", -1)) != worker_count:
        raise ValueError("worker-count must match the source sidecar shard count")
    episode_table = pq.read_table(episodes_path)
    if len(episode_table) != int(summary["episodes"]):
        raise ValueError("Source episode table and summary disagree")
    return (
        index,
        summary,
        episode_table,
        {
            "index_sha256": sha256_file(index_path),
            "summary_sha256": sha256_file(summary_path),
            "episodes_sha256": sha256_file(episodes_path),
        },
    )


def _build_manifest_source(
    manifest_root: Path,
    foveal_repack_dir: Path,
    alignment_audit_root: Path,
    worker_count: int,
) -> tuple[dict[str, object], dict[str, object], pa.Table, dict[str, str]]:
    manifest_index_path = manifest_root / "index.json"
    manifest_index = json.loads(manifest_index_path.read_text())
    exterior_view = str(manifest_index.get("exterior_view", ""))
    if manifest_index.get(
        "index_format_version"
    ) != "droid-stereo-geometry-production-manifests-v2" or exterior_view not in (
        "exterior_image_1_left",
        "exterior_image_2_left",
    ):
        raise ValueError("Unsupported production manifest index")
    alignment_audit = droid_stereo_geometry_store.alignment_audit_attestation(alignment_audit_root)
    if Path(str(alignment_audit["manifest_root"])).resolve() != manifest_root.resolve():
        raise ValueError("Alignment audit belongs to a different manifest root")
    excluded = set(alignment_audit["excluded_annotation_keys"]) | set(alignment_audit["ambiguous_annotation_keys"])
    sample_count_by_key: dict[str, int] = {}
    for manifest_path in sorted((manifest_root / "manifests").glob("shard-*.npz")):
        with np.load(manifest_path, allow_pickle=False) as manifest:
            if str(manifest["exterior_view"]) != exterior_view:
                raise ValueError(f"Manifest exterior view mismatch: {manifest_path}")
            offsets = manifest["episode_offsets"]
            for index, value in enumerate(manifest["annotation_key"]):
                annotation_key = str(value)
                if annotation_key in sample_count_by_key:
                    raise ValueError(f"Duplicate manifest annotation key: {annotation_key}")
                sample_count_by_key[annotation_key] = int(offsets[index + 1]) - int(offsets[index])
    selected_keys = sorted(set(sample_count_by_key) - excluded)
    if not selected_keys:
        raise RuntimeError("Alignment audit excludes every manifest episode")

    foveal_table = pq.read_table(foveal_repack_dir / "episodes.parquet", columns=["annotation_key", "ori_path", "view"])
    rows_by_key = {str(row["annotation_key"]): row for row in foveal_table.to_pylist()}
    if len(rows_by_key) != len(foveal_table):
        raise ValueError("foveal episode metadata contains duplicate annotation keys")
    missing = set(selected_keys) - set(rows_by_key)
    if missing:
        raise ValueError(f"foveal episode metadata is missing manifest keys: {sorted(missing)[:5]}")
    selected_rows = [rows_by_key[key] for key in selected_keys]
    if any(str(row["view"]) != exterior_view for row in selected_rows):
        raise ValueError("foveal episode view does not match the production manifest")
    episode_table = pa.Table.from_pylist(selected_rows)
    selected_hash = droid_stereo_geometry_store.annotation_keys_sha256(selected_keys)
    source = {
        "foveal_repack_dir": str(foveal_repack_dir.resolve()),
        "foveal_repack_fingerprint": droid_stereo_geometry_store.foveal_repack_fingerprint(foveal_repack_dir),
        "alignment_audit": alignment_audit,
        "exclusions": manifest_index.get("exclusions", {"expanded": []}),
        "selected_episodes": len(selected_keys),
        "selected_annotation_keys_sha256": selected_hash,
        "geometry_manifest_root": str(manifest_root.resolve()),
    }
    index = {
        "format_version": droid_pointmap_store.FORMAT_VERSION,
        "exterior_view": exterior_view,
        "store_shards": worker_count,
        "sample_key": "{annotation_key}--{frame_idx}",
        "pointmap": {
            "shape": list(droid_pointmap_store.POINTMAP_SHAPE),
            "model_dtype": "float32",
            "frame": "zed_left_camera",
            "normalization_scale_m": 3.0,
            "storage": "aligned_metric_depth_plus_camera_model",
            "depth_dtype": droid_pointmap_store.DEPTH_DTYPE.str,
            "views": ["full", "crop"],
        },
        "quality": {
            "shape": list(droid_pointmap_store.QUALITY_SHAPE),
            "storage_dtype": droid_pointmap_store.QUALITY_DTYPE.str,
        },
        "value": {
            "compression": "zlib",
            "compression_level": 1,
            "raw_bytes": droid_pointmap_store.RAW_VALUE_BYTES,
        },
        "source": source,
    }
    summary = {
        "episodes": len(selected_keys),
        "samples": sum(sample_count_by_key[key] for key in selected_keys),
    }
    return (
        index,
        summary,
        episode_table,
        {
            "manifest_index_sha256": sha256_file(manifest_index_path),
            "alignment_validation_sha256": sha256_file(alignment_audit_root / "validation.json"),
            "foveal_episodes_sha256": sha256_file(foveal_repack_dir / "episodes.parquet"),
        },
    )


def finalize(args: argparse.Namespace) -> None:
    plan_path = args.output_dir / "plan.json"
    plan = json.loads(plan_path.read_text())
    if plan.get("format_version") != PLAN_FORMAT:
        raise ValueError("Unsupported plan")
    if (args.output_dir / "_SUCCESS").exists():
        raise RuntimeError("Sidecar is already finalized")
    worker_count = int(plan["worker_count"])
    plan_sha256 = sha256_file(plan_path)
    reports = []
    progress_databases = []
    for worker_index in range(worker_count):
        status_path = args.output_dir / "worker_status" / f"worker-{worker_index:03d}.success"
        report_path = args.output_dir / "worker_reports" / f"worker-{worker_index:03d}.json"
        if not status_path.is_file() or not report_path.is_file():
            raise RuntimeError(f"Worker {worker_index} is incomplete")
        report = json.loads(report_path.read_text())
        if (
            report.get("format_version") != REPORT_FORMAT
            or int(report.get("worker_index", -1)) != worker_index
            or report.get("plan_sha256") != plan_sha256
        ):
            raise ValueError(f"Invalid worker report: {report_path}")
        lmdb_path = args.output_dir / "pointmaps" / f"shard-{worker_index:03d}.lmdb"
        environment = lmdb.open(
            str(lmdb_path),
            readonly=True,
            lock=False,
            readahead=False,
            max_dbs=2,
        )
        try:
            with environment.begin() as transaction:
                try:
                    progress = environment.open_db(b"progress", txn=transaction, create=False)
                except lmdb.NotFoundError:
                    progress = None
                main_entries = transaction.stat()["entries"]
                if progress is None:
                    if main_entries != int(report["samples"]):
                        raise ValueError(f"Sample count mismatch in {lmdb_path}")
                else:
                    if main_entries - 1 != int(report["samples"]):
                        raise ValueError(f"Sample count mismatch in {lmdb_path}")
                    if transaction.stat(db=progress)["entries"] != int(report["episodes"]):
                        raise ValueError(f"Episode count mismatch in {lmdb_path}")
                    progress_databases.append((lmdb_path, int(report["samples"])))
        finally:
            environment.close()
        reports.append(report)

    source_sidecar_value = plan.get("source_sidecar")
    if source_sidecar_value is None:
        source_index = dict(plan["direct_source_index"])
    else:
        source_sidecar = Path(str(source_sidecar_value))
        source_index = json.loads((source_sidecar / "index.json").read_text())
    total_episodes = sum(int(report["episodes"]) for report in reports)
    total_samples = sum(int(report["samples"]) for report in reports)
    if total_episodes != int(plan["source"]["episodes"]) or total_samples != int(plan["source"]["samples"]):
        raise ValueError("Completed workers do not cover the source sidecar exactly")
    for lmdb_path, sample_count in progress_databases:
        environment = lmdb.open(str(lmdb_path), map_size=int(plan["map_size_bytes"]), max_dbs=2)
        try:
            progress = environment.open_db(b"progress", create=False)
            with environment.begin(write=True) as transaction:
                transaction.drop(progress, delete=True)
            environment.sync()
            if environment.stat()["entries"] != sample_count:
                raise ValueError(f"Final sample count mismatch in {lmdb_path}")
        finally:
            environment.close()
    compressed_bytes = sum(int(report["compressed_bytes"]) for report in reports)
    pixels_per_view = total_samples * 224 * 224
    summary = {
        "episodes": total_episodes,
        "samples": total_samples,
        "raw_bytes": total_samples * droid_pointmap_store.RAW_VALUE_BYTES,
        "compressed_bytes": compressed_bytes,
        "compression_ratio": compressed_bytes / (total_samples * droid_pointmap_store.RAW_VALUE_BYTES),
        "compressed_bytes_per_sample_mean": compressed_bytes / total_samples,
        "full_valid_fraction": sum(int(report["full_valid_pixels"]) for report in reports) / pixels_per_view,
        "crop_valid_fraction": sum(int(report["crop_valid_pixels"]) for report in reports) / pixels_per_view,
        "workers": worker_count,
        "store_shards": worker_count,
        "full_array_validation": True,
    }
    index = dict(source_index)
    index["store_shards"] = worker_count
    index["depth_source"] = {
        "sensor": "ZED stereo SVO rectified pair",
        "sdk_version": "5.1.2",
        "estimator": "Fast-FoundationStereo",
        "repository_commit": plan["fast_repo_commit"],
        "preprocessing_commit": plan["openpi_commit"],
        "postprocess_source_sha256": plan["postprocess_source_sha256"],
        "checkpoint_sha256": plan["checkpoint_sha256"],
        "input_resolution": plan["input_resolution"],
        "max_disparity": plan["max_disparity"],
        "valid_iters": plan["valid_iters"],
        "low_memory": plan["low_memory"],
        "alignment": "left",
        "metric_conversion": "fx_times_baseline_over_disparity",
        "visibility_mask": "positive_disparity_and_right_correspondence_in_bounds",
        "range_m": plan["depth_range_m"],
    }
    index["source"] = {
        **source_index["source"],
        "fast_foundation_plan_sha256": plan_sha256,
    }
    if source_sidecar_value is not None:
        index["source"]["parent_sidecar"] = str(Path(str(source_sidecar_value)).resolve())
    atomic_json(args.output_dir / "index.json", index)
    atomic_json(args.output_dir / "summary.json", summary)
    provenance = {
        "format_version": droid_pointmap_store.PROVENANCE_FORMAT_VERSION,
        "data": {
            "index_sha256": sha256_file(args.output_dir / "index.json"),
            "summary_sha256": sha256_file(args.output_dir / "summary.json"),
            "episodes_parquet_sha256": sha256_file(args.output_dir / "episodes.parquet"),
            "selected_annotation_keys_sha256": index["source"]["selected_annotation_keys_sha256"],
            "episodes": total_episodes,
            "samples": total_samples,
        },
        "generation": {
            "plan_sha256": plan_sha256,
            "worker_report_sha256": {
                f"worker-{worker_index:03d}": sha256_file(
                    args.output_dir / "worker_reports" / f"worker-{worker_index:03d}.json"
                )
                for worker_index in range(worker_count)
            },
        },
    }
    atomic_json(args.output_dir / "provenance.json", provenance)
    (args.output_dir / "_SUCCESS").write_text("complete\n")
    (args.output_dir / "_INCOMPLETE").unlink()
    print(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    if args.mode == "initialize":
        initialize(args)
    else:
        finalize(args)


if __name__ == "__main__":
    main()
