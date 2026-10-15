from __future__ import annotations

import argparse
import collections
import hashlib
import json
import shutil
import time
from pathlib import Path

from droid_foveation.geometry import FrameInteractionAnnotation, annotate_frame
from droid_foveation.parquet_io import EPISODE_SCHEMA, FRAME_SCHEMA, ShardedParquetWriter
from droid_foveation.role_policy import RolePolicy
from droid_foveation.sources import GCS_DROID_PREFIX, EpisodeResolver, RoboInterAnnotationDb, iter_qsheet


def build_sidecar(args: argparse.Namespace) -> dict[str, object]:
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    (output_dir / "_INCOMPLETE").write_text("annotation run in progress\n")

    role_policy_path = Path(args.role_policy)
    role_policy = RolePolicy.load(role_policy_path)
    shutil.copy2(role_policy_path, output_dir / "role_policy.yaml")

    selected_keys = _load_selected_keys(args.keys_file)
    resolver = EpisodeResolver(args.raw_droid_root, exterior_roles=args.exterior_roles)
    frame_writer = ShardedParquetWriter(output_dir / "frames", FRAME_SCHEMA, args.frame_rows_per_shard)
    episode_writer = ShardedParquetWriter(output_dir / "episodes", EPISODE_SCHEMA, args.episode_rows_per_shard)

    primitive_frames: collections.Counter[str] = collections.Counter()
    primitive_valid_frames: collections.Counter[str] = collections.Counter()
    primitive_episode_views: collections.Counter[str] = collections.Counter()
    primitive_example_keys: dict[str, list[str]] = collections.defaultdict(list)
    invalid_reasons: collections.Counter[str] = collections.Counter()
    source_statuses: collections.Counter[str] = collections.Counter()
    annotation_video_deltas: collections.Counter[str] = collections.Counter()
    hdf5_annotation_deltas: collections.Counter[str] = collections.Counter()

    processed_entries = 0
    total_frames = 0
    valid_frames = 0
    total_future_eligible = 0
    total_future_valid = 0
    missing_lmdb_entries = 0
    scanned_qsheet_entries = 0
    skipped_origin_entries = 0
    start_time = time.monotonic()

    with RoboInterAnnotationDb(args.annotation_lmdb) as annotation_db:
        for annotation_key, qsheet_entry in iter_qsheet(args.qsheet):
            scanned_qsheet_entries += 1
            if selected_keys is not None and annotation_key not in selected_keys:
                continue
            if args.ori_prefix and not str(qsheet_entry.get("ori_path", "")).startswith(args.ori_prefix):
                skipped_origin_entries += 1
                continue
            annotation = annotation_db.get(annotation_key)
            resolved = resolver.resolve(qsheet_entry)
            source_statuses[resolved.status] += 1

            if annotation is None:
                missing_lmdb_entries += 1
                episode_writer.extend(
                    [_episode_row(annotation_key, qsheet_entry, resolved, None, [], args.future_offset, "missing_lmdb")]
                )
                processed_entries += 1
                if _should_stop(processed_entries, args.limit, selected_keys):
                    break
                continue

            frame_rows: list[dict[str, object]] = []
            frame_annotations: dict[int, FrameInteractionAnnotation] = {}
            entry_primitives: set[str] = set()
            for raw_frame_idx in sorted(annotation):
                frame_idx = int(raw_frame_idx)
                frame_annotation = annotate_frame(annotation[raw_frame_idx], role_policy)
                frame_annotations[frame_idx] = frame_annotation
                frame_rows.append(frame_annotation.to_row(annotation_key, frame_idx))
                primitive_name = frame_annotation.primitive_skill or "<missing>"
                primitive_frames[primitive_name] += 1
                entry_primitives.add(primitive_name)
                if (
                    annotation_key not in primitive_example_keys[primitive_name]
                    and len(primitive_example_keys[primitive_name]) < args.primitive_examples
                ):
                    primitive_example_keys[primitive_name].append(annotation_key)
                if frame_annotation.box_valid:
                    valid_frames += 1
                    primitive_valid_frames[primitive_name] += 1
                else:
                    invalid_reasons[frame_annotation.invalid_reason or "unknown"] += 1

            future_eligible, future_valid = _attach_future_targets(
                frame_rows, frame_annotations, qsheet_entry.get("range_nop"), args.future_offset
            )
            frame_writer.extend(frame_rows)
            primitive_episode_views.update(entry_primitives)
            total_frames += len(frame_rows)
            total_future_eligible += future_eligible
            total_future_valid += future_valid

            episode_writer.extend(
                [
                    _episode_row(
                        annotation_key,
                        qsheet_entry,
                        resolved,
                        annotation,
                        list(frame_annotations.values()),
                        args.future_offset,
                        resolved.status,
                    )
                ]
            )

            annotation_frames = len(annotation)
            if resolved.video_frames is not None:
                annotation_video_deltas[str(annotation_frames - resolved.video_frames)] += 1
            if resolved.hdf5_steps is not None:
                hdf5_annotation_deltas[str(resolved.hdf5_steps - annotation_frames)] += 1

            processed_entries += 1
            if processed_entries % args.progress_every == 0:
                elapsed = time.monotonic() - start_time
                print(
                    f"processed={processed_entries} frames={total_frames} "
                    f"valid={valid_frames} elapsed_s={elapsed:.1f}",
                    flush=True,
                )
            if _should_stop(processed_entries, args.limit, selected_keys):
                break

    frame_writer.close()
    episode_writer.close()

    elapsed_seconds = time.monotonic() - start_time
    summary: dict[str, object] = {
        "format_version": "0.2.0",
        "role_policy_version": role_policy.version,
        "role_policy_sha256": hashlib.sha256(role_policy_path.read_bytes()).hexdigest(),
        "future_offset": args.future_offset,
        "processed_episode_views": processed_entries,
        "scanned_qsheet_entries": scanned_qsheet_entries,
        "skipped_origin_entries": skipped_origin_entries,
        "missing_lmdb_entries": missing_lmdb_entries,
        "total_frames": total_frames,
        "valid_box_frames": valid_frames,
        "invalid_box_frames": total_frames - valid_frames,
        "future_eligible_frames": total_future_eligible,
        "future_valid_frames": total_future_valid,
        "elapsed_seconds": elapsed_seconds,
        "episode_views_per_second": processed_entries / elapsed_seconds if elapsed_seconds else None,
        "frames_per_second": total_frames / elapsed_seconds if elapsed_seconds else None,
        "primitive_frames": dict(primitive_frames.most_common()),
        "primitive_valid_frames": dict(primitive_valid_frames.most_common()),
        "primitive_episode_views": dict(primitive_episode_views.most_common()),
        "primitive_example_keys": dict(sorted(primitive_example_keys.items())),
        "invalid_reasons": dict(invalid_reasons.most_common()),
        "source_statuses": dict(source_statuses.most_common()),
        "annotation_minus_video_frames": dict(sorted(annotation_video_deltas.items(), key=lambda item: int(item[0]))),
        "hdf5_minus_annotation_frames": dict(sorted(hdf5_annotation_deltas.items(), key=lambda item: int(item[0]))),
        "inputs": {
            "qsheet": str(Path(args.qsheet).resolve()),
            "annotation_lmdb": str(Path(args.annotation_lmdb).resolve()),
            "raw_droid_root": str(Path(args.raw_droid_root).resolve()),
            "exterior_roles": str(Path(args.exterior_roles).resolve()) if args.exterior_roles else None,
        },
    }
    (output_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (output_dir / "_INCOMPLETE").unlink()
    (output_dir / "_SUCCESS").write_text("annotation run completed\n")
    return summary


def _episode_row(
    annotation_key: str,
    qsheet_entry: dict[str, object],
    resolved: object,
    annotation: dict[int, dict[str, object]] | None,
    frame_annotations: list[FrameInteractionAnnotation],
    future_offset: int,
    status: str,
) -> dict[str, object]:
    from droid_foveation.sources import ResolvedEpisodeView

    assert isinstance(resolved, ResolvedEpisodeView)
    frame_indices = sorted(int(index) for index in annotation) if annotation else []
    annotations_by_index = {
        frame_idx: frame_annotation for frame_idx, frame_annotation in zip(frame_indices, frame_annotations, strict=True)
    }
    future_eligible, future_valid = _future_coverage(
        annotations_by_index, qsheet_entry.get("range_nop"), future_offset
    )
    range_start, range_end, range_count = _parse_range_nop(qsheet_entry.get("range_nop"))
    return {
        "annotation_key": annotation_key,
        "ori_path": _optional_str(qsheet_entry.get("ori_path")),
        "view": _optional_str(qsheet_entry.get("view")),
        "local_h5_path": resolved.local_h5_path,
        "local_episode_path": resolved.local_episode_path,
        "metadata_path": resolved.metadata_path,
        "camera_serial": resolved.camera_serial,
        "video_path": resolved.video_path,
        "annotation_frames": len(frame_indices),
        "annotation_first_frame": frame_indices[0] if frame_indices else None,
        "annotation_last_frame": frame_indices[-1] if frame_indices else None,
        "video_frames": resolved.video_frames,
        "video_width": resolved.video_width,
        "video_height": resolved.video_height,
        "hdf5_steps": resolved.hdf5_steps,
        "range_nop_start": range_start,
        "range_nop_end": range_end,
        "range_nop_count": range_count,
        "box_valid_frames": sum(frame.box_valid for frame in frame_annotations),
        "future_eligible_frames": future_eligible,
        "future_valid_frames": future_valid,
        "status": status,
        "q_primitive_skill": _optional_str(qsheet_entry.get("Q_primitive_skill")),
        "q_object_box": _optional_str(qsheet_entry.get("Q_object_box")),
        "q_placement_proposal": _optional_str(qsheet_entry.get("Q_placement_proposal")),
        "q_gripper_box": _optional_str(qsheet_entry.get("Q_gripper_box")),
    }


def _future_coverage(
    annotations: dict[int, FrameInteractionAnnotation], range_nop: object, future_offset: int
) -> tuple[int, int]:
    if not annotations:
        return 0, 0
    range_start, range_end, _ = _parse_range_nop(range_nop)
    if range_start is None or range_end is None:
        range_start, range_end = min(annotations), max(annotations)
    eligible = 0
    valid = 0
    for frame_idx in range(range_start, range_end + 1):
        future_idx = frame_idx + future_offset
        if frame_idx not in annotations or future_idx not in annotations:
            continue
        eligible += 1
        if annotations[frame_idx].box_valid and annotations[future_idx].box_valid:
            valid += 1
    return eligible, valid


def _attach_future_targets(
    frame_rows: list[dict[str, object]],
    annotations: dict[int, FrameInteractionAnnotation],
    range_nop: object,
    future_offset: int,
) -> tuple[int, int]:
    if not annotations:
        return 0, 0
    range_start, range_end, _ = _parse_range_nop(range_nop)
    if range_start is None or range_end is None:
        range_start, range_end = min(annotations), max(annotations)

    eligible = 0
    valid = 0
    for row in frame_rows:
        frame_idx = int(row["frame_idx"])
        current = annotations[frame_idx]
        future_idx = frame_idx + future_offset
        future = annotations.get(future_idx)
        pair_eligible = range_start <= frame_idx <= range_end and future is not None
        pair_valid = bool(pair_eligible and current.box_valid and future.box_valid)
        if pair_eligible:
            eligible += 1
        if pair_valid:
            valid += 1
        row.update(
            {
                "future_frame_idx": future_idx if future is not None else None,
                "future_primitive_skill": (future.primitive_skill or None) if future is not None else None,
                "future_interaction_box": future.interaction_box if future is not None and future.box_valid else None,
                "future_box_valid": bool(future is not None and future.box_valid),
                "training_pair_valid": pair_valid,
            }
        )
    return eligible, valid


def _parse_range_nop(value: object) -> tuple[int | None, int | None, int | None]:
    if not isinstance(value, list) or len(value) < 2:
        return None, None, None
    try:
        start, end = int(value[0]), int(value[1])
        count = int(value[2]) if len(value) > 2 else end - start + 1
    except (TypeError, ValueError):
        return None, None, None
    return start, end, count


def _load_selected_keys(path: str | None) -> set[str] | None:
    if path is None:
        return None
    return {line.strip() for line in Path(path).read_text().splitlines() if line.strip()}


def _should_stop(processed: int, limit: int | None, selected_keys: set[str] | None) -> bool:
    if limit is not None and processed >= limit:
        return True
    return selected_keys is not None and processed >= len(selected_keys)


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an offline DROID interaction box annotation sidecar.")
    parser.add_argument("--qsheet", required=True)
    parser.add_argument("--annotation-lmdb", required=True)
    parser.add_argument("--raw-droid-root", required=True)
    parser.add_argument("--exterior-roles")
    parser.add_argument(
        "--ori-prefix",
        default=GCS_DROID_PREFIX,
        help="Only process Qsheet entries whose ori_path starts with this value.",
    )
    parser.add_argument("--role-policy", default="configs/role_policy_v1.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--future-offset", type=int, default=8)
    parser.add_argument("--keys-file")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--frame-rows-per-shard", type=int, default=100_000)
    parser.add_argument("--episode-rows-per-shard", type=int, default=10_000)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--primitive-examples", type=int, default=5)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    summary = build_sidecar(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
