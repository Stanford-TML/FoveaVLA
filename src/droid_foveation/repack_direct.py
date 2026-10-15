from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import struct
import time

import cv2
import lmdb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from PIL import Image

from droid_foveation.parquet_io import ShardedParquetWriter, read_frame_table
from droid_foveation.repack import BOX_TYPE
from droid_foveation.repack import (
    _build_paired_frame_index,
    _intersect_ranges,
    _range_count,
    sample_key,
    store_shard,
)

FORMAT_VERSION = "2.2.0"
CROP_MODE_FULL = 1
CROP_MODE_EXACT = 2
CROP_MODE_JITTER = 3
VALUE_HEADER = struct.Struct("<4fB3x4iI")
PIXEL_EPSILON = 1e-3

MANIFEST_SCHEMA = pa.schema(
    [
        ("sample_key", pa.string()),
        ("annotation_key", pa.string()),
        ("ori_path", pa.string()),
        ("view", pa.string()),
        ("frame_idx", pa.int32()),
        ("primitive_skill", pa.string()),
        ("interaction_box", BOX_TYPE),
        ("future_interaction_box", BOX_TYPE),
        ("crop_mode", pa.int8()),
        ("crop_bbox", pa.list_(pa.int32())),
        ("foveal_png_bytes", pa.int32()),
        ("store_shard", pa.int16()),
    ]
)


@dataclasses.dataclass(frozen=True)
class EncodedDirectSample:
    future_interaction_box: np.ndarray
    crop_mode: int
    crop_bbox: tuple[int, int, int, int]
    foveal_png: bytes


@dataclasses.dataclass(frozen=True)
class EncodedView:
    annotation_key: str
    store_shard: int
    samples: list[tuple[str, bytes, dict[str, object], int]]
    video_failures: list[str]


def resize_with_pad(image: np.ndarray, output_size: int) -> np.ndarray:
    """Resize and letterbox like OpenPI's client-side full-image renderer."""
    height, width = image.shape[:2]
    if (height, width) == (output_size, output_size):
        return image
    ratio = max(width / output_size, height / output_size)
    resized_height = int(height / ratio)
    resized_width = int(width / ratio)
    resized = np.asarray(
        Image.fromarray(image).resize(
            (resized_width, resized_height),
            resample=Image.Resampling.BILINEAR,
        )
    )
    canvas = np.zeros((output_size, output_size, 3), dtype=np.uint8)
    top = max(0, (output_size - resized_height) // 2)
    left = max(0, (output_size - resized_width) // 2)
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return canvas


def exact_crop_bbox(
    image_shape: tuple[int, ...],
    box: tuple[float, float, float, float] | list[float] | np.ndarray,
    context_scale: float,
) -> tuple[int, int, int, int]:
    height, width = image_shape[:2]
    x1, y1, x2, y2 = np.asarray(box, dtype=np.float32)
    center_x = float(x1 + x2) * width / 2
    center_y = float(y1 + y2) * height / 2
    return _contained_crop_bbox(image_shape, box, center_x, center_y, context_scale)


def jitter_crop_bbox(
    image_shape: tuple[int, ...],
    box: tuple[float, float, float, float] | list[float] | np.ndarray,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> tuple[int, int, int, int]:
    height, width = image_shape[:2]
    x1, y1, x2, y2 = np.asarray(box, dtype=np.float32)
    box_width = float(x2 - x1) * width
    box_height = float(y2 - y1) * height
    box_side = max(box_width, box_height)
    center_x = float(x1 + x2) * width / 2
    center_y = float(y1 + y2) * height / 2
    center_x += float(rng.uniform(-args.jitter_center_fraction, args.jitter_center_fraction)) * box_side
    center_y += float(rng.uniform(-args.jitter_center_fraction, args.jitter_center_fraction)) * box_side
    scale = math.exp(float(rng.uniform(math.log(args.jitter_scale_min), math.log(args.jitter_scale_max))))
    return _contained_crop_bbox(
        image_shape,
        box,
        center_x,
        center_y,
        args.context_scale * scale,
    )


def _contained_crop_bbox(
    image_shape: tuple[int, ...],
    box: tuple[float, float, float, float] | list[float] | np.ndarray,
    center_x: float,
    center_y: float,
    scale: float,
) -> tuple[int, int, int, int]:
    """Place the requested square in-frame unless the interaction box itself cannot fit."""
    height, width = image_shape[:2]
    x1, y1, x2, y2 = np.asarray(box, dtype=np.float32)
    box_left = float(x1) * width
    box_top = float(y1) * height
    box_right = float(x2) * width
    box_bottom = float(y2) * height
    box_side = max(box_right - box_left, box_bottom - box_top)
    required_side = max(
        2,
        math.ceil(box_right - PIXEL_EPSILON) - math.floor(box_left + PIXEL_EPSILON),
        math.ceil(box_bottom - PIXEL_EPSILON) - math.floor(box_top + PIXEL_EPSILON),
    )
    side = max(
        required_side,
        math.ceil(box_side * scale - PIXEL_EPSILON),
    )
    if required_side <= min(width, height):
        side = min(side, min(width, height))
    else:
        side = min(side, max(width, height))

    requested_left = math.floor(center_x - side / 2 + PIXEL_EPSILON)
    requested_top = math.floor(center_y - side / 2 + PIXEL_EPSILON)
    left = _fit_crop_start(requested_left, side, width, box_left, box_right)
    top = _fit_crop_start(requested_top, side, height, box_top, box_bottom)
    return left, top, left + side, top + side


def _fit_crop_start(
    requested: int,
    side: int,
    image_length: int,
    box_min: float,
    box_max: float,
) -> int:
    if side <= image_length:
        lower = max(0, math.ceil(box_max - side - PIXEL_EPSILON))
        upper = min(image_length - side, math.floor(box_min + PIXEL_EPSILON))
    else:
        lower = image_length - side
        upper = 0
    if lower > upper:
        raise ValueError("Could not place a square crop that contains the interaction box")
    return min(max(requested, lower), upper)


def choose_crop(
    image_shape: tuple[int, ...],
    box: tuple[float, float, float, float] | list[float] | np.ndarray,
    key: str,
    args: argparse.Namespace,
) -> tuple[int, tuple[int, int, int, int]]:
    digest = hashlib.blake2b(f"{args.seed}:{key}".encode(), digest_size=8).digest()
    rng = np.random.default_rng(int.from_bytes(digest, "little"))
    draw = float(rng.random())
    height, width = image_shape[:2]
    if draw < args.full_image_probability:
        return CROP_MODE_FULL, (0, 0, width, height)
    if draw < args.full_image_probability + args.jitter_probability:
        return CROP_MODE_JITTER, jitter_crop_bbox(image_shape, box, rng, args)
    return CROP_MODE_EXACT, exact_crop_bbox(image_shape, box, args.context_scale)


def render_square_crop(
    image: np.ndarray,
    crop_bbox: tuple[int, int, int, int],
    output_size: int,
) -> np.ndarray:
    left, top, right, bottom = crop_bbox
    side = right - left
    if side <= 0 or bottom - top != side:
        raise ValueError(f"Expected a non-empty square crop, got {crop_bbox}")
    height, width = image.shape[:2]
    canvas = np.zeros((side, side, 3), dtype=np.uint8)
    source_left = max(left, 0)
    source_top = max(top, 0)
    source_right = min(right, width)
    source_bottom = min(bottom, height)
    if source_left < source_right and source_top < source_bottom:
        canvas[
            source_top - top : source_bottom - top,
            source_left - left : source_right - left,
        ] = image[source_top:source_bottom, source_left:source_right]
    if side == output_size:
        return canvas
    return cv2.resize(canvas, (output_size, output_size), interpolation=cv2.INTER_AREA)


def encode_png(image: np.ndarray, compression: int) -> bytes:
    encoded, png = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, compression])
    if not encoded:
        raise ValueError("Could not encode PNG image")
    return png.tobytes()


def pack_value(
    future_box: tuple[float, float, float, float] | list[float] | np.ndarray,
    crop_mode: int,
    crop_bbox: tuple[int, int, int, int],
    foveal_png: bytes,
) -> bytes:
    header = VALUE_HEADER.pack(
        *future_box,
        crop_mode,
        *crop_bbox,
        len(foveal_png),
    )
    return header + foveal_png


def unpack_value(value: bytes | memoryview) -> EncodedDirectSample:
    if len(value) <= VALUE_HEADER.size:
        raise ValueError(f"Invalid direct-image value: {len(value)} bytes")
    unpacked = VALUE_HEADER.unpack_from(value)
    image_size = int(unpacked[9])
    payload_size = len(value) - VALUE_HEADER.size
    if image_size <= 0 or image_size != payload_size:
        raise ValueError(f"Invalid direct-image payload length: image={image_size}, payload={payload_size}")
    return EncodedDirectSample(
        future_interaction_box=np.asarray(unpacked[:4], dtype=np.float32),
        crop_mode=int(unpacked[4]),
        crop_bbox=tuple(int(item) for item in unpacked[5:9]),
        foveal_png=bytes(value[VALUE_HEADER.size :]),
    )


def decode_png(png: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode PNG image")
    return image


def _encode_view(
    annotation_key: str,
    rows: list[dict[str, object]],
    episode: dict[str, object],
    args: argparse.Namespace,
) -> EncodedView:
    capture = cv2.VideoCapture(str(episode["video_path"]))
    shard = store_shard(annotation_key, args.store_shards)
    if not capture.isOpened():
        return EncodedView(annotation_key, shard, [], [annotation_key])

    wanted = {int(row["frame_idx"]): row for row in rows}
    samples: list[tuple[str, bytes, dict[str, object], int]] = []
    video_failures: list[str] = []
    for frame_idx in range(max(wanted) + 1):
        if not capture.grab():
            video_failures.append(f"{annotation_key}:ended_at_{frame_idx}")
            break
        row = wanted.get(frame_idx)
        if row is None:
            continue
        ok, image = capture.retrieve()
        if not ok:
            video_failures.append(f"{annotation_key}:decode_{frame_idx}")
            continue
        if image.shape[:2] != (720, 1280):
            raise ValueError(f"Expected 1280x720 raw frame, got {image.shape[1]}x{image.shape[0]}")

        key = sample_key(annotation_key, frame_idx)
        crop_mode, crop_bbox = choose_crop(image.shape, row["interaction_box"], key, args)
        foveal_image = (
            resize_with_pad(image, args.output_size)
            if crop_mode == CROP_MODE_FULL
            else render_square_crop(image, crop_bbox, args.output_size)
        )
        foveal_png = encode_png(foveal_image, args.png_compression)
        samples.append(
            (
                key,
                pack_value(row["future_interaction_box"], crop_mode, crop_bbox, foveal_png),
                {
                    "sample_key": key,
                    "annotation_key": annotation_key,
                    "ori_path": episode["ori_path"],
                    "view": episode["view"],
                    "frame_idx": frame_idx,
                    "primitive_skill": row["primitive_skill"],
                    "interaction_box": row["interaction_box"],
                    "future_interaction_box": row["future_interaction_box"],
                    "crop_mode": crop_mode,
                    "crop_bbox": crop_bbox,
                    "foveal_png_bytes": len(foveal_png),
                    "store_shard": shard,
                },
                crop_mode,
            )
        )
    capture.release()
    return EncodedView(annotation_key, shard, samples, video_failures)


def build_direct_repack(args: argparse.Namespace) -> dict[str, object]:
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.view is not None and args.paired_views:
        raise ValueError("view filtering cannot be combined with paired views")
    if args.full_image_probability + args.jitter_probability > 1.0:
        raise ValueError("full-image and jitter probabilities must not exceed 1 in total")
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    (output_dir / "_INCOMPLETE").write_text("direct 224 repack in progress\n")

    sidecar = Path(args.sidecar)
    episode_table = pq.read_table(sidecar / "episodes")
    ok_episodes = episode_table.filter(pc.equal(episode_table["status"], "ok"))
    if args.view is not None:
        ok_episodes = ok_episodes.filter(pc.equal(ok_episodes["view"], args.view))
    annotation_index: dict[str, tuple[str, int]] | None = None
    frame_bits: dict[str, bytearray] | None = None
    paired_ranges: dict[str, list[list[int]]] | None = None
    if args.paired_views:
        metadata, annotation_index, frame_bits, paired_ranges = _build_paired_frame_index(sidecar, ok_episodes)
        ok_episodes = pa.Table.from_pylist(list(metadata.values()), schema=episode_table.schema)
        (output_dir / "paired_sample_ranges.json").write_text(
            json.dumps(paired_ranges, sort_keys=True, separators=(",", ":")) + "\n"
        )
    else:
        metadata = {str(row["annotation_key"]): row for row in ok_episodes.to_pylist()}
    ok_keys = pa.array(list(metadata), type=pa.string())
    pq.write_table(ok_episodes, output_dir / "episodes.parquet", compression="zstd")

    rlds_paired_ranges: dict[str, list[list[int]]] | None = None
    if paired_ranges is not None and args.rlds_filter is not None:
        source_ranges = json.loads(Path(args.rlds_filter).read_text())
        rlds_paired_ranges = {
            key: intersection
            for key, ranges in paired_ranges.items()
            if key in source_ranges and (intersection := _intersect_ranges(ranges, source_ranges[key]))
        }
        (output_dir / "paired_rlds_sample_ranges.json").write_text(
            json.dumps(rlds_paired_ranges, sort_keys=True, separators=(",", ":")) + "\n"
        )

    images_dir = output_dir / "direct_images"
    images_dir.mkdir()
    environments = [
        lmdb.open(
            str(images_dir / f"shard-{index:03d}.lmdb"),
            map_size=args.map_size_gb_per_shard * 1024**3,
            subdir=True,
        )
        for index in range(args.store_shards)
    ]
    transactions = [environment.begin(write=True) for environment in environments]
    pending = [0] * args.store_shards
    manifest_writer = ShardedParquetWriter(
        output_dir / "manifest",
        MANIFEST_SCHEMA,
        args.manifest_rows_per_shard,
    )

    input_rows = 0
    stored_samples = 0
    scheduled_views = 0
    processed_views = 0
    video_failures: list[str] = []
    mode_counts = {CROP_MODE_FULL: 0, CROP_MODE_EXACT: 0, CROP_MODE_JITTER: 0}
    current_key: str | None = None
    current_rows: list[dict[str, object]] = []
    start = time.perf_counter()

    def consume_view(result: EncodedView) -> None:
        nonlocal stored_samples, processed_views, transactions
        processed_views += 1
        video_failures.extend(result.video_failures)
        output_rows: list[dict[str, object]] = []
        for key, value, manifest_row, crop_mode in result.samples:
            transactions[result.store_shard].put(key.encode(), value)
            pending[result.store_shard] += 1
            stored_samples += 1
            mode_counts[crop_mode] += 1
            output_rows.append(manifest_row)
            if pending[result.store_shard] >= args.commit_every:
                transactions[result.store_shard].commit()
                transactions[result.store_shard] = environments[result.store_shard].begin(write=True)
                pending[result.store_shard] = 0
        manifest_writer.extend(output_rows)
        if processed_views % args.progress_every == 0:
            elapsed = time.perf_counter() - start
            print(
                f"views={processed_views} samples={stored_samples} rate={stored_samples / elapsed:.1f}_samples/s",
                flush=True,
            )

    executor = (
        concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=multiprocessing.get_context("spawn"),
        )
        if args.workers > 1
        else None
    )
    futures: set[concurrent.futures.Future[EncodedView]] = set()

    def drain_completed() -> None:
        if not futures:
            return
        done, not_done = concurrent.futures.wait(
            futures,
            return_when=concurrent.futures.FIRST_COMPLETED,
        )
        futures.clear()
        futures.update(not_done)
        for future in done:
            consume_view(future.result())

    def schedule_view(annotation_key: str, rows: list[dict[str, object]]) -> None:
        nonlocal scheduled_views
        if args.limit_views is not None and scheduled_views >= args.limit_views:
            return
        scheduled_views += 1
        if executor is None:
            consume_view(_encode_view(annotation_key, rows, metadata[annotation_key], args))
            return
        futures.add(executor.submit(_encode_view, annotation_key, rows, metadata[annotation_key], args))
        if len(futures) >= args.workers * 2:
            drain_completed()

    frame_columns = [
        "annotation_key",
        "frame_idx",
        "primitive_skill",
        "interaction_box",
        "future_interaction_box",
        "training_pair_valid",
    ]
    stop = False
    for part in sorted((sidecar / "frames").glob("part-*.parquet")):
        table = read_frame_table(part, frame_columns)
        mask = pc.and_(table["training_pair_valid"], pc.is_in(table["annotation_key"], value_set=ok_keys))
        table = table.filter(mask).drop(["training_pair_valid"])
        for row in table.to_pylist():
            annotation_key = str(row["annotation_key"])
            if annotation_index is not None and frame_bits is not None:
                ori_path, _ = annotation_index[annotation_key]
                frame_idx = int(row["frame_idx"])
                if frame_idx >= len(frame_bits[ori_path]) or frame_bits[ori_path][frame_idx] != 3:
                    continue
            input_rows += 1
            if current_key is None:
                current_key = annotation_key
            if annotation_key != current_key:
                schedule_view(current_key, current_rows)
                current_key = annotation_key
                current_rows = []
                if args.limit_views is not None and scheduled_views >= args.limit_views:
                    stop = True
                    break
            current_rows.append(row)
        if stop:
            break
    if not stop and current_key is not None and current_rows:
        schedule_view(current_key, current_rows)

    while futures:
        drain_completed()
    if executor is not None:
        executor.shutdown()

    for index, transaction in enumerate(transactions):
        transaction.commit()
        environments[index].sync()
        environments[index].close()
    manifest_writer.close()

    shard_bytes = [
        (images_dir / f"shard-{index:03d}.lmdb" / "data.mdb").stat().st_size
        for index in range(args.store_shards)
    ]
    index = {
        "format_version": FORMAT_VERSION,
        "sample_key": "{annotation_key}--{frame_idx}",
        "store_shard": "crc32(annotation_key) % store_shards",
        "store_shards": args.store_shards,
        "image_encoding": "PNG compression 1; OpenCV BGR; consumer converts decoded images to RGB",
        "value_header": {
            "struct": VALUE_HEADER.format,
            "bytes": VALUE_HEADER.size,
            "fields": [
                "future_box_x1",
                "future_box_y1",
                "future_box_x2",
                "future_box_y2",
                "crop_mode",
                "crop_left",
                "crop_top",
                "crop_right",
                "crop_bottom",
                "foveal_png_bytes",
            ],
        },
        "payload": "foveal-crop PNG; full crop mode stores the resized full frame in the same slot",
        "crop_modes": {"1": "full", "2": "exact", "3": "jitter"},
    }
    (output_dir / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")

    elapsed = time.perf_counter() - start
    summary: dict[str, object] = {
        "format_version": FORMAT_VERSION,
        "sidecar": str(sidecar.resolve()),
        "view": args.view,
        "available_episode_views": len(metadata),
        "processed_views": processed_views,
        "input_training_pair_rows": input_rows,
        "stored_samples": stored_samples,
        "video_failures": video_failures,
        "output_size": args.output_size,
        "context_scale": args.context_scale,
        "full_image_probability": args.full_image_probability,
        "jitter_probability": args.jitter_probability,
        "jitter_center_fraction": args.jitter_center_fraction,
        "jitter_scale_min": args.jitter_scale_min,
        "jitter_scale_max": args.jitter_scale_max,
        "png_compression": args.png_compression,
        "workers": args.workers,
        "crop_mode_counts": {
            "full": mode_counts[CROP_MODE_FULL],
            "exact": mode_counts[CROP_MODE_EXACT],
            "jitter": mode_counts[CROP_MODE_JITTER],
        },
        "store_shards": args.store_shards,
        "shard_bytes": shard_bytes,
        "total_lmdb_bytes": sum(shard_bytes),
        "elapsed_seconds": elapsed,
        "samples_per_second": stored_samples / elapsed if elapsed else None,
    }
    if paired_ranges is not None:
        summary.update(
            {
                "paired_episodes": len(paired_ranges),
                "paired_frames": sum(_range_count(ranges) for ranges in paired_ranges.values()),
                "rlds_paired_episodes": len(rlds_paired_ranges) if rlds_paired_ranges is not None else None,
                "rlds_paired_frames": (
                    sum(_range_count(ranges) for ranges in rlds_paired_ranges.values())
                    if rlds_paired_ranges is not None
                    else None
                ),
            }
        )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if video_failures:
        raise RuntimeError(f"Direct repack encountered {len(video_failures)} video failures")
    (output_dir / "_INCOMPLETE").unlink()
    (output_dir / "_SUCCESS").write_text("direct 224 repack completed\n")
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the direct 224 foveal-crop cache.")
    parser.add_argument("--sidecar", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-size", type=int, default=224)
    parser.add_argument("--context-scale", type=float, default=1.5)
    parser.add_argument("--full-image-probability", type=float, default=0.10)
    parser.add_argument("--jitter-probability", type=float, default=0.45)
    parser.add_argument("--jitter-center-fraction", type=float, default=0.25)
    parser.add_argument("--jitter-scale-min", type=float, default=1.0)
    parser.add_argument("--jitter-scale-max", type=float, default=4 / 3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--png-compression", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--store-shards", type=int, default=16)
    parser.add_argument("--map-size-gb-per-shard", type=int, default=256)
    parser.add_argument("--commit-every", type=int, default=1000)
    parser.add_argument("--manifest-rows-per-shard", type=int, default=100_000)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--limit-views", type=int)
    parser.add_argument("--view", choices=("exterior_image_1_left", "exterior_image_2_left"))
    parser.add_argument("--paired-views", action="store_true")
    parser.add_argument("--rlds-filter")
    return parser


def main() -> None:
    summary = build_direct_repack(_parser().parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
