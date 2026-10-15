"""Frame keys and paired-view filters shared by direct foveal preparation."""

from __future__ import annotations

import collections
from pathlib import Path
import zlib

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

BOX_TYPE = pa.list_(pa.float32())

_LEFT_VIEW = "exterior_image_1_left"
_RIGHT_VIEW = "exterior_image_2_left"


def store_shard(annotation_key: str, shard_count: int) -> int:
    return zlib.crc32(annotation_key.encode()) % shard_count


def sample_key(annotation_key: str, frame_idx: int) -> str:
    return f"{annotation_key}--{frame_idx}"


def _build_paired_frame_index(
    sidecar: Path,
    ok_episodes: pa.Table,
) -> tuple[dict[str, dict[str, object]], dict[str, tuple[str, int]], dict[str, bytearray], dict[str, list[list[int]]]]:
    episode_views: dict[str, dict[str, dict[str, object]]] = collections.defaultdict(dict)
    for row in ok_episodes.to_pylist():
        view = str(row["view"])
        if view in {_LEFT_VIEW, _RIGHT_VIEW}:
            episode_views[str(row["ori_path"])][view] = row

    metadata: dict[str, dict[str, object]] = {}
    annotation_index: dict[str, tuple[str, int]] = {}
    frame_bits: dict[str, bytearray] = {}
    for ori_path, views in episode_views.items():
        if set(views) != {_LEFT_VIEW, _RIGHT_VIEW}:
            continue
        last_frames = [views[view]["annotation_last_frame"] for view in (_LEFT_VIEW, _RIGHT_VIEW)]
        last_frame = max(int(value) if value is not None else -1 for value in last_frames)
        frame_bits[ori_path] = bytearray(last_frame + 1)
        for view, bit in ((_LEFT_VIEW, 1), (_RIGHT_VIEW, 2)):
            row = views[view]
            annotation_key = str(row["annotation_key"])
            metadata[annotation_key] = row
            annotation_index[annotation_key] = (ori_path, bit)

    eligible_keys = pa.array(list(annotation_index), type=pa.string())
    for part in sorted((sidecar / "frames").glob("part-*.parquet")):
        table = pq.read_table(part, columns=["annotation_key", "frame_idx", "training_pair_valid"])
        mask = pc.and_(table["training_pair_valid"], pc.is_in(table["annotation_key"], value_set=eligible_keys))
        table = table.filter(mask).drop(["training_pair_valid"])
        for annotation_key, frame_idx in zip(
            table["annotation_key"].to_pylist(),
            table["frame_idx"].to_pylist(),
            strict=True,
        ):
            ori_path, bit = annotation_index[str(annotation_key)]
            index = int(frame_idx)
            if index < len(frame_bits[ori_path]):
                frame_bits[ori_path][index] |= bit

    paired_ranges = {
        _rlds_episode_key(ori_path): _ranges_from_bitmap(bits)
        for ori_path, bits in frame_bits.items()
        if 3 in bits
    }
    return metadata, annotation_index, frame_bits, paired_ranges


def _ranges_from_bitmap(bits: bytearray) -> list[list[int]]:
    ranges: list[list[int]] = []
    start: int | None = None
    for index, value in enumerate(bits):
        if value == 3 and start is None:
            start = index
        elif value != 3 and start is not None:
            ranges.append([start, index])
            start = None
    if start is not None:
        ranges.append([start, len(bits)])
    return ranges


def _intersect_ranges(left: list[list[int]], right: list[list[int]]) -> list[list[int]]:
    result: list[list[int]] = []
    left_index = 0
    right_index = 0
    while left_index < len(left) and right_index < len(right):
        start = max(left[left_index][0], right[right_index][0])
        end = min(left[left_index][1], right[right_index][1])
        if start < end:
            result.append([start, end])
        if left[left_index][1] <= right[right_index][1]:
            left_index += 1
        else:
            right_index += 1
    return result


def _rlds_episode_key(ori_path: str) -> str:
    return f"{ori_path.rsplit('/', 1)[0]}/recordings/MP4--{ori_path}"


def _range_count(ranges: list[list[int]]) -> int:
    return sum(end - start for start, end in ranges)
