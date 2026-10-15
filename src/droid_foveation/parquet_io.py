from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# Use a regular list rather than a fixed-size list.  PyArrow/Parquet currently
# materializes a null fixed-size list as an empty list and then rejects it on
# read ("Expected all lists to be of size=4").  Geometry validation still
# guarantees that every non-null box has exactly four coordinates.
BOX_TYPE = pa.list_(pa.float32())

# Read old annotation sidecars without rewriting their files or fingerprints.
_LEGACY_FRAME_FIELDS = {
    "interaction_box": "interaction_roi",
    "future_interaction_box": "future_interaction_roi",
    "box_valid": "roi_valid",
    "future_box_valid": "future_roi_valid",
}


def read_frame_table(path: str | Path, columns: list[str]) -> pa.Table:
    names = pq.read_schema(path).names
    stored_columns = [name if name in names else _LEGACY_FRAME_FIELDS.get(name, name) for name in columns]
    return pq.read_table(path, columns=stored_columns).rename_columns(columns)


FRAME_SCHEMA = pa.schema(
    [
        ("annotation_key", pa.string()),
        ("frame_idx", pa.int32()),
        ("primitive_skill", pa.string()),
        ("subtask", pa.string()),
        ("instruction", pa.string()),
        ("source_width", pa.int16()),
        ("source_height", pa.int16()),
        ("gripper_box", BOX_TYPE),
        ("object_box", BOX_TYPE),
        ("placement_box", BOX_TYPE),
        ("interaction_box", BOX_TYPE),
        ("box_valid", pa.bool_()),
        ("invalid_reason", pa.string()),
        ("future_frame_idx", pa.int32()),
        ("future_primitive_skill", pa.string()),
        ("future_interaction_box", BOX_TYPE),
        ("future_box_valid", pa.bool_()),
        ("training_pair_valid", pa.bool_()),
    ]
)

EPISODE_SCHEMA = pa.schema(
    [
        ("annotation_key", pa.string()),
        ("ori_path", pa.string()),
        ("view", pa.string()),
        ("local_h5_path", pa.string()),
        ("local_episode_path", pa.string()),
        ("metadata_path", pa.string()),
        ("camera_serial", pa.string()),
        ("video_path", pa.string()),
        ("annotation_frames", pa.int32()),
        ("annotation_first_frame", pa.int32()),
        ("annotation_last_frame", pa.int32()),
        ("video_frames", pa.int32()),
        ("video_width", pa.int16()),
        ("video_height", pa.int16()),
        ("hdf5_steps", pa.int32()),
        ("range_nop_start", pa.int32()),
        ("range_nop_end", pa.int32()),
        ("range_nop_count", pa.int32()),
        ("box_valid_frames", pa.int32()),
        ("future_eligible_frames", pa.int32()),
        ("future_valid_frames", pa.int32()),
        ("status", pa.string()),
        ("q_primitive_skill", pa.string()),
        ("q_object_box", pa.string()),
        ("q_placement_proposal", pa.string()),
        ("q_gripper_box", pa.string()),
    ]
)


class ShardedParquetWriter:
    """Writes bounded-size Parquet parts without holding a full run in memory."""

    def __init__(self, directory: str | Path, schema: pa.Schema, rows_per_shard: int):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.schema = schema
        self.rows_per_shard = rows_per_shard
        self._rows: list[dict[str, object]] = []
        self._part_index = 0
        self.total_rows = 0

    def extend(self, rows: list[dict[str, object]]) -> None:
        self._rows.extend(rows)
        while len(self._rows) >= self.rows_per_shard:
            self._write(self._rows[: self.rows_per_shard])
            del self._rows[: self.rows_per_shard]

    def close(self) -> None:
        if self._rows:
            self._write(self._rows)
            self._rows = []

    def _write(self, rows: list[dict[str, object]]) -> None:
        table = pa.Table.from_pylist(rows, schema=self.schema)
        output_path = self.directory / f"part-{self._part_index:05d}.parquet"
        pq.write_table(
            table,
            output_path,
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
        )
        del table
        pa.default_memory_pool().release_unused()
        self._part_index += 1
        self.total_rows += len(rows)
