from __future__ import annotations

from collections.abc import Iterable
import concurrent.futures
import dataclasses
import hashlib
import json
from pathlib import Path
import struct
import zlib

import lmdb
import numpy as np
import pyarrow.parquet as pq

from foveavla.shared import stereo_geometry_manifest
from foveavla.training import droid_stereo_geometry_store

FORMAT_VERSION = "droid-camera-pointmap-v1"
PROVENANCE_FORMAT_VERSION = "droid-camera-pointmap-provenance-v1"
POINTMAP_SHAPE = (224, 224, 3)
QUALITY_SHAPE = (224, 224)
DEPTH_SHAPE = QUALITY_SHAPE
DEPTH_DTYPE = np.dtype("<f2")
QUALITY_DTYPE = np.dtype("u1")
VALUE_HEADER = struct.Struct("<b3x4i")
RAW_VALUE_BYTES = VALUE_HEADER.size + 2 * (
    int(np.prod(DEPTH_SHAPE)) * DEPTH_DTYPE.itemsize + int(np.prod(QUALITY_SHAPE)) * QUALITY_DTYPE.itemsize
)


def _is_supported_depth_source(depth_source: dict[str, object]) -> bool:
    return (
        depth_source.get("sensor") == "ZED stereo SVO rectified pair"
        and depth_source.get("estimator") == "Fast-FoundationStereo"
        and depth_source.get("alignment") == "left"
        and depth_source.get("metric_conversion") == "fx_times_baseline_over_disparity"
        and depth_source.get("range_m") == [0.2, 3.0]
        and isinstance(depth_source.get("checkpoint_sha256"), str)
        and len(depth_source["checkpoint_sha256"]) == 64
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclasses.dataclass(frozen=True)
class PointmapSample:
    sample_key: str
    crop_mode: int
    crop_bbox: np.ndarray
    depth_full: np.ndarray
    depth_crop: np.ndarray
    quality_full: np.ndarray
    quality_crop: np.ndarray


def encode_depth_pair(
    depth_full: np.ndarray,
    depth_crop: np.ndarray,
    quality_full: np.ndarray,
    quality_crop: np.ndarray,
    crop_mode: int,
    crop_bbox: np.ndarray,
) -> bytes:
    depths = np.stack((depth_full, depth_crop))
    quality = np.stack((quality_full, quality_crop))
    crop_bbox = np.asarray(crop_bbox)
    if depths.shape != (2, *DEPTH_SHAPE):
        raise ValueError(f"Expected two depth maps with shape {DEPTH_SHAPE}, got {depths.shape}")
    if quality.shape != (2, *QUALITY_SHAPE):
        raise ValueError(f"Expected two quality maps with shape {QUALITY_SHAPE}, got {quality.shape}")
    if crop_bbox.shape != (4,):
        raise ValueError(f"Expected crop_bbox shape (4,), got {crop_bbox.shape}")
    if not np.all(np.isfinite(depths)) or not np.all(np.isfinite(quality)):
        raise ValueError("Depth and quality maps must be finite")
    if np.any(depths < 0.0):
        raise ValueError("Metric depth must be non-negative")
    if np.any((quality < 0.0) | (quality > 1.0)):
        raise ValueError("Pointmap quality must be in [0, 1]")
    raw = VALUE_HEADER.pack(int(crop_mode), *(int(value) for value in crop_bbox))
    raw += depths.astype(DEPTH_DTYPE, copy=False).tobytes(order="C")
    raw += np.rint(quality * 255.0).astype(QUALITY_DTYPE).tobytes(order="C")
    return zlib.compress(raw, level=1)


def decode_pointmap_pair(
    sample_key: str,
    value: bytes,
) -> PointmapSample:
    try:
        raw = zlib.decompress(value)
    except zlib.error as error:
        raise ValueError(f"Invalid compressed pointmap value for {sample_key}") from error
    if len(raw) != RAW_VALUE_BYTES:
        raise ValueError(f"Invalid pointmap value size for {sample_key}: expected {RAW_VALUE_BYTES}, got {len(raw)}")
    crop_mode, *crop_bbox = VALUE_HEADER.unpack_from(raw)
    depth_bytes = 2 * int(np.prod(DEPTH_SHAPE)) * DEPTH_DTYPE.itemsize
    depths = np.frombuffer(raw, dtype=DEPTH_DTYPE, count=2 * int(np.prod(DEPTH_SHAPE)), offset=VALUE_HEADER.size)
    depths = depths.reshape(2, *DEPTH_SHAPE).astype(np.float32)
    quality_offset = VALUE_HEADER.size + depth_bytes
    quality = (
        np.frombuffer(raw, dtype=QUALITY_DTYPE, offset=quality_offset).reshape(2, *QUALITY_SHAPE).astype(np.float32)
        / 255.0
    )
    depth_full = depths[0]
    depth_crop = depths[0] if crop_mode == 1 else depths[1]
    quality_full = quality[0]
    quality_crop = quality[0] if crop_mode == 1 else quality[1]
    return PointmapSample(
        sample_key=sample_key,
        crop_mode=int(crop_mode),
        crop_bbox=np.asarray(crop_bbox, dtype=np.int32),
        depth_full=depth_full,
        depth_crop=depth_crop,
        quality_full=quality_full,
        quality_crop=quality_crop,
    )


class PointmapStore:
    """Read metric depth from the retained DROID pointmap sidecar format."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        if not (self.root / "_SUCCESS").is_file() or (self.root / "_INCOMPLETE").exists():
            raise RuntimeError(f"Pointmap sidecar is not complete: {self.root}")

        index_path = self.root / "index.json"
        summary_path = self.root / "summary.json"
        episodes_path = self.root / "episodes.parquet"
        index = json.loads(index_path.read_text())
        if index.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"Unsupported pointmap format in {self.root / 'index.json'}")
        self.exterior_view = str(index.get("exterior_view", ""))
        if self.exterior_view not in stereo_geometry_manifest.EXTERIOR_VIEWS:
            raise ValueError(f"Unsupported pointmap exterior view in {self.root / 'index.json'}")
        pointmap = index.get("pointmap", {})
        quality = index.get("quality", {})
        depth_source = index.get("depth_source", {})
        value = index.get("value", {})
        if (
            pointmap.get("shape") != list(POINTMAP_SHAPE)
            or pointmap.get("model_dtype") != "float32"
            or pointmap.get("frame") != "zed_left_camera"
            or float(pointmap.get("normalization_scale_m", 0.0)) <= 0
            or pointmap.get("storage") != "aligned_metric_depth_plus_camera_model"
            or pointmap.get("depth_dtype") != DEPTH_DTYPE.str
            or quality.get("shape") != list(QUALITY_SHAPE)
            or quality.get("storage_dtype") != QUALITY_DTYPE.str
            or value.get("compression") != "zlib"
            or int(value.get("raw_bytes", -1)) != RAW_VALUE_BYTES
            or not _is_supported_depth_source(depth_source)
        ):
            raise ValueError(f"Unsupported pointmap layout in {self.root / 'index.json'}")
        self.depth_source = dict(depth_source)
        source = index.get("source", {})
        self.foveal_repack_dir = str(source.get("foveal_repack_dir", source.get("roi_repack_dir", "")))
        self._foveal_repack_fingerprint = source.get("foveal_repack_fingerprint", source.get("roi_repack_fingerprint"))
        self._alignment_audit = source.get("alignment_audit")
        if not self.foveal_repack_dir or not isinstance(self._foveal_repack_fingerprint, dict):
            raise ValueError(f"Pointmap source foveal repack is missing in {self.root / 'index.json'}")
        if not isinstance(self._alignment_audit, dict):
            raise ValueError(f"Pointmap alignment audit is missing in {self.root / 'index.json'}")
        selected_episodes = int(source.get("selected_episodes", -1))
        selected_annotation_keys_sha256 = source.get("selected_annotation_keys_sha256")
        provenance_path = self.root / "provenance.json"
        self.provenance = json.loads(provenance_path.read_text()) if provenance_path.is_file() else None
        if self.provenance is not None:
            if self.provenance.get("format_version") != PROVENANCE_FORMAT_VERSION:
                raise ValueError(f"Unsupported pointmap provenance in {provenance_path}")
            provenance_data = self.provenance.get("data", {})
            expected_hashes = {
                "index_sha256": index_path,
                "summary_sha256": summary_path,
                "episodes_parquet_sha256": episodes_path,
            }
            if any(provenance_data.get(key) != _sha256_file(path) for key, path in expected_hashes.items()):
                raise ValueError(f"Pointmap provenance data hashes do not match {self.root}")
        expanded_exclusions = source.get("exclusions", {}).get("expanded", [])
        self.excluded_annotation_keys = (
            frozenset(str(row["annotation_key"]) for row in expanded_exclusions)
            | frozenset(self._alignment_audit.get("excluded_annotation_keys", ()))
            | frozenset(self._alignment_audit.get("ambiguous_annotation_keys", ()))
        )
        self._shard_count = int(index["store_shards"])

        identity_columns = ["annotation_key", "ori_path", "view"]
        episode_table = pq.read_table(episodes_path, columns=identity_columns)
        self._annotation_keys: dict[tuple[str, str], str] = {}
        for row in episode_table.to_pylist():
            join_key = (str(row["ori_path"]), str(row["view"]))
            if join_key[1] != self.exterior_view:
                raise ValueError(f"Pointmap episode view does not match sidecar index: {join_key[1]}")
            if join_key in self._annotation_keys:
                raise ValueError(f"Duplicate pointmap episode-view mapping: {join_key}")
            annotation_key = str(row["annotation_key"])
            self._annotation_keys[join_key] = annotation_key
        self.annotation_keys = frozenset(self._annotation_keys.values())
        if selected_episodes != len(self.annotation_keys):
            raise ValueError("Selected episode count does not match the pointmap sidecar")
        if selected_annotation_keys_sha256 != droid_stereo_geometry_store.annotation_keys_sha256(self.annotation_keys):
            raise ValueError("Selected episode fingerprint does not match the pointmap sidecar")
        if self.provenance is not None:
            provenance_data = self.provenance["data"]
            if provenance_data.get("selected_annotation_keys_sha256") != selected_annotation_keys_sha256 or int(
                provenance_data.get("episodes", -1)
            ) != len(self.annotation_keys):
                raise ValueError("Pointmap provenance episode identity does not match the sidecar")

        pointmap_dir = self.root / "pointmaps"
        self._environments = [
            lmdb.open(
                str(pointmap_dir / f"shard-{shard:03d}.lmdb"),
                readonly=True,
                lock=False,
                readahead=False,
                max_readers=32,
            )
            for shard in range(self._shard_count)
        ]
        self._transactions = [environment.begin() for environment in self._environments]
        self.sample_count = sum(environment.stat()["entries"] for environment in self._environments)
        if self.provenance is not None and int(self.provenance["data"].get("samples", -1)) != self.sample_count:
            self.close()
            raise ValueError("Pointmap provenance sample count does not match the sidecar")

    def validate_foveal_repack(self, root: str | Path) -> None:
        root = Path(root)
        if Path(self.foveal_repack_dir).resolve() != root.resolve():
            raise ValueError(f"Pointmaps were built from a different foveal repack: {self.foveal_repack_dir} != {root}")
        if droid_stereo_geometry_store.foveal_repack_fingerprint(root) != self._foveal_repack_fingerprint:
            raise ValueError("Pointmap foveal metadata fingerprint does not match the configured foveal repack")

    def validate_alignment_audit(self) -> None:
        if (
            droid_stereo_geometry_store.alignment_audit_attestation(self._alignment_audit["root"])
            != self._alignment_audit
        ):
            raise ValueError("Pointmap alignment audit attestation has changed")

    def is_expected_exclusion(self, sample_key: str) -> bool:
        annotation_key, separator, _ = sample_key.rpartition("--")
        if not separator:
            raise ValueError(f"Invalid pointmap sample key: {sample_key}")
        return annotation_key in self.excluded_annotation_keys

    def expects_sample(self, sample_key: str) -> bool:
        annotation_key, separator, _ = sample_key.rpartition("--")
        if not separator:
            raise ValueError(f"Invalid pointmap sample key: {sample_key}")
        return annotation_key in self.annotation_keys and annotation_key not in self.excluded_annotation_keys

    def lookup_batch(
        self,
        episode_paths: Iterable[object],
        exterior_views: Iterable[object],
        frame_indices: Iterable[object],
        *,
        executor: concurrent.futures.Executor | None = None,
    ) -> list[PointmapSample | None]:
        resolved: list[PointmapSample | None] = []
        grouped: dict[int, list[tuple[int, str]]] = {}
        for episode_path, exterior_view, frame_idx in zip(episode_paths, exterior_views, frame_indices, strict=True):
            join_key = (_as_text(episode_path), _as_text(exterior_view))
            annotation_key = self._annotation_keys.get(join_key)
            resolved.append(None)
            if annotation_key is None:
                continue
            sample_key = f"{annotation_key}--{int(frame_idx)}"
            shard = zlib.crc32(annotation_key.encode()) % self._shard_count
            grouped.setdefault(shard, []).append((len(resolved) - 1, sample_key))
        if executor is not None:
            work = ((self._environments[shard], items) for shard, items in grouped.items())
            for decoded in executor.map(_read_and_decode_shard, work):
                for index, sample in decoded:
                    resolved[index] = sample
            return resolved

        encoded: list[tuple[int, str, bytes]] = []
        for shard, items in grouped.items():
            transaction = self._transactions[shard]
            for index, sample_key in items:
                value = transaction.get(sample_key.encode())
                if value is not None:
                    encoded.append((index, sample_key, value))
        for index, sample in map(_decode_lookup_item, encoded):
            resolved[index] = sample
        return resolved

    def close(self) -> None:
        for transaction in self._transactions:
            transaction.abort()
        for environment in self._environments:
            environment.close()


def _as_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return _as_text(value.item())
    return str(value)


def _decode_lookup_item(
    item: tuple[int, str, bytes],
) -> tuple[int, PointmapSample]:
    index, sample_key, value = item
    return index, decode_pointmap_pair(sample_key, value)


def _read_and_decode_shard(
    item: tuple[lmdb.Environment, list[tuple[int, str]]],
) -> list[tuple[int, PointmapSample]]:
    environment, items = item
    decoded = []
    with environment.begin() as transaction:
        for index, sample_key in items:
            value = transaction.get(sample_key.encode())
            if value is not None:
                decoded.append(_decode_lookup_item((index, sample_key, value)))
    return decoded
