from __future__ import annotations

from collections.abc import Iterable, Iterator
import concurrent.futures
import dataclasses
import json
from pathlib import Path
import struct
import zlib

import cv2
import jax
import lmdb
import numpy as np
from foveavla_client import image_tools
import pyarrow.parquet as pq

from foveavla.training import droid_pointmap_store

_DIRECT_VALUE_HEADER = struct.Struct("<4fB3x4iI")
_LEFT_EXTERIOR_VIEW = "exterior_image_1_left"
_RIGHT_EXTERIOR_VIEW = "exterior_image_2_left"


@dataclasses.dataclass(frozen=True)
class EncodedFovealSample:
    sample_key: str
    box_in_crop: np.ndarray
    future_interaction_box: np.ndarray
    jpeg_bytes: bytes
    crop_mode: int | None = None
    crop_bbox: np.ndarray | None = None


@dataclasses.dataclass(frozen=True)
class FovealSample:
    sample_key: str
    box_in_crop: np.ndarray
    future_interaction_box: np.ndarray
    context_image: np.ndarray


class FovealStore:
    """Read-only access to a completed foveal crop repack."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        if not (self.root / "_SUCCESS").is_file() or (self.root / "_INCOMPLETE").exists():
            raise RuntimeError(f"foveal repack is not complete: {self.root}")

        index = json.loads((self.root / "index.json").read_text())
        if index.get("value_header", {}).get("struct") != _DIRECT_VALUE_HEADER.format:
            raise ValueError(f"Expected a direct foveal repack in {self.root / 'index.json'}")
        images_dir = self.root / "direct_images"
        self._shard_count = int(index["store_shards"])

        episode_table = pq.read_table(
            self.root / "episodes.parquet",
            columns=["annotation_key", "ori_path", "view"],
        )
        self._annotation_keys: dict[tuple[str, str], str] = {}
        for row in episode_table.to_pylist():
            join_key = (str(row["ori_path"]), str(row["view"]))
            if join_key in self._annotation_keys:
                raise ValueError(f"Duplicate foveal episode-view mapping: {join_key}")
            self._annotation_keys[join_key] = str(row["annotation_key"])

        self._environments = [
            lmdb.open(
                str(images_dir / f"shard-{shard:03d}.lmdb"),
                readonly=True,
                lock=False,
                readahead=False,
                max_readers=32,
            )
            for shard in range(self._shard_count)
        ]
        self._transactions = [environment.begin() for environment in self._environments]

    def lookup(self, episode_path: object, exterior_view: object, frame_idx: object) -> EncodedFovealSample | None:
        annotation_key = self._annotation_keys.get((_as_text(episode_path), _as_text(exterior_view)))
        if annotation_key is None:
            return None

        sample_key = f"{annotation_key}--{int(frame_idx)}"
        shard = zlib.crc32(annotation_key.encode()) % self._shard_count
        value = self._transactions[shard].get(sample_key.encode())
        if value is None:
            return None
        return _unpack_value(sample_key, value)

    def lookup_batch(
        self,
        episode_paths: Iterable[object],
        exterior_views: Iterable[object],
        frame_indices: Iterable[object],
        *,
        executor: concurrent.futures.Executor | None = None,
    ) -> list[EncodedFovealSample | None]:
        resolved: list[EncodedFovealSample | None] = []
        grouped: dict[int, list[tuple[int, str]]] = {}
        for episode_path, exterior_view, frame_idx in zip(episode_paths, exterior_views, frame_indices, strict=True):
            annotation_key = self._annotation_keys.get((_as_text(episode_path), _as_text(exterior_view)))
            resolved.append(None)
            if annotation_key is None:
                continue
            sample_key = f"{annotation_key}--{int(frame_idx)}"
            shard = zlib.crc32(annotation_key.encode()) % self._shard_count
            grouped.setdefault(shard, []).append((len(resolved) - 1, sample_key))

        if executor is not None:
            work = ((self._environments[shard], items) for shard, items in grouped.items())
            for samples in executor.map(_read_foveal_shard, work):
                for index, sample in samples:
                    resolved[index] = sample
            return resolved

        for shard, items in grouped.items():
            transaction = self._transactions[shard]
            for index, sample_key in items:
                value = transaction.get(sample_key.encode())
                if value is not None:
                    resolved[index] = _unpack_value(sample_key, value)
        return resolved

    def lookup_pair_batch(
        self,
        episode_paths: Iterable[object],
        frame_indices: Iterable[object],
        *,
        executor: concurrent.futures.Executor | None = None,
    ) -> list[tuple[EncodedFovealSample, EncodedFovealSample] | None]:
        flattened_episode_paths: list[object] = []
        flattened_views: list[str] = []
        flattened_frame_indices: list[object] = []
        for episode_path, frame_idx in zip(episode_paths, frame_indices, strict=True):
            flattened_episode_paths.extend((episode_path, episode_path))
            flattened_views.extend((_LEFT_EXTERIOR_VIEW, _RIGHT_EXTERIOR_VIEW))
            flattened_frame_indices.extend((frame_idx, frame_idx))

        flattened_samples = self.lookup_batch(
            flattened_episode_paths,
            flattened_views,
            flattened_frame_indices,
            executor=executor,
        )
        pairs: list[tuple[EncodedFovealSample, EncodedFovealSample] | None] = []
        for index in range(0, len(flattened_samples), 2):
            left, right = flattened_samples[index : index + 2]
            pairs.append(None if left is None or right is None else (left, right))
        return pairs

    def get(
        self,
        episode_path: object,
        exterior_view: object,
        frame_idx: object,
        *,
        output_size: int = 224,
    ) -> FovealSample | None:
        encoded = self.lookup(episode_path, exterior_view, frame_idx)
        return None if encoded is None else decode_foveal_sample(encoded, output_size=output_size)

    def close(self) -> None:
        for transaction in self._transactions:
            transaction.abort()
        for environment in self._environments:
            environment.close()


class DroidFovealBatchDataset:
    """Join DROID with foveal crops and metric depth, keeping finite-pass remainders."""

    def __init__(
        self,
        dataset: Iterable[dict],
        foveal_repack_dir: str | Path,
        *,
        batch_size: int,
        output_size: int = 224,
        decode_workers: int = 8,
        excluded_annotation_keys: Iterable[str] = (),
        pointmap_dir: str | Path | None = None,
        pointmap_aux_dirs: Iterable[str | Path] = (),
        dual_view: bool = False,
    ):
        if decode_workers < 1:
            raise ValueError("decode_workers must be at least 1")
        self._dataset = dataset
        self._store = FovealStore(foveal_repack_dir)
        self._pointmap_stores = ()
        try:
            pointmap_dirs = () if pointmap_dir is None else (pointmap_dir, *tuple(pointmap_aux_dirs))
            stores = []
            for path in pointmap_dirs:
                stores.append(droid_pointmap_store.PointmapStore(path))
                self._pointmap_stores = tuple(stores)
            pointmap_views = [store.exterior_view for store in self._pointmap_stores]
            if len(pointmap_views) != len(set(pointmap_views)):
                raise ValueError(f"Depth sidecars have duplicate exterior views: {pointmap_views}")
            self._validate_pointmap_crop_metadata = False
            for store in self._pointmap_stores:
                if Path(store.foveal_repack_dir).resolve() == Path(foveal_repack_dir).resolve():
                    store.validate_foveal_repack(foveal_repack_dir)
                else:
                    self._validate_pointmap_crop_metadata = True
        except Exception:
            for store in self._pointmap_stores:
                store.close()
            self._store.close()
            raise
        self._pointmap_store = self._pointmap_stores[0] if self._pointmap_stores else None
        self._batch_size = batch_size
        self._output_size = output_size
        self._excluded_annotation_keys = frozenset(excluded_annotation_keys)
        if "" in self._excluded_annotation_keys:
            raise ValueError("excluded_annotation_keys cannot contain an empty key")
        self._dual_view = dual_view
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=decode_workers)

    def __iter__(self) -> Iterator[dict]:
        accepted: list[dict] = []
        for batch in self._dataset:
            if self._dual_view:
                yield from self._join_dual_view_batch(batch, accepted)
                continue

            encoded_samples = self._store.lookup_batch(
                batch["episode_path"],
                batch["exterior_view"],
                batch["frame_idx"],
                executor=self._executor,
            )
            pointmap_samples = (
                self._lookup_pointmaps(
                    batch["episode_path"],
                    batch["exterior_view"],
                    batch["frame_idx"],
                )
                if self._pointmap_store is not None
                else [None] * len(encoded_samples)
            )
            present = []
            for index, (foveal_sample, pointmap_sample) in enumerate(
                zip(encoded_samples, pointmap_samples, strict=True)
            ):
                if foveal_sample is None or not self._is_allowed(foveal_sample):
                    continue
                if self._pointmap_store is not None:
                    if self._pointmap_store.is_expected_exclusion(foveal_sample.sample_key):
                        continue
                    if pointmap_sample is None:
                        if self._pointmap_store.expects_sample(foveal_sample.sample_key):
                            raise RuntimeError(f"Pointmap is unexpectedly missing for {foveal_sample.sample_key}")
                        continue
                    if foveal_sample.sample_key != pointmap_sample.sample_key:
                        raise ValueError(
                            f"foveal/pointmap sample key mismatch: {foveal_sample.sample_key} != {pointmap_sample.sample_key}"
                        )
                    self._validate_crop_alignment(foveal_sample, pointmap_sample)
                present.append((index, foveal_sample, pointmap_sample))
            rendered_samples = list(
                self._executor.map(
                    self._decode_and_render_context,
                    ((sample, batch["observation"]["image"][index]) for index, sample, _ in present),
                )
            )

            if not accepted and len(present) == self._batch_size == len(batch["frame_idx"]):
                output = dict(batch)
                output.pop("episode_path")
                output.pop("exterior_view")
                output.pop("frame_idx")
                output["observation"] = dict(batch["observation"])
                output["observation"]["image"] = np.stack(
                    [resized_image for _, _, resized_image in rendered_samples],
                    axis=0,
                )
                output["observation"]["foveal_image"] = np.stack(
                    [context_image for _, context_image, _ in rendered_samples],
                    axis=0,
                )
                output["future_interaction_box"] = np.stack(
                    [foveal_sample.future_interaction_box for foveal_sample, _, _ in rendered_samples],
                    axis=0,
                )
                output["future_interaction_box_mask"] = np.ones(self._batch_size, dtype=bool)
                if self._pointmap_store is not None:
                    output["observation"]["metric_depth_full"] = np.stack(
                        [sample.depth_full for _, _, sample in present],
                        axis=0,
                    )
                    output["observation"]["metric_depth_crop"] = np.stack(
                        [sample.depth_crop for _, _, sample in present],
                        axis=0,
                    )
                    output["observation"]["pointmap_quality_full"] = np.stack(
                        [sample.quality_full for _, _, sample in present],
                        axis=0,
                    )
                    output["observation"]["pointmap_quality_crop"] = np.stack(
                        [sample.quality_crop for _, _, sample in present],
                        axis=0,
                    )
                yield output
                continue

            for (index, _, pointmap_sample), (foveal_sample, context_image, resized_image) in zip(
                present, rendered_samples, strict=True
            ):
                sample = jax.tree.map(lambda value, index=index: value[index], batch)
                sample.pop("episode_path")
                sample.pop("exterior_view")
                sample.pop("frame_idx")
                sample["observation"] = dict(sample["observation"])
                sample["observation"]["image"] = resized_image
                sample["observation"]["foveal_image"] = context_image
                if pointmap_sample is not None:
                    sample["observation"]["metric_depth_full"] = pointmap_sample.depth_full
                    sample["observation"]["metric_depth_crop"] = pointmap_sample.depth_crop
                    sample["observation"]["pointmap_quality_full"] = pointmap_sample.quality_full
                    sample["observation"]["pointmap_quality_crop"] = pointmap_sample.quality_crop
                sample["future_interaction_box"] = foveal_sample.future_interaction_box
                sample["future_interaction_box_mask"] = np.True_
                accepted.append(sample)

                if len(accepted) == self._batch_size:
                    yield jax.tree.map(lambda *values: np.stack(values, axis=0), *accepted)
                    accepted.clear()

        if accepted:
            yield jax.tree.map(lambda *values: np.stack(values, axis=0), *accepted)

    def _join_dual_view_batch(self, batch: dict, accepted: list[dict]) -> Iterator[dict]:
        encoded_pairs = self._store.lookup_pair_batch(
            batch["episode_path"],
            batch["frame_idx"],
            executor=self._executor,
        )
        pointmap_pairs = self._lookup_dual_pointmaps(batch["episode_path"], batch["frame_idx"])
        present = []
        for index, (pair, pointmap_pair) in enumerate(zip(encoded_pairs, pointmap_pairs, strict=True)):
            if pair is None or not all(self._is_allowed(sample) for sample in pair):
                continue
            if self._pointmap_store is not None:
                if any(self._pointmap_is_expected_exclusion(sample.sample_key) for sample in pair):
                    continue
                if pointmap_pair is None:
                    if any(self._pointmap_expects_sample(sample.sample_key) for sample in pair):
                        raise RuntimeError(f"Pointmap is unexpectedly missing for dual sample {pair[0].sample_key}")
                    continue
                for foveal_sample, pointmap_sample in zip(pair, pointmap_pair, strict=True):
                    if foveal_sample.sample_key != pointmap_sample.sample_key:
                        raise ValueError(
                            f"foveal/pointmap sample key mismatch: {foveal_sample.sample_key} != {pointmap_sample.sample_key}"
                        )
                    self._validate_crop_alignment(foveal_sample, pointmap_sample)
            present.append((index, pair, pointmap_pair))
        decode_items = (
            (encoded_sample, batch["observation"][image_key][index])
            for index, pair, _ in present
            for encoded_sample, image_key in zip(pair, ("left_image", "right_image"), strict=True)
        )
        rendered_samples = iter(self._executor.map(self._decode_and_render_context, decode_items))
        rendered_pairs = [(next(rendered_samples), next(rendered_samples)) for _ in present]

        if (
            not accepted
            and self._pointmap_store is None
            and len(present) == self._batch_size == len(batch["frame_idx"])
        ):
            output = dict(batch)
            output.pop("episode_path")
            output.pop("frame_idx")
            output["observation"] = dict(batch["observation"])
            output["observation"]["left_image"] = np.stack([pair[0][2] for pair in rendered_pairs])
            output["observation"]["right_image"] = np.stack([pair[1][2] for pair in rendered_pairs])
            output["observation"]["left_foveal_image"] = np.stack([pair[0][1] for pair in rendered_pairs])
            output["observation"]["right_foveal_image"] = np.stack([pair[1][1] for pair in rendered_pairs])
            output["future_interaction_box"] = np.stack(
                [
                    np.stack((pair[0][0].future_interaction_box, pair[1][0].future_interaction_box))
                    for pair in rendered_pairs
                ]
            )
            output["future_interaction_box_mask"] = np.ones((self._batch_size, 2), dtype=bool)
            yield output
            return

        for (index, _, pointmap_pair), (left_rendered, right_rendered) in zip(present, rendered_pairs, strict=True):
            left_foveal_sample, left_context_image, left_image = left_rendered
            right_foveal_sample, right_context_image, right_image = right_rendered
            sample = jax.tree.map(lambda value, index=index: value[index], batch)
            sample.pop("episode_path")
            sample.pop("frame_idx")
            sample["observation"] = dict(sample["observation"])
            sample["observation"]["left_image"] = left_image
            sample["observation"]["right_image"] = right_image
            sample["observation"]["left_foveal_image"] = left_context_image
            sample["observation"]["right_foveal_image"] = right_context_image
            if pointmap_pair is not None:
                for view_name, pointmap_sample in zip(("left", "right"), pointmap_pair, strict=True):
                    sample["observation"][f"metric_depth_{view_name}_full"] = pointmap_sample.depth_full
                    sample["observation"][f"metric_depth_{view_name}_crop"] = pointmap_sample.depth_crop
                    sample["observation"][f"pointmap_quality_{view_name}_full"] = pointmap_sample.quality_full
                    sample["observation"][f"pointmap_quality_{view_name}_crop"] = pointmap_sample.quality_crop
            sample["future_interaction_box"] = np.stack(
                (left_foveal_sample.future_interaction_box, right_foveal_sample.future_interaction_box),
                axis=0,
            )
            sample["future_interaction_box_mask"] = np.ones(2, dtype=bool)
            accepted.append(sample)

            if len(accepted) == self._batch_size:
                yield jax.tree.map(lambda *values: np.stack(values, axis=0), *accepted)
                accepted.clear()

    def close(self) -> None:
        self._executor.shutdown()
        for pointmap_store in self._pointmap_stores:
            pointmap_store.close()
        self._store.close()

    def _lookup_pointmaps(
        self,
        episode_paths: Iterable[object],
        exterior_views: Iterable[object],
        frame_indices: Iterable[object],
    ) -> list[droid_pointmap_store.PointmapSample | None]:
        episode_paths = tuple(episode_paths)
        exterior_views = tuple(exterior_views)
        frame_indices = tuple(frame_indices)
        merged: list[droid_pointmap_store.PointmapSample | None] = [None] * len(episode_paths)
        for store in self._pointmap_stores:
            samples = store.lookup_batch(
                episode_paths,
                exterior_views,
                frame_indices,
                executor=self._executor,
            )
            for index, sample in enumerate(samples):
                if sample is None:
                    continue
                if merged[index] is not None:
                    raise ValueError(f"Multiple pointmap sidecars matched {sample.sample_key}")
                merged[index] = sample
        return merged

    def _lookup_dual_pointmaps(
        self,
        episode_paths: Iterable[object],
        frame_indices: Iterable[object],
    ) -> list[tuple[droid_pointmap_store.PointmapSample, droid_pointmap_store.PointmapSample] | None]:
        flattened_episode_paths = []
        flattened_views = []
        flattened_frame_indices = []
        for episode_path, frame_idx in zip(episode_paths, frame_indices, strict=True):
            flattened_episode_paths.extend((episode_path, episode_path))
            flattened_views.extend((_LEFT_EXTERIOR_VIEW, _RIGHT_EXTERIOR_VIEW))
            flattened_frame_indices.extend((frame_idx, frame_idx))
        flattened = self._lookup_pointmaps(
            flattened_episode_paths,
            flattened_views,
            flattened_frame_indices,
        )
        pairs = []
        for index in range(0, len(flattened), 2):
            left, right = flattened[index : index + 2]
            pairs.append(None if left is None or right is None else (left, right))
        return pairs

    def _pointmap_is_expected_exclusion(self, sample_key: str) -> bool:
        return any(store.is_expected_exclusion(sample_key) for store in self._pointmap_stores)

    def _pointmap_expects_sample(self, sample_key: str) -> bool:
        return any(store.expects_sample(sample_key) for store in self._pointmap_stores)

    def _validate_crop_alignment(
        self,
        foveal_sample: EncodedFovealSample,
        pointmap_sample: droid_pointmap_store.PointmapSample,
    ) -> None:
        if not self._validate_pointmap_crop_metadata:
            return
        if foveal_sample.crop_mode is None or foveal_sample.crop_bbox is None:
            raise ValueError(f"foveal crop metadata is unavailable for cross-repack join: {foveal_sample.sample_key}")
        if foveal_sample.crop_mode != pointmap_sample.crop_mode or not np.array_equal(
            foveal_sample.crop_bbox, pointmap_sample.crop_bbox
        ):
            raise ValueError(f"foveal/pointmap crop metadata mismatch for {foveal_sample.sample_key}")

    def _decode_and_render_context(
        self,
        item: tuple[EncodedFovealSample, np.ndarray],
    ) -> tuple[FovealSample, np.ndarray, np.ndarray]:
        encoded_sample, full_image = item
        resized_image = image_tools.resize_with_pad(full_image, self._output_size, self._output_size)
        foveal_sample = decode_foveal_sample(encoded_sample, output_size=self._output_size)
        return foveal_sample, foveal_sample.context_image, resized_image

    def _is_allowed(self, sample: EncodedFovealSample) -> bool:
        annotation_key, separator, _ = sample.sample_key.rpartition("--")
        if not separator:
            raise ValueError(f"Invalid foveal sample key: {sample.sample_key}")
        return annotation_key not in self._excluded_annotation_keys


def decode_foveal_sample(sample: EncodedFovealSample, *, output_size: int | None) -> FovealSample:
    image = cv2.imdecode(np.frombuffer(sample.jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode foveal PNG for {sample.sample_key}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if output_size is not None:
        image = _resize_context(image, output_size)
    return FovealSample(
        sample_key=sample.sample_key,
        box_in_crop=sample.box_in_crop,
        future_interaction_box=sample.future_interaction_box,
        context_image=image,
    )


def _resize_context(image: np.ndarray, output_size: int) -> np.ndarray:
    if image.shape[:2] == (output_size, output_size):
        return image
    return cv2.resize(image, (output_size, output_size), interpolation=cv2.INTER_AREA)


def _as_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return _as_text(value.item())
    return str(value)


def _unpack_value(sample_key: str, value: bytes) -> EncodedFovealSample:
    if len(value) <= _DIRECT_VALUE_HEADER.size:
        raise ValueError(f"Invalid direct foveal value for {sample_key}: {len(value)} bytes")
    header = _DIRECT_VALUE_HEADER.unpack_from(value)
    payload_size = len(value) - _DIRECT_VALUE_HEADER.size
    if int(header[9]) != payload_size:
        raise ValueError(f"Invalid direct foveal payload for {sample_key}: {payload_size} bytes")
    return EncodedFovealSample(
        sample_key=sample_key,
        box_in_crop=np.asarray((0.0, 0.0, 1.0, 1.0), dtype=np.float32),
        future_interaction_box=np.asarray(header[:4], dtype=np.float32),
        jpeg_bytes=bytes(value[_DIRECT_VALUE_HEADER.size :]),
        crop_mode=int(header[4]),
        crop_bbox=np.asarray(header[5:9], dtype=np.int32),
    )


def _read_foveal_shard(
    item: tuple[lmdb.Environment, list[tuple[int, str]]],
) -> list[tuple[int, EncodedFovealSample]]:
    environment, items = item
    samples = []
    with environment.begin() as transaction:
        for index, sample_key in items:
            value = transaction.get(sample_key.encode())
            if value is not None:
                samples.append((index, _unpack_value(sample_key, value)))
    return samples
