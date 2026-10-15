from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np

MANIFEST_FORMAT_VERSION = "droid-stereo-geometry-manifest-v1"
EXTERIOR_VIEW = "exterior_image_1_left"
EXTERIOR_VIEWS = (EXTERIOR_VIEW, "exterior_image_2_left")


@dataclasses.dataclass(frozen=True)
class GeometryManifestEpisode:
    annotation_key: str
    camera_serial: str
    svo_path: str
    video_path: str
    frame_idx: np.ndarray
    crop_mode: np.ndarray
    crop_bbox: np.ndarray


@dataclasses.dataclass(frozen=True)
class GeometryManifest:
    format_version: str
    exterior_view: str
    foveal_repack_dir: str
    shard_index: int
    shard_count: int
    episodes: tuple[GeometryManifestEpisode, ...]

    @property
    def sample_count(self) -> int:
        return sum(len(episode.frame_idx) for episode in self.episodes)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_geometry_manifest(path: str | Path) -> GeometryManifest:
    path = Path(path)
    if path.suffix == ".json":
        manifest = _load_json_manifest(path)
    elif path.suffix == ".npz":
        manifest = _load_npz_manifest(path)
    else:
        raise ValueError(f"Unsupported geometry manifest extension: {path}")
    _validate_manifest(manifest, path)
    return manifest


def write_geometry_manifest(path: str | Path, manifest: GeometryManifest) -> None:
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("Production geometry manifests must use the .npz extension")
    _validate_manifest(manifest, path)

    episode_offsets = np.zeros(len(manifest.episodes) + 1, dtype=np.int64)
    for index, episode in enumerate(manifest.episodes):
        episode_offsets[index + 1] = episode_offsets[index] + len(episode.frame_idx)
    frame_idx = np.concatenate([episode.frame_idx for episode in manifest.episodes])
    crop_mode = np.concatenate([episode.crop_mode for episode in manifest.episodes])
    crop_bbox = np.concatenate([episode.crop_bbox for episode in manifest.episodes])

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".partial")
    with temporary_path.open("wb") as file:
        np.savez_compressed(
            file,
            format_version=np.asarray(manifest.format_version),
            exterior_view=np.asarray(manifest.exterior_view),
            foveal_repack_dir=np.asarray(manifest.foveal_repack_dir),
            shard_index=np.asarray(manifest.shard_index, dtype=np.int32),
            shard_count=np.asarray(manifest.shard_count, dtype=np.int32),
            annotation_key=np.asarray([episode.annotation_key for episode in manifest.episodes]),
            camera_serial=np.asarray([episode.camera_serial for episode in manifest.episodes]),
            svo_path=np.asarray([episode.svo_path for episode in manifest.episodes]),
            video_path=np.asarray([episode.video_path for episode in manifest.episodes]),
            episode_offsets=episode_offsets,
            frame_idx=frame_idx,
            crop_mode=crop_mode,
            crop_bbox=crop_bbox,
        )
    temporary_path.replace(path)


def _scalar_text(value: np.ndarray) -> str:
    if value.shape != ():
        raise ValueError(f"expected scalar text metadata, got shape {value.shape}")
    return str(value.item())


def _load_json_manifest(path: Path) -> GeometryManifest:
    payload = json.loads(path.read_text())
    episodes = []
    for row in payload["episodes"]:
        samples = sorted(row["samples"], key=lambda sample: int(sample["frame_idx"]))
        episodes.append(
            GeometryManifestEpisode(
                annotation_key=str(row["annotation_key"]),
                camera_serial=str(row["camera_serial"]),
                svo_path=str(row["svo_path"]),
                video_path=str(row["video_path"]),
                frame_idx=np.asarray([sample["frame_idx"] for sample in samples], dtype=np.int32),
                crop_mode=np.asarray([sample["crop_mode"] for sample in samples], dtype=np.int8),
                crop_bbox=np.asarray([sample["crop_bbox"] for sample in samples], dtype=np.int32),
            )
        )
    return GeometryManifest(
        format_version=str(payload["format_version"]),
        exterior_view=str(payload["exterior_view"]),
        foveal_repack_dir=str(payload["foveal_repack_dir" if "foveal_repack_dir" in payload else "roi_repack_dir"]),
        shard_index=int(payload.get("shard_index", 0)),
        shard_count=int(payload.get("shard_count", 1)),
        episodes=tuple(episodes),
    )


def _load_npz_manifest(path: Path) -> GeometryManifest:
    with np.load(path, allow_pickle=False) as payload:
        offsets = payload["episode_offsets"]
        annotation_keys = payload["annotation_key"]
        camera_serials = payload["camera_serial"]
        svo_paths = payload["svo_path"]
        video_paths = payload["video_path"]
        frame_idx = payload["frame_idx"]
        crop_mode = payload["crop_mode"]
        crop_bbox = payload["crop_bbox"]
        episodes = []
        for index, annotation_key in enumerate(annotation_keys):
            start, end = int(offsets[index]), int(offsets[index + 1])
            episodes.append(
                GeometryManifestEpisode(
                    annotation_key=str(annotation_key),
                    camera_serial=str(camera_serials[index]),
                    svo_path=str(svo_paths[index]),
                    video_path=str(video_paths[index]),
                    frame_idx=frame_idx[start:end].copy(),
                    crop_mode=crop_mode[start:end].copy(),
                    crop_bbox=crop_bbox[start:end].copy(),
                )
            )
        return GeometryManifest(
            format_version=str(payload["format_version"]),
            exterior_view=str(payload["exterior_view"]),
            foveal_repack_dir=str(payload["foveal_repack_dir" if "foveal_repack_dir" in payload else "roi_repack_dir"]),
            shard_index=int(payload["shard_index"]),
            shard_count=int(payload["shard_count"]),
            episodes=tuple(episodes),
        )


def _validate_manifest(manifest: GeometryManifest, path: Path) -> None:
    if manifest.format_version != MANIFEST_FORMAT_VERSION:
        raise ValueError(f"Unsupported geometry manifest version in {path}: {manifest.format_version}")
    if manifest.exterior_view not in EXTERIOR_VIEWS:
        raise ValueError(f"Geometry manifest exterior view must be one of {EXTERIOR_VIEWS}")
    if manifest.shard_count < 1 or not 0 <= manifest.shard_index < manifest.shard_count:
        raise ValueError(f"Invalid shard {manifest.shard_index}/{manifest.shard_count} in {path}")
    if not manifest.episodes:
        raise ValueError(f"Geometry manifest has no episodes: {path}")

    annotation_keys = [episode.annotation_key for episode in manifest.episodes]
    if len(annotation_keys) != len(set(annotation_keys)):
        raise ValueError(f"Duplicate annotation keys in {path}")
    for episode in manifest.episodes:
        sample_count = len(episode.frame_idx)
        if sample_count < 1:
            raise ValueError(f"Episode has no samples: {episode.annotation_key}")
        if episode.frame_idx.shape != (sample_count,) or episode.frame_idx.dtype != np.int32:
            raise ValueError(f"Invalid frame_idx in {episode.annotation_key}")
        if np.any(episode.frame_idx[1:] <= episode.frame_idx[:-1]):
            raise ValueError(f"frame_idx must be strictly increasing in {episode.annotation_key}")
        if episode.crop_mode.shape != (sample_count,) or episode.crop_mode.dtype != np.int8:
            raise ValueError(f"Invalid crop_mode in {episode.annotation_key}")
        if not np.isin(episode.crop_mode, (1, 2, 3)).all():
            raise ValueError(f"Unsupported crop_mode in {episode.annotation_key}")
        if episode.crop_bbox.shape != (sample_count, 4) or episode.crop_bbox.dtype != np.int32:
            raise ValueError(f"Invalid crop_bbox in {episode.annotation_key}")
