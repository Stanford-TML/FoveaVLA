from __future__ import annotations

import csv
import dataclasses
import json
import pickle
from collections import OrderedDict
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import cv2
import h5py
import ijson
import lmdb

GCS_DROID_PREFIX = "gs://xembodiment_data/r2d2/r2d2-data-full/"
_CAMERA_SERIAL_FIELDS = ("ext1_cam_serial", "ext2_cam_serial", "wrist_cam_serial")


@dataclasses.dataclass(frozen=True)
class ResolvedEpisodeView:
    local_h5_path: str | None
    local_episode_path: str | None
    metadata_path: str | None
    camera_serial: str | None
    video_path: str | None
    video_frames: int | None
    video_width: int | None
    video_height: int | None
    hdf5_steps: int | None
    status: str


def iter_qsheet(path: str | Path) -> Iterator[tuple[str, dict[str, object]]]:
    with Path(path).open("rb") as stream:
        for key, value in ijson.kvitems(stream, ""):
            yield str(key), value


class RoboInterAnnotationDb:
    def __init__(self, path: str | Path):
        self._environment = lmdb.open(
            str(path),
            readonly=True,
            lock=False,
            readahead=False,
            max_readers=32,
        )
        self._transaction = self._environment.begin()

    def get(self, annotation_key: str) -> dict[int, dict[str, object]] | None:
        value = self._transaction.get(annotation_key.encode())
        if value is None:
            return None
        result = pickle.loads(value)
        if not isinstance(result, dict):
            raise TypeError(f"Expected dict for {annotation_key}, got {type(result).__name__}")
        return result

    def close(self) -> None:
        self._transaction.abort()
        self._environment.close()

    def __enter__(self) -> "RoboInterAnnotationDb":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class EpisodeResolver:
    def __init__(
        self,
        raw_droid_root: str | Path,
        cache_size: int = 4096,
        exterior_roles: str | Path | None = None,
    ):
        self.raw_droid_root = Path(raw_droid_root)
        self.cache_size = cache_size
        self._exterior_roles = _load_exterior_roles(exterior_roles)
        self._episode_cache: OrderedDict[
            str, tuple[Path | None, dict[str, object] | None, int | None]
        ] = OrderedDict()

    def resolve(self, qsheet_entry: Mapping[str, object]) -> ResolvedEpisodeView:
        ori_path = str(qsheet_entry.get("ori_path", ""))
        view = str(qsheet_entry.get("view", ""))
        return self._resolve_uncached(ori_path, view)

    def _resolve_uncached(self, ori_path: str, view: str) -> ResolvedEpisodeView:
        if not ori_path.startswith(GCS_DROID_PREFIX):
            return _unresolved("unsupported_ori_path")
        if view not in {"exterior_image_1_left", "exterior_image_2_left"}:
            return _unresolved("unsupported_view")

        local_h5 = self.raw_droid_root / ori_path.removeprefix(GCS_DROID_PREFIX)
        episode_path = local_h5.parent
        episode_root = ori_path.rsplit("/", 1)[0]
        exterior_roles = self._exterior_roles.get(episode_root)
        episode_cache_key = str(episode_path)
        if episode_cache_key in self._episode_cache:
            self._episode_cache.move_to_end(episode_cache_key)
        else:
            metadata_files = sorted(episode_path.glob("metadata*.json")) if episode_path.is_dir() else []
            if exterior_roles is not None:
                metadata_path = metadata_files[0] if metadata_files else None
                metadata = None
            else:
                metadata_path, metadata = read_episode_metadata(metadata_files)
            hdf5_steps = _read_hdf5_steps(local_h5) if local_h5.is_file() else None
            self._episode_cache[episode_cache_key] = (metadata_path, metadata, hdf5_steps)
            if len(self._episode_cache) > self.cache_size:
                self._episode_cache.popitem(last=False)

        metadata_path, metadata, hdf5_steps = self._episode_cache[episode_cache_key]
        if exterior_roles is None and metadata is None:
            status = "missing_metadata" if local_h5.is_file() else "missing_h5+missing_metadata"
            return ResolvedEpisodeView(
                str(local_h5), str(episode_path), None, None, None, None, None, None, hdf5_steps, status
            )

        view_number = 1 if view == "exterior_image_1_left" else 2
        camera_serial = (
            exterior_roles[view_number - 1]
            if exterior_roles is not None
            else metadata.get(f"ext{view_number}_cam_serial")
        )
        if camera_serial is None:
            return ResolvedEpisodeView(
                str(local_h5),
                str(episode_path),
                _optional_str(metadata_path),
                None,
                None,
                None,
                None,
                None,
                hdf5_steps,
                "missing_camera_serial",
            )

        video_path = episode_path / "recordings" / "MP4" / f"{camera_serial}.mp4"
        issues: list[str] = []
        if not local_h5.is_file():
            issues.append("missing_h5")
        if not video_path.is_file():
            issues.append("missing_video")
            return ResolvedEpisodeView(
                str(local_h5),
                str(episode_path),
                _optional_str(metadata_path),
                str(camera_serial),
                str(video_path),
                None,
                None,
                None,
                hdf5_steps,
                "+".join(issues),
            )

        video_frames, video_width, video_height = _read_video_info(video_path)
        if video_frames is None:
            issues.append("unreadable_video")
        status = "+".join(issues) if issues else "ok"
        return ResolvedEpisodeView(
            str(local_h5),
            str(episode_path),
            _optional_str(metadata_path),
            str(camera_serial),
            str(video_path),
            video_frames,
            video_width,
            video_height,
            hdf5_steps,
            status,
        )


def read_episode_metadata(metadata_files: Sequence[Path]) -> tuple[Path | None, dict[str, object] | None]:
    """Read sorted episode metadata files; some public episodes repeat the same cameras in several files."""
    if not metadata_files:
        return None, None
    metadata = [json.loads(path.read_text()) for path in metadata_files]
    cameras = {tuple(str(entry.get(field, "")) for field in _CAMERA_SERIAL_FIELDS) for entry in metadata}
    if len(cameras) != 1:
        return None, None
    return metadata_files[0], metadata[0]


def _load_exterior_roles(path: str | Path | None) -> dict[str, tuple[str, str]]:
    if path is None:
        return {}
    roles: dict[str, tuple[str, str]] = {}
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required_columns = {"episode_root", "exterior_serial_1", "exterior_serial_2"}
        missing_columns = required_columns.difference(reader.fieldnames or ())
        if missing_columns:
            raise ValueError(f"Missing exterior role columns: {', '.join(sorted(missing_columns))}")
        for line_number, row in enumerate(reader, start=2):
            episode_root = (row.get("episode_root") or "").strip()
            serials = (
                (row.get("exterior_serial_1") or "").strip(),
                (row.get("exterior_serial_2") or "").strip(),
            )
            if not episode_root:
                raise ValueError(f"Missing episode root at line {line_number}")
            if episode_root in roles:
                raise ValueError(f"Duplicate exterior role entry: {episode_root}")
            if not all(serials):
                raise ValueError(f"Missing exterior camera serial for {episode_root}")
            roles[episode_root] = serials
    return roles


def _read_hdf5_steps(path: Path) -> int | None:
    try:
        with h5py.File(path, "r") as handle:
            return int(handle["action/joint_position"].shape[0])
    except (OSError, KeyError):
        return None


def _read_video_info(path: Path) -> tuple[int | None, int | None, int | None]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        return None, None, None
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    return frames, width, height


def _unresolved(status: str) -> ResolvedEpisodeView:
    return ResolvedEpisodeView(None, None, None, None, None, None, None, None, None, status)


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)
