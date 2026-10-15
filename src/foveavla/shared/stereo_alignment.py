from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

OFFSETS = tuple(range(-2, 3))
MAX_MEAN_MAE = 8.0
MAX_FRAME_MAE = 16.0


@dataclasses.dataclass(frozen=True)
class AlignmentResult:
    best_offset: int
    frames_compared: int
    mae_by_offset: dict[int, float]
    max_mae_by_offset: dict[int, float]

    @property
    def passed(self) -> bool:
        return (
            self.best_offset == 0
            and self.mae_by_offset[0] <= MAX_MEAN_MAE
            and self.max_mae_by_offset[0] <= MAX_FRAME_MAE
        )


def select_internal_frames(frame_indices: np.ndarray, count: int) -> np.ndarray:
    """Select evenly spaced interior frames, avoiding video boundaries when possible."""
    frame_indices = np.asarray(frame_indices, dtype=np.int32)
    if frame_indices.ndim != 1 or len(frame_indices) == 0:
        raise ValueError("frame_indices must be a nonempty one-dimensional array")
    if count < 1:
        raise ValueError("alignment frame count must be positive")
    positions = np.linspace(0, len(frame_indices) - 1, min(count, len(frame_indices)) + 2)[1:-1]
    positions = np.unique(np.rint(positions).astype(int))
    return frame_indices[positions]


def compare_video_frames(
    video_path: str | Path,
    svo_frames: np.ndarray,
    frame_indices: np.ndarray,
    *,
    stride: int,
) -> AlignmentResult:
    import cv2

    svo_frames = np.asarray(svo_frames)
    frame_indices = np.asarray(frame_indices, dtype=np.int64)
    if (
        svo_frames.ndim != 4
        or svo_frames.shape[0] != len(frame_indices)
        or svo_frames.shape[-1] != 3
        or svo_frames.dtype != np.uint8
    ):
        raise ValueError(f"Invalid SVO alignment image array: {svo_frames.shape} {svo_frames.dtype}")
    if stride < 1:
        raise ValueError("Alignment stride must be positive")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open alignment video: {video_path}")
    try:
        video_frame_count = round(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        usable = [
            index
            for index, frame_idx in enumerate(frame_indices)
            if frame_idx + min(OFFSETS) >= 0 and frame_idx + max(OFFSETS) < video_frame_count
        ]
        if not usable:
            raise ValueError(f"No alignment frames are inside usable video bounds: {video_path}")
        errors_by_offset: dict[int, list[float]] = {offset: [] for offset in OFFSETS}
        for index in usable:
            svo_frame = svo_frames[index]
            frame_idx = frame_indices[index]
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx + min(OFFSETS)))
            for offset in OFFSETS:
                ok, video_frame = capture.read()
                if not ok:
                    raise ValueError(f"Could not read frame {frame_idx + offset} from {video_path}")
                video_frame = video_frame[::stride, ::stride]
                if video_frame.shape != svo_frame.shape:
                    raise ValueError(
                        f"SVO/MP4 alignment shape mismatch for {video_path}: {svo_frame.shape} != {video_frame.shape}"
                    )
                errors_by_offset[offset].append(
                    float(np.abs(video_frame.astype(np.float32) - svo_frame.astype(np.float32)).mean())
                )
    finally:
        capture.release()

    mean_errors = {offset: float(np.mean(errors)) for offset, errors in errors_by_offset.items()}
    max_errors = {offset: float(np.max(errors)) for offset, errors in errors_by_offset.items()}
    return AlignmentResult(
        best_offset=min(mean_errors, key=mean_errors.get),
        frames_compared=len(usable),
        mae_by_offset=mean_errors,
        max_mae_by_offset=max_errors,
    )
