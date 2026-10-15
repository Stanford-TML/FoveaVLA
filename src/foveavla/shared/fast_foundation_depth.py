"""Shared Fast-FoundationStereo disparity-to-policy-depth preprocessing."""

from __future__ import annotations

import cv2
import numpy as np


def _aligned_source_grid(
    source_width: int,
    source_height: int,
    output_size: int,
    *,
    crop_bbox: tuple[int, int, int, int] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target = np.arange(output_size, dtype=np.float32)
    if crop_bbox is None:
        ratio = max(source_width / output_size, source_height / output_size)
        resized_width = int(source_width / ratio)
        resized_height = int(source_height / ratio)
        left = (output_size - resized_width) // 2
        top = (output_size - resized_height) // 2
        source_x = np.floor((target - left + 0.5) * source_width / resized_width).astype(np.int32)
        source_y = np.floor((target - top + 0.5) * source_height / resized_height).astype(np.int32)
        content_x = (target >= left) & (target < left + resized_width)
        content_y = (target >= top) & (target < top + resized_height)
    else:
        crop_left, crop_top, crop_right, crop_bottom = crop_bbox
        crop_side = crop_right - crop_left
        if crop_side <= 0 or crop_bottom - crop_top != crop_side:
            raise ValueError(f"Invalid square crop: {crop_bbox}")
        source_x = np.floor(crop_left + (target + 0.5) * crop_side / output_size).astype(np.int32)
        source_y = np.floor(crop_top + (target + 0.5) * crop_side / output_size).astype(np.int32)
        content_x = (source_x >= 0) & (source_x < source_width)
        content_y = (source_y >= 0) & (source_y < source_height)
    source_x = np.clip(source_x, 0, source_width - 1)
    source_y = np.clip(source_y, 0, source_height - 1)
    grid_x, grid_y = np.meshgrid(source_x, source_y)
    return grid_x, grid_y, content_y[:, None] & content_x[None, :]


def _sample_aligned_depth(
    depth: np.ndarray,
    valid: np.ndarray,
    output_size: int,
    *,
    crop_bbox: tuple[int, int, int, int] | None,
) -> tuple[np.ndarray, np.ndarray]:
    source_height, source_width = depth.shape
    grid_x, grid_y, content = _aligned_source_grid(
        source_width,
        source_height,
        output_size,
        crop_bbox=crop_bbox,
    )
    output_valid = content & valid[grid_y, grid_x]
    output_depth = np.where(output_valid, depth[grid_y, grid_x], 0.0).astype(np.float32)
    return output_depth, output_valid


def disparity_to_policy_depth(
    disparity: np.ndarray,
    *,
    fx: float,
    baseline_m: float,
    input_width: int,
    source_width: int,
    source_height: int,
    crop_mode: int,
    crop_bbox: tuple[int, int, int, int],
    output_size: int = 224,
    depth_range_m: tuple[float, float] = (0.2, 3.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert left-reference input disparity into RGB-aligned full/ROI metric Z-depth."""
    if disparity.ndim != 2 or disparity.shape[1] != input_width:
        raise ValueError(f"Unexpected disparity shape: {disparity.shape}")
    input_scale = input_width / source_width
    if not 0.0 < input_scale <= 1.0 or fx <= 0.0 or baseline_m <= 0.0:
        raise ValueError("Camera geometry and input scale must be positive")
    if output_size <= 0 or depth_range_m[0] <= 0.0 or depth_range_m[1] <= depth_range_m[0]:
        raise ValueError("Output size and depth range are invalid")

    original_disparity = cv2.resize(
        disparity,
        (source_width, source_height),
        interpolation=cv2.INTER_LINEAR,
    ) / input_scale
    _, xx = np.indices(original_disparity.shape)
    valid = np.isfinite(original_disparity) & (original_disparity > 0.0)
    valid &= (xx - original_disparity) >= 0.0
    depth = np.zeros(original_disparity.shape, dtype=np.float32)
    depth[valid] = fx * baseline_m / original_disparity[valid]
    valid &= (depth >= depth_range_m[0]) & (depth <= depth_range_m[1])
    depth[~valid] = 0.0

    full_depth, full_valid = _sample_aligned_depth(depth, valid, output_size, crop_bbox=None)
    if crop_mode == 1:
        return full_depth, full_depth, full_valid, full_valid
    crop_depth, crop_valid = _sample_aligned_depth(
        depth,
        valid,
        output_size,
        crop_bbox=crop_bbox,
    )
    return full_depth, crop_depth, full_valid, crop_valid
