"""Online Fast-FoundationStereo preprocessing for DROID policies."""

from __future__ import annotations

import hashlib
from pathlib import Path
import time
from typing import Protocol

import numpy as np
from foveavla_client import base_policy
from typing_extensions import override

from foveavla.shared import fast_foundation_depth

_PREFIX = "observation/fast_foundation_"


class DisparityRuntime(Protocol):
    input_resolution: tuple[int, int]

    def infer(self, left_rgb: np.ndarray, right_rgb: np.ndarray) -> np.ndarray: ...


class FastFoundationDepthPolicy(base_policy.BasePolicy):
    """Populate the metric-depth fields expected by ``DroidInputs``."""

    def __init__(self, policy: base_policy.BasePolicy, runtime: DisparityRuntime, metadata: dict) -> None:
        digest = hashlib.sha256(Path(fast_foundation_depth.__file__).read_bytes()).hexdigest()
        if digest != metadata["fast_foundation_postprocess_source_sha256"]:
            raise ValueError("Fast-FoundationStereo online postprocess hash does not match the training sidecar")
        self._policy = policy
        self._runtime = runtime
        self._metadata = metadata
        dual = tuple(metadata.get("droid_exterior_views", ())) == ("physical_left", "physical_right")
        self._views = ("left_", "right_") if dual else ("",)
        self._depth_range_m = tuple(float(value) for value in metadata["metric_depth_range_m"])

    def _prepare(self, observation: dict) -> tuple[dict, float]:
        started = time.monotonic()
        observation = dict(observation)
        for view in self._views:
            self._prepare_view(observation, view)
        return observation, (time.monotonic() - started) * 1000

    def _prepare_view(self, observation: dict, view: str) -> None:
        prefix = f"{_PREFIX}{view}"
        left = np.asarray(observation.pop(f"{prefix}left_rgb"))
        right = np.asarray(observation.pop(f"{prefix}right_rgb"))
        source_size = np.asarray(observation.pop(f"{prefix}source_size"), dtype=np.int32)
        fx = float(observation.pop(f"{prefix}fx"))
        baseline_m = float(observation.pop(f"{prefix}baseline_m"))
        crop_bbox_value = observation.pop(f"{prefix}crop_bbox")
        if source_size.shape != (2,):
            raise ValueError(f"Expected FastFS source_size [width, height], got {source_size.shape}")
        source_width, source_height = (int(value) for value in source_size)
        if crop_bbox_value is None:
            crop_mode = 1
            crop_bbox = (0, 0, source_width, source_height)
        else:
            crop_mode = 0
            crop_bbox_array = np.asarray(crop_bbox_value, dtype=np.int32)
            if crop_bbox_array.shape != (4,):
                raise ValueError(f"Expected FastFS crop_bbox shape (4,), got {crop_bbox_array.shape}")
            crop_bbox = tuple(int(value) for value in crop_bbox_array)

        disparity = self._runtime.infer(left, right)
        input_width, input_height = self._runtime.input_resolution
        if disparity.shape != (input_height, input_width):
            raise ValueError(f"Expected FastFS disparity shape {(input_height, input_width)}, got {disparity.shape}")
        full_depth, crop_depth, full_valid, crop_valid = fast_foundation_depth.disparity_to_policy_depth(
            disparity,
            fx=fx,
            baseline_m=baseline_m,
            input_width=input_width,
            source_width=source_width,
            source_height=source_height,
            crop_mode=crop_mode,
            crop_bbox=crop_bbox,
            depth_range_m=self._depth_range_m,
        )
        observation.update(
            {
                # Training sidecars store float16 depth, then decode to float32.
                f"observation/metric_depth_{view}full": full_depth.astype(np.float16).astype(np.float32),
                f"observation/metric_depth_{view}crop": crop_depth.astype(np.float16).astype(np.float32),
                f"observation/pointmap_quality_{view}full": full_valid.astype(np.float32),
                f"observation/pointmap_quality_{view}crop": crop_valid.astype(np.float32),
            }
        )

    def _add_timing(self, result: dict, depth_ms: float, observation: dict) -> dict:
        result.setdefault("policy_timing", {})["fast_foundation_depth_ms"] = depth_ms
        result["metric_depth"] = {
            view: observation[f"observation/metric_depth_{view}"]
            for view in (f"{prefix}{kind}" for prefix in self._views for kind in ("full", "crop"))
        }
        return result

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        observation, depth_ms = self._prepare(obs)
        return self._add_timing(self._policy.infer(observation), depth_ms, observation)

    def infer_actions_with_pending_bbox(self, obs: dict, *, noise=None):
        observation, depth_ms = self._prepare(obs)
        result, pending = self._policy.infer_actions_with_pending_bbox(observation, noise=noise)
        return self._add_timing(result, depth_ms, observation), pending

    def infer_future_bbox(self, pending):
        return self._policy.infer_future_bbox(pending)

    def reset(self) -> None:
        self._policy.reset()

    @property
    def metadata(self) -> dict:
        return self._metadata
