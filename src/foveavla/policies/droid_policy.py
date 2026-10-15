"""Map DROID observations to FoveaVLA global/foveal RGB and depth streams and eight robot actions."""

import dataclasses

import einops
import numpy as np
from foveavla_client import foveation

from foveavla import transforms


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class DroidInputs(transforms.DataTransformFn):
    dual_exterior_views: bool = False

    def __call__(self, data: dict) -> dict:
        data = foveation.canonicalize_fields(data)
        gripper = np.asarray(data["observation/gripper_position"])
        state = np.concatenate([data["observation/joint_position"], np.atleast_1d(gripper)])
        if self.dual_exterior_views:
            streams = (
                ("base_0_rgb", "exterior_image_1_left", "left_full"),
                ("left_wrist_0_rgb", "left_foveal_image", "left_crop"),
                ("base_1_rgb", "exterior_image_2_left", "right_full"),
                ("right_wrist_0_rgb", "right_foveal_image", "right_crop"),
            )
        else:
            streams = (
                ("base_0_rgb", "exterior_image_1_left", "full"),
                ("left_wrist_0_rgb", "foveal_image", "crop"),
            )
        # Slot and sidecar names stay compatible with the released weights and data.
        images, depths, quality = {}, {}, {}
        for slot, image, suffix in streams:
            images[slot] = _parse_image(data[f"observation/{image}"])
            depths[slot] = _parse_metric_depth(data[f"observation/metric_depth_{suffix}"])
            quality[slot] = _parse_depth_quality(data[f"observation/pointmap_quality_{suffix}"])
        inputs = {
            "state": state,
            "image": images,
            "image_mask": dict.fromkeys(images, np.True_),
            "depth": depths,
            "depth_quality": quality,
        }
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])
        if "prompt" in data:
            prompt = data["prompt"]
            inputs["prompt"] = prompt.decode("utf-8") if isinstance(prompt, bytes) else prompt
        if "future_interaction_box" in data:
            inputs["future_interaction_box"] = np.asarray(data["future_interaction_box"])
            inputs["future_interaction_box_mask"] = np.asarray(data["future_interaction_box_mask"])
        return inputs


def _parse_depth_quality(value) -> np.ndarray:
    quality = np.asarray(value, dtype=np.float32)
    if quality.shape != (224, 224):
        raise ValueError(f"Expected depth quality shape (224, 224), got {quality.shape}")
    if not np.all(np.isfinite(quality)) or np.any((quality < 0.0) | (quality > 1.0)):
        raise ValueError("Depth quality must contain finite values in [0, 1]")
    return quality


def _parse_metric_depth(value) -> np.ndarray:
    depth = np.asarray(value, dtype=np.float32)
    if depth.shape != (224, 224):
        raise ValueError(f"Expected metric depth shape (224, 224), got {depth.shape}")
    if not np.all(np.isfinite(depth)) or np.any(depth < 0.0):
        raise ValueError("Metric depth must contain finite non-negative values")
    return depth


@dataclasses.dataclass(frozen=True)
class DroidOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        data = foveation.canonicalize_fields(data)
        # The model pads actions to 32 dimensions; DROID uses seven joints and a gripper.
        outputs = {"actions": np.asarray(data["actions"][..., :8])}
        for key in ("future_interaction_box", "future_interaction_box_valid"):
            if key in data:
                outputs[key] = np.asarray(data[key])
        return outputs
