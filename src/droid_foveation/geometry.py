from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence

from droid_foveation.role_policy import RolePolicy

NormalizedBox = tuple[float, float, float, float]


@dataclasses.dataclass(frozen=True)
class FrameInteractionAnnotation:
    primitive_skill: str
    subtask: str | None
    instruction: str | None
    source_width: int | None
    source_height: int | None
    gripper_box: NormalizedBox | None
    object_box: NormalizedBox | None
    placement_box: NormalizedBox | None
    interaction_box: NormalizedBox | None
    box_valid: bool
    invalid_reason: str | None

    def to_row(self, annotation_key: str, frame_idx: int) -> dict[str, object]:
        return {
            "annotation_key": annotation_key,
            "frame_idx": frame_idx,
            "primitive_skill": self.primitive_skill or None,
            "subtask": self.subtask,
            "instruction": self.instruction,
            "source_width": self.source_width,
            "source_height": self.source_height,
            "gripper_box": self.gripper_box,
            "object_box": self.object_box,
            "placement_box": self.placement_box,
            "interaction_box": self.interaction_box,
            "box_valid": self.box_valid,
            "invalid_reason": self.invalid_reason,
        }


def annotate_frame(frame: Mapping[str, object], policy: RolePolicy) -> FrameInteractionAnnotation:
    primitive = _normalize_text(frame.get("primitive_skill")) or ""
    subtask = _normalize_text(frame.get("substask"))
    instruction = _normalize_text(frame.get("instruction_add"))
    shape = _parse_origin_shape(frame.get("origin_shape"))

    if shape is None:
        return FrameInteractionAnnotation(
            primitive,
            subtask,
            instruction,
            None,
            None,
            None,
            None,
            None,
            None,
            False,
            "invalid_origin_shape",
        )

    width, height = shape
    boxes_by_field = {
        field: _normalize_box(frame.get(field), width, height) for field in policy.role_fields.values()
    }
    boxes_by_role = {role: boxes_by_field[field] for role, field in policy.role_fields.items()}

    gripper_box = boxes_by_field.get("gripper_box")
    object_box = boxes_by_field.get("object_box")
    placement_box = boxes_by_field.get("placement_proposal")

    required_roles = policy.required_roles(primitive)
    if required_roles is None:
        return FrameInteractionAnnotation(
            primitive,
            subtask,
            instruction,
            width,
            height,
            gripper_box,
            object_box,
            placement_box,
            None,
            False,
            f"unsupported_primitive:{primitive or '<missing>'}",
        )

    missing_roles = [role for role in required_roles if boxes_by_role[role] is None]
    if missing_roles:
        return FrameInteractionAnnotation(
            primitive,
            subtask,
            instruction,
            width,
            height,
            gripper_box,
            object_box,
            placement_box,
            None,
            False,
            "missing_roles:" + ",".join(missing_roles),
        )

    interaction_box = union_boxes([boxes_by_role[role] for role in required_roles])
    return FrameInteractionAnnotation(
        primitive,
        subtask,
        instruction,
        width,
        height,
        gripper_box,
        object_box,
        placement_box,
        interaction_box,
        True,
        None,
    )


def union_boxes(boxes: Sequence[NormalizedBox | None]) -> NormalizedBox:
    present = [box for box in boxes if box is not None]
    if not present:
        raise ValueError("Cannot union an empty set of boxes")
    return (
        min(box[0] for box in present),
        min(box[1] for box in present),
        max(box[2] for box in present),
        max(box[3] for box in present),
    )


def _parse_origin_shape(value: object) -> tuple[int, int] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        return None
    try:
        width, height = int(value[0]), int(value[1])
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    return width, height


def _normalize_box(value: object, width: int, height: int) -> NormalizedBox | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        return None
    try:
        top_left, bottom_right = value
        if len(top_left) != 2 or len(bottom_right) != 2:  # type: ignore[arg-type]
            return None
        x1, y1 = float(top_left[0]), float(top_left[1])  # type: ignore[index]
        x2, y2 = float(bottom_right[0]), float(bottom_right[1])  # type: ignore[index]
    except (TypeError, ValueError, IndexError):
        return None

    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        return None
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        return None
    return x1 / width, y1 / height, x2 / width, y2 / height


def _normalize_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
