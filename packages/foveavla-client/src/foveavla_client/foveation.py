"""Translate the released ROI field names at data and policy boundaries."""

# Canonical fields take precedence when a caller supplies both names.
FIELD_ALIASES = {
    "future_interaction_box": "future_interaction_roi",
    "future_interaction_box_mask": "future_interaction_roi_mask",
    "future_interaction_box_valid": "future_interaction_roi_valid",
    "foveal_image": "roi_context_image",
    "left_foveal_image": "left_roi_context_image",
    "right_foveal_image": "right_roi_context_image",
    "observation/foveal_image": "observation/roi_context_image",
    "observation/left_foveal_image": "observation/left_roi_context_image",
    "observation/right_foveal_image": "observation/right_roi_context_image",
    "foveal_context_scale": "roi_context_scale",
    "foveal_context_format": "roi_context_format",
}


def canonicalize_fields(data: dict) -> dict:
    """Read legacy fields without propagating them through the model."""
    result = dict(data)
    for canonical, legacy in FIELD_ALIASES.items():
        if legacy in result:
            value = result.pop(legacy)
            result.setdefault(canonical, value)
    return result


def with_legacy_fields(data: dict) -> dict:
    """Expose aliases so released robot clients and servers remain usable."""
    result = dict(data)
    for canonical, legacy in FIELD_ALIASES.items():
        if canonical in result:
            result[legacy] = result[canonical]
    return result
