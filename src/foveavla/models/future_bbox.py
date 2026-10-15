"""Coordinate utilities for the DROID future interaction box target."""

import numpy as np

RAW_IMAGE_WIDTH = 1280
RAW_IMAGE_HEIGHT = 720
MODEL_IMAGE_SIZE = 224
NUM_LOCATION_BINS = 1024

LOCATION_TOKEN_START_ID = 256_000


def source_xyxy_to_canvas_yxyx(box: np.ndarray) -> np.ndarray:
    """Convert normalized 1280x720 source xyxy coordinates to normalized 224x224 canvas yxyx."""
    box = _validate_box_array(box)
    x1, y1, x2, y2 = np.moveaxis(box, -1, 0)

    ratio, resized_width, resized_height, pad_x, pad_y = _letterbox_geometry()
    del resized_width, resized_height
    x1_canvas = (x1 * RAW_IMAGE_WIDTH / ratio + pad_x) / MODEL_IMAGE_SIZE
    x2_canvas = (x2 * RAW_IMAGE_WIDTH / ratio + pad_x) / MODEL_IMAGE_SIZE
    y1_canvas = (y1 * RAW_IMAGE_HEIGHT / ratio + pad_y) / MODEL_IMAGE_SIZE
    y2_canvas = (y2 * RAW_IMAGE_HEIGHT / ratio + pad_y) / MODEL_IMAGE_SIZE
    return np.stack([y1_canvas, x1_canvas, y2_canvas, x2_canvas], axis=-1).astype(np.float32)


def canvas_yxyx_to_source_xyxy(coords: np.ndarray) -> np.ndarray:
    """Convert normalized 224x224 canvas yxyx coordinates to normalized 1280x720 source xyxy."""
    coords = np.asarray(coords, dtype=np.float32)
    if coords.shape[-1:] != (4,):
        raise ValueError(f"Expected coordinates with final dimension 4, got {coords.shape}")
    y1, x1, y2, x2 = np.moveaxis(coords, -1, 0)

    ratio, _, _, pad_x, pad_y = _letterbox_geometry()
    x1_source = (x1 * MODEL_IMAGE_SIZE - pad_x) * ratio / RAW_IMAGE_WIDTH
    x2_source = (x2 * MODEL_IMAGE_SIZE - pad_x) * ratio / RAW_IMAGE_WIDTH
    y1_source = (y1 * MODEL_IMAGE_SIZE - pad_y) * ratio / RAW_IMAGE_HEIGHT
    y2_source = (y2 * MODEL_IMAGE_SIZE - pad_y) * ratio / RAW_IMAGE_HEIGHT
    result = np.stack([x1_source, y1_source, x2_source, y2_source], axis=-1)
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def quantize_canvas_coordinates(coords: np.ndarray) -> np.ndarray:
    """Quantize normalized canvas coordinates to local location classes 0..1023."""
    coords = np.asarray(coords, dtype=np.float32)
    return np.clip(np.rint(coords * (NUM_LOCATION_BINS - 1)), 0, NUM_LOCATION_BINS - 1).astype(np.int32)


def dequantize_location_classes(classes: np.ndarray) -> np.ndarray:
    """Convert local location classes 0..1023 to normalized canvas coordinates."""
    classes = np.asarray(classes)
    if np.any((classes < 0) | (classes >= NUM_LOCATION_BINS)):
        raise ValueError("Location classes must be in [0, 1023]")
    return classes.astype(np.float32) / (NUM_LOCATION_BINS - 1)


def source_xyxy_to_location_classes(box: np.ndarray) -> np.ndarray:
    return quantize_canvas_coordinates(source_xyxy_to_canvas_yxyx(box))


def location_classes_to_source_xyxy(classes: np.ndarray) -> np.ndarray:
    return canvas_yxyx_to_source_xyxy(dequantize_location_classes(classes))


def location_classes_to_token_ids(classes: np.ndarray) -> np.ndarray:
    classes = np.asarray(classes)
    if np.any((classes < 0) | (classes >= NUM_LOCATION_BINS)):
        raise ValueError("Location classes must be in [0, 1023]")
    return (classes + LOCATION_TOKEN_START_ID).astype(np.int32)


def _letterbox_geometry() -> tuple[float, int, int, int, int]:
    ratio = max(RAW_IMAGE_WIDTH / MODEL_IMAGE_SIZE, RAW_IMAGE_HEIGHT / MODEL_IMAGE_SIZE)
    resized_height = int(RAW_IMAGE_HEIGHT / ratio)
    resized_width = int(RAW_IMAGE_WIDTH / ratio)
    pad_y = (MODEL_IMAGE_SIZE - resized_height) // 2
    pad_x = (MODEL_IMAGE_SIZE - resized_width) // 2
    return ratio, resized_width, resized_height, pad_x, pad_y


def _validate_box_array(box: np.ndarray) -> np.ndarray:
    box = np.asarray(box, dtype=np.float32)
    if box.shape[-1:] != (4,):
        raise ValueError(f"Expected bbox with final dimension 4, got {box.shape}")
    if not np.all(np.isfinite(box)):
        raise ValueError("BBox coordinates must be finite")
    if np.any((box < 0.0) | (box > 1.0)):
        raise ValueError("BBox coordinates must be normalized to [0, 1]")
    if np.any(box[..., 0] >= box[..., 2]) or np.any(box[..., 1] >= box[..., 3]):
        raise ValueError("BBox corners must be ordered xyxy")
    return box
