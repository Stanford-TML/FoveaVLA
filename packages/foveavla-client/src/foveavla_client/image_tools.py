from __future__ import annotations

import math

import cv2
import numpy as np
from PIL import Image

_DROID_FOVEAL_CONTEXT_SIZE = 384
_DROID_FOVEAL_JPEG_QUALITY = 90
_DROID_DIRECT_PIXEL_EPSILON = 1e-3


def convert_to_uint8(img: np.ndarray) -> np.ndarray:
    """Converts an image to uint8 if it is a float image.

    This is important for reducing the size of the image when sending it over the network.
    """
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    return img


def resize_with_pad(images: np.ndarray, height: int, width: int, method=Image.BILINEAR) -> np.ndarray:
    """Replicates tf.image.resize_with_pad for multiple images using PIL. Resizes a batch of images to a target height.

    Args:
        images: A batch of images in [..., height, width, channel] format.
        height: The target height of the image.
        width: The target width of the image.
        method: The interpolation method to use. Default is bilinear.

    Returns:
        The resized images in [..., height, width, channel].
    """
    # If the images are already the correct size, return them as is.
    if images.shape[-3:-1] == (height, width):
        return images

    original_shape = images.shape

    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack([_resize_with_pad_pil(Image.fromarray(im), height, width, method=method) for im in images])
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])


def _resize_with_pad_pil(image: Image.Image, height: int, width: int, method: int) -> Image.Image:
    """Replicates tf.image.resize_with_pad for one image using PIL. Resizes an image to a target height and
    width without distortion by padding with zeros.

    Unlike the jax version, note that PIL uses [width, height, channel] ordering instead of [batch, h, w, c].
    """
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return image  # No need to resize if the image is already the correct size.

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=method)

    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    assert zero_image.size == (width, height)
    return zero_image


def extract_foveal_crop(
    image: np.ndarray,
    box: np.ndarray,
    *,
    output_size: int = 224,
    context_scale: float = 2.0,
) -> np.ndarray:
    """Render the square foveal crop exactly like the DROID foveal training repack.

    ``box`` is normalized full-image ``[x1, y1, x2, y2]``. The square may
    extend outside the source image; those pixels are filled with black. The
    crop follows the repack-v1 pipeline: resize to 384, JPEG-encode at quality
    90, decode, then resize to the model input size with OpenCV INTER_AREA.
    """
    image = np.asarray(image)
    box = np.asarray(box, dtype=np.float32)
    if image.ndim != 3 or image.shape[-1] != 3 or image.size == 0:
        raise ValueError(f"Expected a non-empty HxWxC image, got {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"Expected a uint8 RGB image, got {image.dtype}")
    if box.shape != (4,):
        raise ValueError(f"Expected box shape (4,), got {box.shape}")
    if not np.all(np.isfinite(box)) or np.any((box < 0.0) | (box > 1.0)):
        raise ValueError("Interaction box coordinates must be finite and normalized to [0, 1]")
    x1, y1, x2, y2 = box
    if x1 >= x2 or y1 >= y2:
        raise ValueError("Interaction box corners must be ordered xyxy")
    if output_size <= 0:
        raise ValueError("output_size must be positive")
    if context_scale <= 0:
        raise ValueError("context_scale must be positive")

    height, width = image.shape[:2]
    box_width = (x2 - x1) * width
    box_height = (y2 - y1) * height
    side = min(math.ceil(max(box_width, box_height) * context_scale), max(width, height))
    side = max(side, 2)
    center_x = (x1 + x2) * width / 2
    center_y = (y1 + y2) * height / 2
    left = math.floor(center_x - side / 2)
    top = math.floor(center_y - side / 2)
    right = left + side
    bottom = top + side

    canvas = np.zeros((side, side, image.shape[-1]), dtype=image.dtype)
    source_left = max(left, 0)
    source_top = max(top, 0)
    source_right = min(right, width)
    source_bottom = min(bottom, height)
    if source_left < source_right and source_top < source_bottom:
        dest_left = source_left - left
        dest_top = source_top - top
        canvas[
            dest_top : dest_top + source_bottom - source_top,
            dest_left : dest_left + source_right - source_left,
        ] = image[source_top:source_bottom, source_left:source_right]

    context_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    context_bgr = cv2.resize(
        context_bgr,
        (_DROID_FOVEAL_CONTEXT_SIZE, _DROID_FOVEAL_CONTEXT_SIZE),
        interpolation=cv2.INTER_AREA,
    )
    encoded, jpeg = cv2.imencode(
        ".jpg",
        context_bgr,
        [cv2.IMWRITE_JPEG_QUALITY, _DROID_FOVEAL_JPEG_QUALITY],
    )
    if not encoded:
        raise ValueError("Could not JPEG-encode foveal crop")
    decoded_bgr = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
    if decoded_bgr is None:
        raise ValueError("Could not JPEG-decode foveal crop")
    decoded_rgb = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)
    if decoded_rgb.shape[:2] == (output_size, output_size):
        return decoded_rgb
    return cv2.resize(decoded_rgb, (output_size, output_size), interpolation=cv2.INTER_AREA)


def extract_foveal_crop_direct_with_bbox(
    image: np.ndarray,
    box: np.ndarray,
    *,
    output_size: int = 224,
    context_scale: float = 1.5,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Render a direct-v2.2 foveal crop and return its source-image crop box.

    The square is shifted to remain in-frame while still containing the interaction box.
    Black padding is used only when the required square is larger than one of
    the source-image dimensions. The crop is resized directly to ``output_size``
    without a JPEG round trip.
    """
    image = np.asarray(image)
    box = np.asarray(box, dtype=np.float32)
    if image.ndim != 3 or image.shape[-1] != 3 or image.size == 0:
        raise ValueError(f"Expected a non-empty HxWxC image, got {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"Expected a uint8 RGB image, got {image.dtype}")
    if box.shape != (4,):
        raise ValueError(f"Expected box shape (4,), got {box.shape}")
    if not np.all(np.isfinite(box)) or np.any((box < 0.0) | (box > 1.0)):
        raise ValueError("Interaction box coordinates must be finite and normalized to [0, 1]")
    x1, y1, x2, y2 = box
    if x1 >= x2 or y1 >= y2:
        raise ValueError("Interaction box corners must be ordered xyxy")
    if output_size <= 0:
        raise ValueError("output_size must be positive")
    if context_scale <= 0:
        raise ValueError("context_scale must be positive")

    left, top, right, bottom = resolve_foveal_crop_direct_bbox(
        image.shape[:2],
        box,
        context_scale=context_scale,
    )
    side = right - left

    height, width = image.shape[:2]
    canvas = np.zeros((side, side, image.shape[-1]), dtype=image.dtype)
    source_left = max(left, 0)
    source_top = max(top, 0)
    source_right = min(right, width)
    source_bottom = min(bottom, height)
    if source_left < source_right and source_top < source_bottom:
        canvas[
            source_top - top : source_bottom - top,
            source_left - left : source_right - left,
        ] = image[source_top:source_bottom, source_left:source_right]
    if side == output_size:
        return canvas, (left, top, right, bottom)
    return cv2.resize(canvas, (output_size, output_size), interpolation=cv2.INTER_AREA), (left, top, right, bottom)


def resolve_foveal_crop_direct_bbox(
    image_shape: tuple[int, int],
    box: np.ndarray,
    *,
    context_scale: float = 1.5,
) -> tuple[int, int, int, int]:
    """Return the source-pixel square used by the direct-v2.2 foveal renderer."""
    if len(image_shape) != 2 or image_shape[0] <= 0 or image_shape[1] <= 0:
        raise ValueError(f"Expected a positive (height, width) image shape, got {image_shape}")
    box = np.asarray(box, dtype=np.float32)
    if box.shape != (4,):
        raise ValueError(f"Expected box shape (4,), got {box.shape}")
    if not np.all(np.isfinite(box)) or np.any((box < 0.0) | (box > 1.0)):
        raise ValueError("Interaction box coordinates must be finite and normalized to [0, 1]")
    x1, y1, x2, y2 = box
    if x1 >= x2 or y1 >= y2:
        raise ValueError("Interaction box corners must be ordered xyxy")
    if context_scale <= 0:
        raise ValueError("context_scale must be positive")

    height, width = image_shape
    box_left = float(x1) * width
    box_top = float(y1) * height
    box_right = float(x2) * width
    box_bottom = float(y2) * height
    box_side = max(box_right - box_left, box_bottom - box_top)
    required_side = max(
        2,
        math.ceil(box_right - _DROID_DIRECT_PIXEL_EPSILON) - math.floor(box_left + _DROID_DIRECT_PIXEL_EPSILON),
        math.ceil(box_bottom - _DROID_DIRECT_PIXEL_EPSILON) - math.floor(box_top + _DROID_DIRECT_PIXEL_EPSILON),
    )
    side = max(required_side, math.ceil(box_side * context_scale - _DROID_DIRECT_PIXEL_EPSILON))
    if required_side <= min(width, height):
        side = min(side, min(width, height))
    else:
        side = min(side, max(width, height))

    center_x = float(x1 + x2) * width / 2
    center_y = float(y1 + y2) * height / 2
    requested_left = math.floor(center_x - side / 2 + _DROID_DIRECT_PIXEL_EPSILON)
    requested_top = math.floor(center_y - side / 2 + _DROID_DIRECT_PIXEL_EPSILON)
    left = _fit_direct_crop_start(requested_left, side, width, box_left, box_right)
    top = _fit_direct_crop_start(requested_top, side, height, box_top, box_bottom)
    return left, top, left + side, top + side


def extract_foveal_crop_direct(
    image: np.ndarray,
    box: np.ndarray,
    *,
    output_size: int = 224,
    context_scale: float = 1.5,
) -> np.ndarray:
    """Render a foveal crop using the lossless direct-v2.2 training geometry."""
    context, _ = extract_foveal_crop_direct_with_bbox(
        image,
        box,
        output_size=output_size,
        context_scale=context_scale,
    )
    return context


def _fit_direct_crop_start(
    requested: int,
    side: int,
    image_length: int,
    box_min: float,
    box_max: float,
) -> int:
    if side <= image_length:
        lower = max(0, math.ceil(box_max - side - _DROID_DIRECT_PIXEL_EPSILON))
        upper = min(image_length - side, math.floor(box_min + _DROID_DIRECT_PIXEL_EPSILON))
    else:
        lower = image_length - side
        upper = 0
    if lower > upper:
        raise ValueError("Could not place a square crop that contains the interaction box")
    return min(max(requested, lower), upper)


# Compatibility wrappers for released clients using the old function and argument names.
def extract_roi_context(image, roi, *, output_size=224, context_scale=2.0):
    return extract_foveal_crop(image, roi, output_size=output_size, context_scale=context_scale)


def extract_roi_context_direct_with_bbox(image, roi, *, output_size=224, context_scale=1.5):
    return extract_foveal_crop_direct_with_bbox(image, roi, output_size=output_size, context_scale=context_scale)


def resolve_roi_context_direct_crop_bbox(image_shape, roi, *, context_scale=1.5):
    return resolve_foveal_crop_direct_bbox(image_shape, roi, context_scale=context_scale)


def extract_roi_context_direct(image, roi, *, output_size=224, context_scale=1.5):
    return extract_foveal_crop_direct(image, roi, output_size=output_size, context_scale=context_scale)
