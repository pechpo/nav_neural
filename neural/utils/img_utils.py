"""Small image helpers for map visualization."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def crop_white_border(image: np.ndarray) -> np.ndarray:
    """Crop the image to the bounding box of non-white pixels (BGR)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    non_white = np.argwhere(gray != 255)
    if non_white.size == 0:
        return image
    min_row, min_col = np.min(non_white, axis=0)
    max_row, max_col = np.max(non_white, axis=0)
    return image[min_row : max_row + 1, min_col : max_col + 1, :]


def pad_to_square(
    img: np.ndarray,
    padding_color: Tuple[int, int, int] = (255, 255, 255),
    extra_pad: int = 0,
) -> np.ndarray:
    """Pad an image to a square canvas using the given background color."""
    height, width = img.shape[:2]
    square_size = max(height, width) + int(extra_pad)
    canvas = np.ones((square_size, square_size, 3), dtype=np.uint8) * np.array(padding_color, dtype=np.uint8)
    y0 = (square_size - height) // 2
    x0 = (square_size - width) // 2
    canvas[y0 : y0 + height, x0 : x0 + width] = img
    return canvas


def pad_larger_dim(image: np.ndarray, target_dimension: int) -> np.ndarray:
    """Pad an image so its larger dimension reaches target_dimension."""
    height, width = image.shape[:2]
    larger = max(height, width)
    if larger >= int(target_dimension):
        return image
    pad_amount = int(target_dimension) - larger
    pad_a = pad_amount // 2
    pad_b = pad_amount - pad_a
    if height > width:
        left_pad = np.ones((height, pad_a, 3), dtype=np.uint8) * 255
        right_pad = np.ones((height, pad_b, 3), dtype=np.uint8) * 255
        return np.hstack((left_pad, image, right_pad))
    top_pad = np.ones((pad_a, width, 3), dtype=np.uint8) * 255
    bottom_pad = np.ones((pad_b, width, 3), dtype=np.uint8) * 255
    return np.vstack((top_pad, image, bottom_pad))


def reorient_rescale_map(vis_map_img: np.ndarray) -> np.ndarray:
    """Crop and pad a map image for consistent display."""
    vis_map_img = crop_white_border(vis_map_img)
    vis_map_img = pad_larger_dim(vis_map_img, 150)
    vis_map_img = pad_to_square(vis_map_img, extra_pad=50)
    vis_map_img = cv2.copyMakeBorder(
        vis_map_img,
        50,
        50,
        50,
        50,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )
    return vis_map_img
