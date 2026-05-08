from __future__ import annotations

import numpy as np
from PIL import Image


Box = tuple[int, int, int, int]


def scale_box(box: Box, scale_x: float, scale_y: float) -> Box:
    x, y, w, h = box
    x1 = round(x * scale_x)
    y1 = round(y * scale_y)
    x2 = round((x + w) * scale_x)
    y2 = round((y + h) * scale_y)
    return (x1, y1, max(0, x2 - x1), max(0, y2 - y1))


def clamp_box(box: Box, width: int, height: int) -> Box:
    x, y, w, h = box
    x = max(0, min(x, width))
    y = max(0, min(y, height))
    right = max(x, min(x + max(0, w), width))
    bottom = max(y, min(y + max(0, h), height))
    return x, y, right - x, bottom - y


def resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(mask.astype("uint8") * 255)
    resized = image.resize(size, resample=Image.Resampling.NEAREST)
    return np.array(resized) > 0
