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


def mask_bbox(mask: np.ndarray) -> Box | None:
    points = np.argwhere(mask)
    if points.size == 0:
        return None
    y0, x0 = points.min(axis=0)
    y1, x1 = points.max(axis=0)
    return int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1)


def fixed_windows_covering_box(box: Box, window_size: int, width: int, height: int) -> list[tuple[int, int, int, int, int, int]]:
    x, y, w, h = box
    window_size = max(1, min(window_size, width, height))
    if w <= window_size:
        x_starts = [_centered_start(x, w, window_size, width)]
    else:
        x_starts = _axis_starts(x, w, window_size, width)
    if h <= window_size:
        y_starts = [_centered_start(y, h, window_size, height)]
    else:
        y_starts = _axis_starts(y, h, window_size, height)

    windows: list[tuple[int, int, int, int, int, int]] = []
    for row, y_start in enumerate(y_starts):
        for col, x_start in enumerate(x_starts):
            windows.append((x_start, y_start, window_size, window_size, row, col))
    return windows


def _axis_starts(start: int, span: int, window_size: int, limit: int) -> list[int]:
    min_start = max(0, min(start, limit - window_size))
    max_start = max(0, min(start + span - window_size, limit - window_size))
    positions = list(range(min_start, max_start + 1, window_size))
    if not positions:
        positions = [min_start]
    if positions[-1] != max_start:
        positions.append(max_start)
    return positions


def _centered_start(start: int, span: int, window_size: int, limit: int) -> int:
    center = start + (span / 2.0)
    candidate = round(center - (window_size / 2.0))
    return max(0, min(candidate, max(0, limit - window_size)))
