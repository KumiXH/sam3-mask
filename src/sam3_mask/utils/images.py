from __future__ import annotations

from pathlib import Path

from PIL import Image
import numpy as np


def load_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def load_mask_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.array(image.convert("L")) > 0
