from pathlib import Path

import numpy as np
from PIL import Image

from sam3_mask.export.writer import export_alpha_cutout, export_bbox_crop, export_mask


def test_export_bbox_crop_saves_png(tmp_path: Path) -> None:
    image = Image.new("RGB", (10, 10), "white")
    out_path = tmp_path / "crop.png"

    export_bbox_crop(image, (2, 2, 4, 4), out_path)

    saved = Image.open(out_path)
    assert saved.size == (4, 4)


def test_export_alpha_cutout_saves_rgba(tmp_path: Path) -> None:
    image = Image.new("RGB", (4, 4), "white")
    mask = np.ones((4, 4), dtype=bool)
    out_path = tmp_path / "cutout.png"

    export_alpha_cutout(image, mask, (0, 0, 4, 4), out_path)

    saved = Image.open(out_path)
    assert saved.mode == "RGBA"
    assert saved.size == (4, 4)


def test_export_alpha_cutout_crops_alpha_channel(tmp_path: Path) -> None:
    image = Image.new("RGB", (4, 4), "white")
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True
    out_path = tmp_path / "cutout.png"

    export_alpha_cutout(image, mask, (1, 1, 2, 2), out_path)

    saved = Image.open(out_path)
    alpha = np.array(saved.getchannel("A"))
    assert alpha.shape == (2, 2)
    assert alpha.min() == 255


def test_export_mask_saves_binary_png(tmp_path: Path) -> None:
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True
    out_path = tmp_path / "mask.png"

    export_mask(mask, out_path)

    saved = np.array(Image.open(out_path))
    assert saved.max() == 255
    assert saved.min() == 0
