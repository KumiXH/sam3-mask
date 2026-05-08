from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from sam3_mask.scanner.filesystem import ScannedImage


@dataclass(slots=True, frozen=True)
class ImagePair:
    relative_path: Path
    lq_path: Path
    hr_path: Path


def build_pairs(
    lq_images: list[ScannedImage],
    hr_images: list[ScannedImage],
) -> tuple[list[ImagePair], list[Path], list[Path]]:
    lq_map = {item.relative_path.as_posix(): item for item in lq_images}
    hr_map = {item.relative_path.as_posix(): item for item in hr_images}
    common = sorted(set(lq_map) & set(hr_map))
    pairs = [
        ImagePair(lq_map[key].relative_path, lq_map[key].path, hr_map[key].path)
        for key in common
    ]
    missing_lq = [hr_map[key].relative_path for key in sorted(set(hr_map) - set(lq_map))]
    missing_hr = [lq_map[key].relative_path for key in sorted(set(lq_map) - set(hr_map))]
    return pairs, missing_lq, missing_hr


def read_pair_scale(lq_path: Path, hr_path: Path) -> tuple[float, float]:
    with Image.open(lq_path) as lq_image, Image.open(hr_path) as hr_image:
        return hr_image.width / lq_image.width, hr_image.height / lq_image.height
