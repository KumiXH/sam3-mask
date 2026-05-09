from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from sam3_mask.pairing.pairs import ImagePair
from sam3_mask.pipeline.types import RunSummary
from sam3_mask.utils.geometry import Box


def export_bbox_crop(image: Image.Image, box: Box, out_path: Path) -> None:
    x, y, w, h = box
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.crop((x, y, x + w, y + h)).save(out_path)


def export_alpha_cutout(
    image: Image.Image,
    mask: np.ndarray,
    box: Box,
    out_path: Path,
) -> None:
    x, y, w, h = box
    rgba = image.convert("RGBA")
    alpha = Image.fromarray(mask.astype("uint8") * 255)
    rgba.putalpha(alpha)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rgba.crop((x, y, x + w, y + h)).save(out_path)


def export_mask(mask: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype("uint8") * 255).save(out_path)


def export_single_channel_mask(mask: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(mask.astype("uint8") * 255)
    image.save(out_path)


def export_mask_crop(mask: np.ndarray, box: Box, out_path: Path) -> None:
    x, y, w, h = box
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(mask.astype("uint8") * 255)
    image.crop((x, y, x + w, y + h)).save(out_path)


def append_jsonl_record(out_path: Path, record: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_summary(out_path: Path, summary: RunSummary) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")


def write_pairs_csv(out_path: Path, pairs: list[ImagePair]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["relative_path", "lq_path", "hr_path"])
        for pair in pairs:
            writer.writerow([pair.relative_path.as_posix(), str(pair.lq_path), str(pair.hr_path)])
