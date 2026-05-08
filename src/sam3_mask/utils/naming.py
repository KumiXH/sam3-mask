from __future__ import annotations

import re
from pathlib import PurePosixPath


def normalize_label(label: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z]+", "_", label.strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or "object"


def build_object_stem(relative_path: str, label: str, index: int) -> str:
    rel = PurePosixPath(relative_path.replace("\\", "/"))
    prefix = "_".join(rel.parts[:-1])
    label_part = normalize_label(label)
    parts = [part for part in [prefix, rel.stem, label_part, f"{index:02d}"] if part]
    return "_".join(parts)
