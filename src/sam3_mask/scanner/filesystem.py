from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True, frozen=True)
class ScannedImage:
    root: Path
    path: Path
    relative_path: Path


def scan_images(root: Path, exts: list[str]) -> list[ScannedImage]:
    allowed = {ext.lower() for ext in exts}
    items: list[ScannedImage] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in allowed:
            continue
        items.append(
            ScannedImage(
                root=root,
                path=path,
                relative_path=path.relative_to(root),
            )
        )
    return items
