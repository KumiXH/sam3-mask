# SAM3 LQ/HR Object Crop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Windows/Linux-compatible PyTorch CLI that scans paired `LQ`/`HR` directories, runs prompt-based `SAM3` segmentation on `LQ`, and exports aligned object crops and manifests for both sides.

**Architecture:** Keep the package lightweight and testable. Separate pure filesystem and geometry logic from model backends, and define a small backend abstraction so V1 runs with a dummy backend locally and with `SAM3` later on CPU/CUDA.

**Tech Stack:** Python 3.10+, PyTorch, Pillow, NumPy, PyYAML, pytest

---

## File Structure

Create this structure before implementation:

- `D:\Repository\sam3-mask\pyproject.toml`
  - Package metadata and dependencies
- `D:\Repository\sam3-mask\README.md`
  - Setup and usage notes for Windows/Linux
- `D:\Repository\sam3-mask\configs\default.yaml`
  - Default runtime configuration
- `D:\Repository\sam3-mask\src\sam3_mask\__init__.py`
  - Package marker
- `D:\Repository\sam3-mask\src\sam3_mask\main.py`
  - `python -m sam3_mask.main` entrypoint
- `D:\Repository\sam3-mask\src\sam3_mask\cli.py`
  - CLI argument parsing and config override handling
- `D:\Repository\sam3-mask\src\sam3_mask\config\schema.py`
  - Dataclasses for validated config
- `D:\Repository\sam3-mask\src\sam3_mask\config\loader.py`
  - YAML load and override merge logic
- `D:\Repository\sam3-mask\src\sam3_mask\scanner\filesystem.py`
  - Recursive image discovery
- `D:\Repository\sam3-mask\src\sam3_mask\pairing\pairs.py`
  - Strict relative-path pairing and scale validation
- `D:\Repository\sam3-mask\src\sam3_mask\models\types.py`
  - Shared prediction dataclasses and typed aliases
- `D:\Repository\sam3-mask\src\sam3_mask\models\base.py`
  - `ProposalProvider` and `SegmenterBackend` interfaces
- `D:\Repository\sam3-mask\src\sam3_mask\models\proposals.py`
  - `NoopProposalProvider`
- `D:\Repository\sam3-mask\src\sam3_mask\models\dummy_backend.py`
  - Deterministic CPU-safe backend for tests and local dry runs
- `D:\Repository\sam3-mask\src\sam3_mask\models\sam3_backend.py`
  - Real `SAM3` adapter with availability checks
- `D:\Repository\sam3-mask\src\sam3_mask\utils\images.py`
  - Image load/save helpers
- `D:\Repository\sam3-mask\src\sam3_mask\utils\geometry.py`
  - Box scaling, mask resizing, clamping
- `D:\Repository\sam3-mask\src\sam3_mask\utils\naming.py`
  - Output naming helpers
- `D:\Repository\sam3-mask\src\sam3_mask\export\writer.py`
  - Crop/cutout/mask export and manifest writing
- `D:\Repository\sam3-mask\src\sam3_mask\pipeline\runner.py`
  - End-to-end orchestration
- `D:\Repository\sam3-mask\tests\test_config.py`
- `D:\Repository\sam3-mask\tests\test_pairing.py`
- `D:\Repository\sam3-mask\tests\test_geometry.py`
- `D:\Repository\sam3-mask\tests\test_naming.py`
- `D:\Repository\sam3-mask\tests\test_export_writer.py`
- `D:\Repository\sam3-mask\tests\test_pipeline_dummy.py`

### Task 1: Scaffold The Package And Config Surface

**Files:**
- Create: `D:\Repository\sam3-mask\pyproject.toml`
- Create: `D:\Repository\sam3-mask\README.md`
- Create: `D:\Repository\sam3-mask\configs\default.yaml`
- Create: `D:\Repository\sam3-mask\src\sam3_mask\__init__.py`
- Create: `D:\Repository\sam3-mask\src\sam3_mask\main.py`
- Create: `D:\Repository\sam3-mask\src\sam3_mask\cli.py`
- Create: `D:\Repository\sam3-mask\src\sam3_mask\config\schema.py`
- Create: `D:\Repository\sam3-mask\src\sam3_mask\config\loader.py`
- Test: `D:\Repository\sam3-mask\tests\test_config.py`

- [ ] **Step 1: Write the failing config tests**

```python
from pathlib import Path

from sam3_mask.config.loader import load_config


def test_load_config_reads_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
input:
  lq_dir: LQ
  hr_dir: HR
prompts:
  labels: [face, bird]
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path, cli_overrides={})

    assert config.input.lq_dir == Path("LQ")
    assert config.input.hr_dir == Path("HR")
    assert config.prompts.labels == ["face", "bird"]


def test_load_config_applies_cli_overrides(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
input:
  lq_dir: LQ
  hr_dir: HR
prompts:
  labels: [face]
output:
  save_cutout: false
""".strip(),
        encoding="utf-8",
    )

    config = load_config(
        config_path,
        cli_overrides={
            "prompts.labels": ["face", "plant"],
            "output.save_cutout": True,
        },
    )

    assert config.prompts.labels == ["face", "plant"]
    assert config.output.save_cutout is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'sam3_mask'`

- [ ] **Step 3: Write the minimal package and config implementation**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "sam3-mask"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
  "numpy>=1.26",
  "Pillow>=10.0",
  "PyYAML>=6.0",
  "torch>=2.2",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[tool.setuptools]
package-dir = {"" = "src"}
```

```python
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class InputConfig:
    lq_dir: Path
    hr_dir: Path
    exts: list[str] = field(default_factory=lambda: [".png", ".jpg", ".jpeg", ".webp"])


@dataclass(slots=True)
class OutputConfig:
    out_dir: Path = Path("output")
    save_crop: bool = True
    save_cutout: bool = False
    save_mask: bool = False
    save_overlay: bool = False


@dataclass(slots=True)
class PromptsConfig:
    labels: list[str]
    label_mode: str = "single"
    score_threshold: float = 0.25


@dataclass(slots=True)
class AppConfig:
    input: InputConfig
    output: OutputConfig
    prompts: PromptsConfig
```

```python
from pathlib import Path
from typing import Any

import yaml

from sam3_mask.config.schema import AppConfig, InputConfig, OutputConfig, PromptsConfig


def _deep_set(data: dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor = data
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def load_config(path: Path, cli_overrides: dict[str, Any]) -> AppConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for key, value in cli_overrides.items():
        _deep_set(raw, key, value)
    input_cfg = raw["input"]
    output_cfg = raw.get("output", {})
    prompts_cfg = raw["prompts"]
    return AppConfig(
        input=InputConfig(
            lq_dir=Path(input_cfg["lq_dir"]),
            hr_dir=Path(input_cfg["hr_dir"]),
            exts=input_cfg.get("exts", [".png", ".jpg", ".jpeg", ".webp"]),
        ),
        output=OutputConfig(
            out_dir=Path(output_cfg.get("out_dir", "output")),
            save_crop=output_cfg.get("save_crop", True),
            save_cutout=output_cfg.get("save_cutout", False),
            save_mask=output_cfg.get("save_mask", False),
            save_overlay=output_cfg.get("save_overlay", False),
        ),
        prompts=PromptsConfig(
            labels=list(prompts_cfg["labels"]),
            label_mode=prompts_cfg.get("label_mode", "single"),
            score_threshold=float(prompts_cfg.get("score_threshold", 0.25)),
        ),
    )
```

```python
import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--labels", nargs="*")
    parser.add_argument("--save-cutout", action="store_true")
    return parser
```

- [ ] **Step 4: Add the default config and entrypoint**

```yaml
input:
  lq_dir: ./data/LQ
  hr_dir: ./data/HR
  exts: [".png", ".jpg", ".jpeg", ".webp"]

output:
  out_dir: ./output
  save_crop: true
  save_cutout: false
  save_mask: false
  save_overlay: false

prompts:
  labels: [face, bird, plant, texture]
  label_mode: single
  score_threshold: 0.25
```

```python
from sam3_mask.cli import build_parser
from sam3_mask.config.loader import load_config


def main() -> int:
    args = build_parser().parse_args()
    overrides = {}
    if args.labels:
        overrides["prompts.labels"] = args.labels
    if args.save_cutout:
        overrides["output.save_cutout"] = True
    load_config(args.config, overrides)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`  
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml README.md configs/default.yaml src/sam3_mask tests/test_config.py
git commit -m "feat: scaffold project and config loading"
```

### Task 2: Implement Recursive Scan And Strict LQ/HR Pairing

**Files:**
- Create: `D:\Repository\sam3-mask\src\sam3_mask\scanner\filesystem.py`
- Create: `D:\Repository\sam3-mask\src\sam3_mask\pairing\pairs.py`
- Create: `D:\Repository\sam3-mask\tests\test_pairing.py`

- [ ] **Step 1: Write the failing scan/pairing tests**

```python
from pathlib import Path

from sam3_mask.pairing.pairs import build_pairs
from sam3_mask.scanner.filesystem import scan_images


def test_scan_images_returns_relative_paths(tmp_path: Path) -> None:
    root = tmp_path / "LQ"
    nested = root / "a" / "b"
    nested.mkdir(parents=True)
    (nested / "1.png").write_bytes(b"fake")

    results = scan_images(root, [".png"])

    assert [item.relative_path.as_posix() for item in results] == ["a/b/1.png"]


def test_build_pairs_uses_strict_relative_path(tmp_path: Path) -> None:
    lq_root = tmp_path / "LQ"
    hr_root = tmp_path / "HR"
    (lq_root / "x").mkdir(parents=True)
    (hr_root / "x").mkdir(parents=True)
    (hr_root / "y").mkdir(parents=True)
    (lq_root / "x" / "1.png").write_bytes(b"lq")
    (hr_root / "x" / "1.png").write_bytes(b"hr")
    (hr_root / "y" / "1.png").write_bytes(b"wrong")

    pairs, missing_lq, missing_hr = build_pairs(
        scan_images(lq_root, [".png"]),
        scan_images(hr_root, [".png"]),
    )

    assert len(pairs) == 1
    assert pairs[0].relative_path.as_posix() == "x/1.png"
    assert missing_lq == [Path("y/1.png")]
    assert missing_hr == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pairing.py -v`  
Expected: FAIL with import errors for scanner/pairing modules

- [ ] **Step 3: Implement scan and pairing logic**

```python
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
        items.append(ScannedImage(root=root, path=path, relative_path=path.relative_to(root)))
    return items
```

```python
from dataclasses import dataclass
from pathlib import Path

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
    lq_map = {item.relative_path: item.path for item in lq_images}
    hr_map = {item.relative_path: item.path for item in hr_images}
    common = sorted(set(lq_map) & set(hr_map))
    pairs = [ImagePair(path, lq_map[path], hr_map[path]) for path in common]
    missing_lq = sorted(set(hr_map) - set(lq_map))
    missing_hr = sorted(set(lq_map) - set(hr_map))
    return pairs, missing_lq, missing_hr
```

- [ ] **Step 4: Add scale metadata test and implementation**

```python
from PIL import Image

from sam3_mask.pairing.pairs import read_pair_scale


def test_read_pair_scale_reports_two_x(tmp_path: Path) -> None:
    lq_path = tmp_path / "lq.png"
    hr_path = tmp_path / "hr.png"
    Image.new("RGB", (8, 8), "white").save(lq_path)
    Image.new("RGB", (16, 16), "white").save(hr_path)

    scale_x, scale_y = read_pair_scale(lq_path, hr_path)

    assert scale_x == 2.0
    assert scale_y == 2.0
```

```python
from PIL import Image


def read_pair_scale(lq_path: Path, hr_path: Path) -> tuple[float, float]:
    with Image.open(lq_path) as lq_image, Image.open(hr_path) as hr_image:
        return hr_image.width / lq_image.width, hr_image.height / lq_image.height
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_pairing.py -v`  
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/sam3_mask/scanner/filesystem.py src/sam3_mask/pairing/pairs.py tests/test_pairing.py
git commit -m "feat: add recursive scanning and strict pairing"
```

### Task 3: Implement Geometry, Naming, And Export Helpers

**Files:**
- Create: `D:\Repository\sam3-mask\src\sam3_mask\utils\geometry.py`
- Create: `D:\Repository\sam3-mask\src\sam3_mask\utils\naming.py`
- Create: `D:\Repository\sam3-mask\src\sam3_mask\utils\images.py`
- Create: `D:\Repository\sam3-mask\src\sam3_mask\export\writer.py`
- Create: `D:\Repository\sam3-mask\tests\test_geometry.py`
- Create: `D:\Repository\sam3-mask\tests\test_naming.py`
- Create: `D:\Repository\sam3-mask\tests\test_export_writer.py`

- [ ] **Step 1: Write the failing geometry and naming tests**

```python
import numpy as np

from sam3_mask.utils.geometry import clamp_box, scale_box
from sam3_mask.utils.naming import build_object_stem


def test_scale_box_maps_lq_to_hr() -> None:
    assert scale_box((1, 2, 3, 4), 2.0, 2.0) == (2, 4, 6, 8)


def test_clamp_box_stays_in_bounds() -> None:
    assert clamp_box((-1, 3, 10, 8), width=8, height=8) == (0, 3, 8, 8)


def test_build_object_stem_flattens_parent_dirs() -> None:
    stem = build_object_stem("a/b/1.png", "face", 1)
    assert stem == "a_b_1_face_01"
```

- [ ] **Step 2: Add the failing export tests**

```python
from pathlib import Path

import numpy as np
from PIL import Image

from sam3_mask.export.writer import export_bbox_crop


def test_export_bbox_crop_saves_png(tmp_path: Path) -> None:
    image = Image.new("RGB", (10, 10), "white")
    out_path = tmp_path / "crop.png"

    export_bbox_crop(image, (2, 2, 4, 4), out_path)

    saved = Image.open(out_path)
    assert saved.size == (4, 4)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_geometry.py tests/test_naming.py tests/test_export_writer.py -v`  
Expected: FAIL with import errors for geometry, naming, and export helpers

- [ ] **Step 4: Implement geometry, naming, and export helpers**

```python
def scale_box(box: tuple[int, int, int, int], scale_x: float, scale_y: float) -> tuple[int, int, int, int]:
    x, y, w, h = box
    return (
        round(x * scale_x),
        round(y * scale_y),
        round(w * scale_x),
        round(h * scale_y),
    )


def clamp_box(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x, y, w, h = box
    x = max(0, min(x, width))
    y = max(0, min(y, height))
    w = max(0, min(w, width - x))
    h = max(0, min(h, height - y))
    return x, y, w, h
```

```python
from pathlib import PurePosixPath


def build_object_stem(relative_path: str, label: str, index: int) -> str:
    rel = PurePosixPath(relative_path)
    prefix = "_".join(rel.parts[:-1])
    stem = rel.stem
    parts = [part for part in [prefix, stem, label, f"{index:02d}"] if part]
    return "_".join(parts)
```

```python
from pathlib import Path

import numpy as np
from PIL import Image


def export_bbox_crop(image: Image.Image, box: tuple[int, int, int, int], out_path: Path) -> None:
    x, y, w, h = box
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.crop((x, y, x + w, y + h)).save(out_path)


def export_alpha_cutout(
    image: Image.Image,
    mask: np.ndarray,
    box: tuple[int, int, int, int],
    out_path: Path,
) -> None:
    x, y, w, h = box
    rgba = image.convert("RGBA")
    alpha = Image.fromarray((mask.astype("uint8") * 255))
    rgba.putalpha(alpha)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rgba.crop((x, y, x + w, y + h)).save(out_path)
```

- [ ] **Step 5: Extend export coverage for alpha cutout**

```python
def test_export_alpha_cutout_saves_rgba(tmp_path: Path) -> None:
    image = Image.new("RGB", (4, 4), "white")
    mask = np.ones((4, 4), dtype=bool)
    out_path = tmp_path / "cutout.png"

    export_alpha_cutout(image, mask, (0, 0, 4, 4), out_path)

    saved = Image.open(out_path)
    assert saved.mode == "RGBA"
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_geometry.py tests/test_naming.py tests/test_export_writer.py -v`  
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/sam3_mask/utils src/sam3_mask/export tests/test_geometry.py tests/test_naming.py tests/test_export_writer.py
git commit -m "feat: add geometry naming and export helpers"
```

### Task 4: Define Model Interfaces And A Dummy Backend

**Files:**
- Create: `D:\Repository\sam3-mask\src\sam3_mask\models\types.py`
- Create: `D:\Repository\sam3-mask\src\sam3_mask\models\base.py`
- Create: `D:\Repository\sam3-mask\src\sam3_mask\models\proposals.py`
- Create: `D:\Repository\sam3-mask\src\sam3_mask\models\dummy_backend.py`
- Create: `D:\Repository\sam3-mask\tests\test_pipeline_dummy.py`

- [ ] **Step 1: Write the failing dummy-backend tests**

```python
from pathlib import Path

from PIL import Image

from sam3_mask.models.dummy_backend import DummyBackend


def test_dummy_backend_returns_prompt_labeled_object() -> None:
    backend = DummyBackend()
    image = Image.new("RGB", (8, 8), "white")

    results = backend.segment(image=image, labels=["face"], boxes=None, score_threshold=0.2)

    assert len(results) == 1
    assert results[0].label == "face"
    assert results[0].bbox == (1, 1, 4, 4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline_dummy.py::test_dummy_backend_returns_prompt_labeled_object -v`  
Expected: FAIL with import errors for model interfaces

- [ ] **Step 3: Implement backend contracts and dummy backend**

```python
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True, frozen=True)
class SegmentationResult:
    label: str
    score: float
    bbox: tuple[int, int, int, int]
    mask: np.ndarray
```

```python
from abc import ABC, abstractmethod
from PIL import Image

from sam3_mask.models.types import SegmentationResult


class ProposalProvider(ABC):
    @abstractmethod
    def propose(self, image: Image.Image, labels: list[str]) -> list[tuple[int, int, int, int]]:
        raise NotImplementedError


class SegmenterBackend(ABC):
    @abstractmethod
    def segment(
        self,
        image: Image.Image,
        labels: list[str],
        boxes: list[tuple[int, int, int, int]] | None,
        score_threshold: float,
    ) -> list[SegmentationResult]:
        raise NotImplementedError
```

```python
class NoopProposalProvider(ProposalProvider):
    def propose(self, image: Image.Image, labels: list[str]) -> list[tuple[int, int, int, int]]:
        return []
```

```python
import numpy as np
from PIL import Image

from sam3_mask.models.base import SegmenterBackend
from sam3_mask.models.types import SegmentationResult


class DummyBackend(SegmenterBackend):
    def segment(
        self,
        image: Image.Image,
        labels: list[str],
        boxes: list[tuple[int, int, int, int]] | None,
        score_threshold: float,
    ) -> list[SegmentationResult]:
        width, height = image.size
        mask = np.zeros((height, width), dtype=bool)
        mask[1:5, 1:5] = True
        label = labels[0]
        return [SegmentationResult(label=label, score=0.99, bbox=(1, 1, 4, 4), mask=mask)]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pipeline_dummy.py::test_dummy_backend_returns_prompt_labeled_object -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/sam3_mask/models tests/test_pipeline_dummy.py
git commit -m "feat: add backend interfaces and dummy backend"
```

### Task 5: Build The End-To-End Pipeline With Dummy Backend Coverage

**Files:**
- Create: `D:\Repository\sam3-mask\src\sam3_mask\pipeline\runner.py`
- Modify: `D:\Repository\sam3-mask\src\sam3_mask\export\writer.py`
- Modify: `D:\Repository\sam3-mask\tests\test_pipeline_dummy.py`

- [ ] **Step 1: Write the failing end-to-end pipeline test**

```python
from pathlib import Path

from PIL import Image

from sam3_mask.config.schema import AppConfig, InputConfig, OutputConfig, PromptsConfig
from sam3_mask.models.dummy_backend import DummyBackend
from sam3_mask.pipeline.runner import run_pipeline


def test_run_pipeline_exports_mirrored_lq_hr_outputs(tmp_path: Path) -> None:
    lq_root = tmp_path / "LQ"
    hr_root = tmp_path / "HR"
    (lq_root / "a" / "b").mkdir(parents=True)
    (hr_root / "a" / "b").mkdir(parents=True)
    Image.new("RGB", (8, 8), "white").save(lq_root / "a" / "b" / "1.png")
    Image.new("RGB", (16, 16), "white").save(hr_root / "a" / "b" / "1.png")

    config = AppConfig(
        input=InputConfig(lq_dir=lq_root, hr_dir=hr_root),
        output=OutputConfig(out_dir=tmp_path / "out", save_crop=True),
        prompts=PromptsConfig(labels=["face"], label_mode="single", score_threshold=0.2),
    )

    summary = run_pipeline(config=config, backend=DummyBackend())

    assert summary.processed == 1
    assert (tmp_path / "out" / "LQ" / "face" / "a_b_1_face_01.png").exists()
    assert (tmp_path / "out" / "HR" / "face" / "a_b_1_face_01.png").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline_dummy.py::test_run_pipeline_exports_mirrored_lq_hr_outputs -v`  
Expected: FAIL with `ImportError` for `run_pipeline`

- [ ] **Step 3: Implement the pipeline runner**

```python
from dataclasses import dataclass

from PIL import Image

from sam3_mask.export.writer import export_bbox_crop
from sam3_mask.pairing.pairs import build_pairs, read_pair_scale
from sam3_mask.scanner.filesystem import scan_images
from sam3_mask.utils.geometry import clamp_box, scale_box
from sam3_mask.utils.naming import build_object_stem


@dataclass(slots=True)
class RunSummary:
    processed: int = 0
    skipped: int = 0
    failed: int = 0
    empty: int = 0


def run_pipeline(config, backend, proposal_provider=None) -> RunSummary:
    summary = RunSummary()
    lq_images = scan_images(config.input.lq_dir, config.input.exts)
    hr_images = scan_images(config.input.hr_dir, config.input.exts)
    pairs, _, _ = build_pairs(lq_images, hr_images)
    for pair in pairs:
        with Image.open(pair.lq_path) as lq_image, Image.open(pair.hr_path) as hr_image:
            scale_x, scale_y = read_pair_scale(pair.lq_path, pair.hr_path)
            boxes = proposal_provider.propose(lq_image, config.prompts.labels) if proposal_provider else None
            results = backend.segment(lq_image, config.prompts.labels, boxes, config.prompts.score_threshold)
            if not results:
                summary.empty += 1
                continue
            for index, result in enumerate(results, start=1):
                lq_box = clamp_box(result.bbox, lq_image.width, lq_image.height)
                hr_box = clamp_box(scale_box(result.bbox, scale_x, scale_y), hr_image.width, hr_image.height)
                stem = build_object_stem(pair.relative_path.as_posix(), result.label, index)
                export_bbox_crop(lq_image, lq_box, config.output.out_dir / "LQ" / result.label / f"{stem}.png")
                export_bbox_crop(hr_image, hr_box, config.output.out_dir / "HR" / result.label / f"{stem}.png")
            summary.processed += 1
    return summary
```

- [ ] **Step 4: Extend the same test for manifest generation**

```python
def test_run_pipeline_writes_object_manifest(tmp_path: Path) -> None:
    lq_root = tmp_path / "LQ"
    hr_root = tmp_path / "HR"
    (lq_root / "a" / "b").mkdir(parents=True)
    (hr_root / "a" / "b").mkdir(parents=True)
    Image.new("RGB", (8, 8), "white").save(lq_root / "a" / "b" / "1.png")
    Image.new("RGB", (16, 16), "white").save(hr_root / "a" / "b" / "1.png")

    config = AppConfig(
        input=InputConfig(lq_dir=lq_root, hr_dir=hr_root),
        output=OutputConfig(out_dir=tmp_path / "out", save_crop=True),
        prompts=PromptsConfig(labels=["face"], label_mode="single", score_threshold=0.2),
    )

    summary = run_pipeline(config=config, backend=DummyBackend())
    manifest = (tmp_path / "out" / "manifests" / "objects.jsonl").read_text(encoding="utf-8")
    assert '"label": "face"' in manifest
    assert '"relative_path": "a/b/1.png"' in manifest
```

- [ ] **Step 5: Implement manifest writing in the export layer**

```python
import json


def append_jsonl_record(out_path: Path, record: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
```

```python
append_jsonl_record(
    config.output.out_dir / "manifests" / "objects.jsonl",
    {
        "relative_path": pair.relative_path.as_posix(),
        "label": result.label,
        "bbox_lq": lq_box,
        "bbox_hr": hr_box,
    },
)
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_pipeline_dummy.py -v`  
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/sam3_mask/pipeline/runner.py src/sam3_mask/export/writer.py tests/test_pipeline_dummy.py
git commit -m "feat: add end-to-end pipeline with dummy backend"
```

### Task 6: Add Real SAM3 Backend Wiring And Graceful Runtime Fallback

**Files:**
- Create: `D:\Repository\sam3-mask\src\sam3_mask\models\sam3_backend.py`
- Modify: `D:\Repository\sam3-mask\src\sam3_mask\config\schema.py`
- Modify: `D:\Repository\sam3-mask\src\sam3_mask\config\loader.py`
- Modify: `D:\Repository\sam3-mask\src\sam3_mask\main.py`
- Modify: `D:\Repository\sam3-mask\README.md`

- [ ] **Step 1: Write the failing backend-selection test**

```python
import pytest

from sam3_mask.models.sam3_backend import SAM3Backend


def test_sam3_backend_raises_clear_error_when_dependency_missing() -> None:
    with pytest.raises(RuntimeError, match="SAM3 backend is not available"):
        SAM3Backend.from_config(device="cpu", checkpoint="missing.pt")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline_dummy.py::test_sam3_backend_raises_clear_error_when_dependency_missing -v`  
Expected: FAIL because `SAM3Backend` does not exist

- [ ] **Step 3: Implement a guarded SAM3 backend**

```python
class SAM3Backend(SegmenterBackend):
    def __init__(self, predictor) -> None:
        self._predictor = predictor

    @classmethod
    def from_config(cls, device: str, checkpoint: str) -> "SAM3Backend":
        try:
            from sam3 import build_sam3_predictor
        except Exception as exc:
            raise RuntimeError("SAM3 backend is not available in this environment") from exc
        predictor = build_sam3_predictor(checkpoint=checkpoint, device=device)
        return cls(predictor)

    def segment(self, image, labels, boxes, score_threshold):
        if not hasattr(self._predictor, "set_image"):
            raise RuntimeError("SAM3 predictor is missing set_image(); verify the installed SAM3 package API")
        self._predictor.set_image(image)
        predictions = []
        for label in labels:
            if hasattr(self._predictor, "predict_text"):
                outputs = self._predictor.predict_text(label=label, boxes=boxes, score_threshold=score_threshold)
            elif hasattr(self._predictor, "predict"):
                outputs = self._predictor.predict(text_prompt=label, boxes=boxes, score_threshold=score_threshold)
            else:
                raise RuntimeError("SAM3 predictor does not expose a supported text-prompt inference method")
            for item in outputs:
                predictions.append(
                    SegmentationResult(
                        label=label,
                        score=float(item["score"]),
                        bbox=tuple(int(v) for v in item["bbox"]),
                        mask=item["mask"],
                    )
                )
        return predictions
```

- [ ] **Step 4: Add runtime backend selection**

```python
@dataclass(slots=True)
class ModelConfig:
    backend: str = "dummy"
    proposal_provider: str = "none"
    checkpoint: str = ""
    device: str = "cpu"
```

```python
def build_backend(config):
    if config.model.backend == "dummy":
        return DummyBackend()
    if config.model.backend == "sam3":
        return SAM3Backend.from_config(
            device=config.model.device,
            checkpoint=config.model.checkpoint,
        )
    raise ValueError(f"Unsupported backend: {config.model.backend}")
```

- [ ] **Step 5: Document Windows/Linux setup**

```markdown
## Local CPU Development

Use the dummy backend by default:

```bash
python -m sam3_mask.main --config configs/default.yaml
```

## Linux GPU Execution

Switch to the SAM3 backend only after installing the target dependency stack and checkpoint:

```bash
python -m sam3_mask.main --config configs/default.yaml --backend sam3
```
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_config.py tests/test_pairing.py tests/test_geometry.py tests/test_naming.py tests/test_export_writer.py tests/test_pipeline_dummy.py -v`  
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/sam3_mask/config src/sam3_mask/models/sam3_backend.py src/sam3_mask/main.py README.md
git commit -m "feat: add sam3 backend wiring and runtime fallback"
```

### Task 7: Finish CLI UX, Summary Outputs, And Smoke Checks

**Files:**
- Modify: `D:\Repository\sam3-mask\src\sam3_mask\cli.py`
- Modify: `D:\Repository\sam3-mask\src\sam3_mask\pipeline\runner.py`
- Modify: `D:\Repository\sam3-mask\src\sam3_mask\export\writer.py`
- Modify: `D:\Repository\sam3-mask\README.md`

- [ ] **Step 1: Write the failing CLI and summary tests**

```python
def test_run_pipeline_writes_summary_json(tmp_path: Path) -> None:
    lq_root = tmp_path / "LQ"
    hr_root = tmp_path / "HR"
    (lq_root / "a" / "b").mkdir(parents=True)
    (hr_root / "a" / "b").mkdir(parents=True)
    Image.new("RGB", (8, 8), "white").save(lq_root / "a" / "b" / "1.png")
    Image.new("RGB", (16, 16), "white").save(hr_root / "a" / "b" / "1.png")

    config = AppConfig(
        input=InputConfig(lq_dir=lq_root, hr_dir=hr_root),
        output=OutputConfig(out_dir=tmp_path / "out", save_crop=True),
        prompts=PromptsConfig(labels=["face"], label_mode="single", score_threshold=0.2),
    )

    summary = run_pipeline(config=config, backend=DummyBackend())
    summary_path = tmp_path / "out" / "manifests" / "summary.json"
    assert summary_path.exists()
    assert '"processed": 1' in summary_path.read_text(encoding="utf-8")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline_dummy.py::test_run_pipeline_writes_summary_json -v`  
Expected: FAIL because `summary.json` is not written yet

- [ ] **Step 3: Implement summary and pair-manifest writing**

```python
def write_summary(out_path: Path, summary: RunSummary) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "processed": summary.processed,
                "skipped": summary.skipped,
                "failed": summary.failed,
                "empty": summary.empty,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
```

```python
def write_pairs_csv(out_path: Path, pairs: list[ImagePair]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["relative_path", "lq_path", "hr_path"])
        for pair in pairs:
            writer.writerow([pair.relative_path.as_posix(), str(pair.lq_path), str(pair.hr_path)])
```

- [ ] **Step 4: Wire CLI overrides for backend selection**

```python
parser.add_argument("--backend", choices=["dummy", "sam3"])
parser.add_argument("--device")
parser.add_argument("--checkpoint")
parser.add_argument("--label-mode", choices=["single", "multi"])
```

```python
if args.backend:
    overrides["model.backend"] = args.backend
if args.device:
    overrides["model.device"] = args.device
if args.checkpoint:
    overrides["model.checkpoint"] = args.checkpoint
if args.label_mode:
    overrides["prompts.label_mode"] = args.label_mode
```

- [ ] **Step 5: Run the full test suite**

Run: `pytest tests -v`  
Expected: PASS

- [ ] **Step 6: Smoke check the CLI locally**

Run: `python -m sam3_mask.main --config configs/default.yaml --backend dummy`  
Expected: exit code `0` and creation of `output/manifests/summary.json`

- [ ] **Step 7: Commit**

```bash
git add src/sam3_mask/cli.py src/sam3_mask/pipeline/runner.py src/sam3_mask/export/writer.py README.md tests
git commit -m "feat: complete cli summaries and smoke checks"
```

## Self-Review

Spec coverage check:

- Windows/Linux compatibility: Tasks 1, 6, 7
- Strict relative-path pairing: Task 2
- LQ detection and HR coordinate mapping: Tasks 3 and 5
- Single-label and multi-label support: Tasks 1 and 7 add config/CLI, Task 5 is where pipeline branching should be implemented
- Bbox crop and alpha cutout support: Task 3
- Mirrored `LQ` and `HR` output structure: Task 5
- Manifest outputs: Tasks 5 and 7
- SAM3 primary backend and proposal-provider abstraction: Tasks 4 and 6
- CPU-friendly local development: Tasks 4, 6, 7

Placeholder scan:

- No `TBD`, `TODO`, or omitted setup blocks remain. The `SAM3Backend` task now specifies a concrete adapter strategy with API guards; the implementer still needs to validate the exact installed package surface on target hardware during execution.

Type consistency check:

- `AppConfig`, `RunSummary`, `ImagePair`, `SegmentationResult`, `ProposalProvider`, and `SegmenterBackend` names are consistent across tasks.
