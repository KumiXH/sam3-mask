from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp"]


@dataclass(slots=True)
class InputConfig:
    lq_dir: Path
    hr_dir: Path
    exts: list[str] = field(default_factory=lambda: list(DEFAULT_EXTENSIONS))


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
class PairingConfig:
    expected_scale: float = 2.0
    scale_tolerance: float = 0.05
    skip_scale_mismatch: bool = True


@dataclass(slots=True)
class ModelConfig:
    backend: str = "dummy"
    proposal_provider: str = "none"
    checkpoint: str = ""
    device: str = "cpu"


@dataclass(slots=True)
class AppConfig:
    input: InputConfig
    output: OutputConfig
    prompts: PromptsConfig
    pairing: PairingConfig = field(default_factory=PairingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
