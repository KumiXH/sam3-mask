from __future__ import annotations

from sam3_mask.config.schema import AppConfig
from sam3_mask.models.base import SegmenterBackend
from sam3_mask.models.dummy_backend import DummyBackend
from sam3_mask.models.sam3_backend import SAM3Backend


def build_backend(config: AppConfig) -> SegmenterBackend:
    if config.model.backend == "dummy":
        return DummyBackend()
    if config.model.backend == "sam3":
        return SAM3Backend.from_config(
            device=config.model.device,
            checkpoint=config.model.checkpoint,
        )
    raise ValueError(f"Unsupported backend: {config.model.backend}")
