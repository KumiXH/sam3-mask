from __future__ import annotations

import numpy as np
from PIL import Image

from sam3_mask.models.base import SegmenterBackend
from sam3_mask.models.types import SegmentationResult
from sam3_mask.utils.geometry import Box


class DummyBackend(SegmenterBackend):
    def __init__(self, score: float = 0.99, bbox: Box = (1, 1, 4, 4)) -> None:
        self._score = score
        self._bbox = bbox

    def segment(
        self,
        image: Image.Image,
        labels: list[str],
        boxes: list[Box] | None,
        score_threshold: float,
    ) -> list[SegmentationResult]:
        if not labels or self._score < score_threshold:
            return []

        width, height = image.size
        mask = np.zeros((height, width), dtype=bool)
        x, y, w, h = self._bbox
        mask[max(0, y) : min(height, y + h), max(0, x) : min(width, x + w)] = True
        return [
            SegmentationResult(
                label=labels[0],
                score=self._score,
                bbox=self._bbox,
                mask=mask,
            )
        ]
