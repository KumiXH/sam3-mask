from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sam3_mask.utils.geometry import Box


@dataclass(slots=True, frozen=True)
class SegmentationResult:
    label: str
    score: float
    bbox: Box
    mask: np.ndarray
