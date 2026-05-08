from __future__ import annotations

from abc import ABC, abstractmethod

from PIL import Image

from sam3_mask.models.types import SegmentationResult
from sam3_mask.utils.geometry import Box


class ProposalProvider(ABC):
    @abstractmethod
    def propose(self, image: Image.Image, labels: list[str]) -> list[Box]:
        raise NotImplementedError


class SegmenterBackend(ABC):
    @abstractmethod
    def segment(
        self,
        image: Image.Image,
        labels: list[str],
        boxes: list[Box] | None,
        score_threshold: float,
    ) -> list[SegmentationResult]:
        raise NotImplementedError
