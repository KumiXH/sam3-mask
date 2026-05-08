from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from sam3_mask.models.base import SegmenterBackend
from sam3_mask.models.types import SegmentationResult
from sam3_mask.utils.geometry import Box


class SAM3Backend(SegmenterBackend):
    def __init__(self, predictor: Any) -> None:
        self._predictor = predictor

    @classmethod
    def from_config(cls, device: str, checkpoint: str) -> "SAM3Backend":
        try:
            from sam3 import build_sam3_predictor  # type: ignore
        except Exception as exc:
            raise RuntimeError("SAM3 backend is not available in this environment") from exc

        try:
            predictor = build_sam3_predictor(checkpoint=checkpoint, device=device)
        except TypeError:
            predictor = build_sam3_predictor(checkpoint, device=device)
        return cls(predictor)

    def segment(
        self,
        image: Image.Image,
        labels: list[str],
        boxes: list[Box] | None,
        score_threshold: float,
    ) -> list[SegmentationResult]:
        if not labels:
            return []
        if hasattr(self._predictor, "set_image"):
            self._predictor.set_image(image)

        results: list[SegmentationResult] = []
        for label in labels:
            outputs = self._predict_label(label, boxes, score_threshold)
            results.extend(self._convert_outputs(label, outputs, score_threshold))
        return results

    def _predict_label(
        self,
        label: str,
        boxes: list[Box] | None,
        score_threshold: float,
    ) -> Any:
        if hasattr(self._predictor, "predict_text"):
            return self._predictor.predict_text(
                label=label,
                boxes=boxes,
                score_threshold=score_threshold,
            )
        if hasattr(self._predictor, "predict"):
            return self._predictor.predict(
                text_prompt=label,
                boxes=boxes,
                score_threshold=score_threshold,
            )
        raise RuntimeError("SAM3 predictor does not expose a supported text-prompt inference method")

    @staticmethod
    def _convert_outputs(
        label: str,
        outputs: Any,
        score_threshold: float,
    ) -> list[SegmentationResult]:
        items = outputs.values() if isinstance(outputs, dict) else outputs
        converted: list[SegmentationResult] = []
        for item in items:
            score = float(item["score"])
            if score < score_threshold:
                continue
            converted.append(
                SegmentationResult(
                    label=label,
                    score=score,
                    bbox=tuple(int(v) for v in item["bbox"]),
                    mask=np.asarray(item["mask"], dtype=bool),
                )
            )
        return converted
