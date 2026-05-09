from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image
import torch

from sam3_mask.models.base import SegmenterBackend
from sam3_mask.models.types import SegmentationResult
from sam3_mask.utils.geometry import Box


class SAM3Backend(SegmenterBackend):
    def __init__(self, predictor: Any) -> None:
        self._predictor = predictor

    @classmethod
    def from_config(cls, device: str, checkpoint: str) -> "SAM3Backend":
        try:
            from sam3.model_builder import build_sam3_image_model  # type: ignore
            from sam3.model.sam3_image_processor import Sam3Processor  # type: ignore
        except Exception as exc:
            raise RuntimeError("SAM3 backend is not available in this environment") from exc

        # The official builder only checks for the literal "cuda" when moving the
        # model, while the processor is fine with specific devices like "cuda:0".
        build_device = "cuda" if device.startswith("cuda") else device
        build_kwargs: dict[str, Any] = {"device": build_device}
        if checkpoint:
            build_kwargs["checkpoint_path"] = checkpoint
            build_kwargs["load_from_HF"] = False

        model = build_sam3_image_model(**build_kwargs)
        predictor = Sam3Processor(model=model, device=device)
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

        device = str(getattr(self._predictor, "device", "cpu"))
        use_cuda_autocast = device.startswith("cuda") and torch.cuda.is_available()
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if use_cuda_autocast
            else torch.autocast(device_type="cpu", enabled=False)
        )

        with autocast_context:
            state = None
            if hasattr(self._predictor, "set_image"):
                state = self._predictor.set_image(image)

            results: list[SegmentationResult] = []
            for label in labels:
                outputs = self._predict_label(label, boxes, score_threshold, state=state)
                results.extend(self._convert_outputs(label, outputs, score_threshold))
            return results

    def _predict_label(
        self,
        label: str,
        boxes: list[Box] | None,
        score_threshold: float,
        state: Any = None,
    ) -> Any:
        if hasattr(self._predictor, "set_text_prompt") and state is not None:
            return self._predictor.set_text_prompt(state=state, prompt=label)
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
        if isinstance(outputs, dict) and {"boxes", "scores", "masks"} <= outputs.keys():
            return SAM3Backend._convert_processor_state(label, outputs, score_threshold)

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

    @staticmethod
    def _convert_processor_state(
        label: str,
        state: dict[str, Any],
        score_threshold: float,
    ) -> list[SegmentationResult]:
        boxes = SAM3Backend._to_numpy(state["boxes"])
        scores = SAM3Backend._to_numpy(state["scores"])
        masks = SAM3Backend._to_numpy(state["masks"]).astype(bool)

        converted: list[SegmentationResult] = []
        for box, score, mask in zip(boxes, scores, masks):
            score_value = float(score)
            if score_value < score_threshold:
                continue
            mask_array = np.asarray(mask, dtype=bool)
            if mask_array.ndim == 3 and mask_array.shape[0] == 1:
                mask_array = mask_array[0]
            x0, y0, x1, y1 = [int(v) for v in box]
            converted.append(
                SegmentationResult(
                    label=label,
                    score=score_value,
                    bbox=(x0, y0, max(0, x1 - x0), max(0, y1 - y0)),
                    mask=mask_array,
                )
            )
        return converted

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            tensor = value.detach()
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.to(dtype=torch.float32)
            return tensor.cpu().numpy()
        return np.asarray(value)
