from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from sam3_mask.config.schema import (
    AppConfig,
    CropConfig,
    DEFAULT_EXTENSIONS,
    InputConfig,
    ModelConfig,
    OutputConfig,
    PairingConfig,
    PipelineConfig,
    PromptsConfig,
)


def _deep_set(data: dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor = data
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        next_value = cursor.get(part)
        if next_value is None:
            next_value = {}
            cursor[part] = next_value
        if not isinstance(next_value, dict):
            raise TypeError(f"Cannot set nested config value for '{dotted_key}'.")
        cursor = next_value
    cursor[parts[-1]] = value


def _require_mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if value is None:
        raise KeyError(f"Missing required config section: {key}")
    if not isinstance(value, dict):
        raise TypeError(f"Config section '{key}' must be a mapping.")
    return value


def _require_bool(section: str, key: str, value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise TypeError(f"Config field '{section}.{key}' must be a boolean.")


def _require_string_list(section: str, key: str, value: Any, default: list[str] | None = None) -> list[str]:
    if value is None:
        return list(default or [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TypeError(f"Config field '{section}.{key}' must be a list of strings.")
    return list(value)


def load_config(path: Path, cli_overrides: dict[str, Any]) -> AppConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise TypeError("Top-level config must be a mapping.")

    merged = deepcopy(raw)
    for key, value in cli_overrides.items():
        _deep_set(merged, key, value)

    input_cfg = _require_mapping(merged, "input")
    prompts_cfg = _require_mapping(merged, "prompts")
    output_cfg = merged.get("output", {})
    if not isinstance(output_cfg, dict):
        raise TypeError("Config section 'output' must be a mapping.")
    model_cfg = merged.get("model", {})
    if not isinstance(model_cfg, dict):
        raise TypeError("Config section 'model' must be a mapping.")
    pairing_cfg = merged.get("pairing", {})
    if not isinstance(pairing_cfg, dict):
        raise TypeError("Config section 'pairing' must be a mapping.")
    pipeline_cfg = merged.get("pipeline", {})
    if not isinstance(pipeline_cfg, dict):
        raise TypeError("Config section 'pipeline' must be a mapping.")
    crop_cfg = merged.get("crop", {})
    if not isinstance(crop_cfg, dict):
        raise TypeError("Config section 'crop' must be a mapping.")

    labels = prompts_cfg.get("labels")
    if not isinstance(labels, list) or not labels:
        raise ValueError("Config field 'prompts.labels' must be a non-empty list.")
    label_mode = str(prompts_cfg.get("label_mode", "single"))
    if label_mode not in {"single", "multi"}:
        raise ValueError("Config field 'prompts.label_mode' must be one of: single, multi.")
    stage = str(pipeline_cfg.get("stage", "mask"))
    if stage not in {"mask", "crop"}:
        raise ValueError("Config field 'pipeline.stage' must be one of: mask, crop.")

    return AppConfig(
        input=InputConfig(
            lq_dir=Path(input_cfg["lq_dir"]),
            hr_dir=Path(input_cfg["hr_dir"]),
            exts=_require_string_list("input", "exts", input_cfg.get("exts"), DEFAULT_EXTENSIONS),
        ),
        output=OutputConfig(
            out_dir=Path(output_cfg.get("out_dir", "output")),
            save_crop=_require_bool("output", "save_crop", output_cfg.get("save_crop"), True),
            save_cutout=_require_bool("output", "save_cutout", output_cfg.get("save_cutout"), False),
            save_mask=_require_bool("output", "save_mask", output_cfg.get("save_mask"), False),
            save_overlay=_require_bool("output", "save_overlay", output_cfg.get("save_overlay"), False),
        ),
        prompts=PromptsConfig(
            labels=_require_string_list("prompts", "labels", labels),
            label_mode=label_mode,
            score_threshold=float(prompts_cfg.get("score_threshold", 0.25)),
        ),
        pairing=PairingConfig(
            expected_scale=float(pairing_cfg.get("expected_scale", 2.0)),
            scale_tolerance=float(pairing_cfg.get("scale_tolerance", 0.05)),
            skip_scale_mismatch=_require_bool(
                "pairing",
                "skip_scale_mismatch",
                pairing_cfg.get("skip_scale_mismatch"),
                True,
            ),
        ),
        model=ModelConfig(
            backend=str(model_cfg.get("backend", "dummy")),
            proposal_provider=str(model_cfg.get("proposal_provider", "none")),
            checkpoint=str(model_cfg.get("checkpoint", "")),
            device=str(model_cfg.get("device", "cpu")),
        ),
        pipeline=PipelineConfig(stage=stage),
        crop=CropConfig(hr_crop_size=int(crop_cfg.get("hr_crop_size", 480))),
    )
