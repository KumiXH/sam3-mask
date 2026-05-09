from __future__ import annotations

import numpy as np

from sam3_mask.config.schema import AppConfig
from sam3_mask.export.writer import (
    append_jsonl_record,
    export_alpha_cutout,
    export_bbox_crop,
    export_mask_crop,
    export_mask,
    export_single_channel_mask,
    write_pairs_csv,
    write_summary,
)
from sam3_mask.models.base import ProposalProvider, SegmenterBackend
from sam3_mask.pairing.pairs import build_pairs, read_pair_scale
from sam3_mask.pipeline.types import RunSummary
from sam3_mask.scanner.filesystem import scan_images
from sam3_mask.utils.geometry import clamp_box, fixed_windows_covering_box, mask_bbox, resize_mask, scale_box
from sam3_mask.utils.images import load_mask_image, load_rgb_image
from sam3_mask.utils.naming import build_object_stem, normalize_label


def run_pipeline(
    config: AppConfig,
    backend: SegmenterBackend,
    proposal_provider: ProposalProvider | None = None,
) -> RunSummary:
    if config.pipeline.stage == "mask":
        return run_mask_stage(config=config, backend=backend, proposal_provider=proposal_provider)
    if config.pipeline.stage == "crop":
        return run_crop_stage(config=config)

    summary = RunSummary()
    lq_images = scan_images(config.input.lq_dir, config.input.exts)
    hr_images = scan_images(config.input.hr_dir, config.input.exts)
    pairs, _, _ = build_pairs(lq_images, hr_images)
    manifests_dir = config.output.out_dir / "manifests"
    write_pairs_csv(manifests_dir / "pairs.csv", pairs)
    objects_path = manifests_dir / "objects.jsonl"
    objects_path.parent.mkdir(parents=True, exist_ok=True)
    objects_path.write_text("", encoding="utf-8")

    for pair in pairs:
        try:
            lq_image = load_rgb_image(pair.lq_path)
            hr_image = load_rgb_image(pair.hr_path)
            scale_x, scale_y = read_pair_scale(pair.lq_path, pair.hr_path)
            if _scale_mismatches(config, scale_x, scale_y):
                summary.skipped += 1
                continue
            boxes = proposal_provider.propose(lq_image, config.prompts.labels) if proposal_provider else None
            results = resolve_label_mode(
                backend.segment(lq_image, config.prompts.labels, boxes, config.prompts.score_threshold),
                config.prompts.label_mode,
            )
            if not results:
                summary.empty += 1
                continue

            for index, result in enumerate(results, start=1):
                label_dir = normalize_label(result.label)
                lq_box = clamp_box(result.bbox, lq_image.width, lq_image.height)
                hr_box = clamp_box(scale_box(result.bbox, scale_x, scale_y), hr_image.width, hr_image.height)
                stem = build_object_stem(pair.relative_path.as_posix(), result.label, index)

                lq_crop_path = config.output.out_dir / "LQ" / label_dir / f"{stem}.png"
                hr_crop_path = config.output.out_dir / "HR" / label_dir / f"{stem}.png"
                if config.output.save_crop:
                    export_bbox_crop(lq_image, lq_box, lq_crop_path)
                    export_bbox_crop(hr_image, hr_box, hr_crop_path)
                hr_mask = resize_mask(result.mask, hr_image.size)
                if config.output.save_cutout:
                    export_alpha_cutout(
                        lq_image,
                        result.mask,
                        lq_box,
                        config.output.out_dir / "LQ" / label_dir / f"{stem}_cutout.png",
                    )
                    export_alpha_cutout(
                        hr_image,
                        hr_mask,
                        hr_box,
                        config.output.out_dir / "HR" / label_dir / f"{stem}_cutout.png",
                    )
                if config.output.save_mask:
                    export_mask(result.mask, config.output.out_dir / "LQ" / label_dir / f"{stem}_mask.png")
                    export_mask(hr_mask, config.output.out_dir / "HR" / label_dir / f"{stem}_mask.png")

                append_jsonl_record(
                    objects_path,
                    {
                        "relative_path": pair.relative_path.as_posix(),
                        "label": result.label,
                        "score": result.score,
                        "bbox_lq": list(lq_box),
                        "bbox_hr": list(hr_box),
                        "lq_output": str(lq_crop_path),
                        "hr_output": str(hr_crop_path),
                    },
                )
            summary.processed += 1
        except Exception:
            summary.failed += 1
    write_summary(manifests_dir / "summary.json", summary)
    return summary


def run_mask_stage(
    config: AppConfig,
    backend: SegmenterBackend,
    proposal_provider: ProposalProvider | None = None,
) -> RunSummary:
    summary = RunSummary()
    lq_images = scan_images(config.input.lq_dir, config.input.exts)
    hr_images = scan_images(config.input.hr_dir, config.input.exts)
    pairs, _, _ = build_pairs(lq_images, hr_images)
    manifests_dir = config.output.out_dir / "manifests"
    write_pairs_csv(manifests_dir / "pairs.csv", pairs)
    objects_path = manifests_dir / "objects.jsonl"
    objects_path.parent.mkdir(parents=True, exist_ok=True)
    objects_path.write_text("", encoding="utf-8")

    for pair in pairs:
        try:
            lq_image = load_rgb_image(pair.lq_path)
            hr_image = load_rgb_image(pair.hr_path)
            scale_x, scale_y = read_pair_scale(pair.lq_path, pair.hr_path)
            if _scale_mismatches(config, scale_x, scale_y):
                summary.skipped += 1
                continue
            boxes = proposal_provider.propose(lq_image, config.prompts.labels) if proposal_provider else None
            raw_results = backend.segment(lq_image, config.prompts.labels, boxes, config.prompts.score_threshold)
            results_by_label = _group_results_by_label(raw_results, config.prompts.labels)

            detected_any = False
            for label in config.prompts.labels:
                aggregate_mask = _aggregate_label_mask(
                    results=results_by_label.get(label, []),
                    width=lq_image.width,
                    height=lq_image.height,
                )
                if aggregate_mask.any():
                    detected_any = True
                mask_path = config.output.out_dir / "masks" / normalize_label(label) / pair.relative_path.with_suffix(".png")
                export_single_channel_mask(aggregate_mask, mask_path)

            if detected_any:
                summary.processed += 1
            else:
                summary.empty += 1
        except Exception:
            summary.failed += 1

    write_summary(manifests_dir / "summary.json", summary)
    return summary


def run_crop_stage(config: AppConfig) -> RunSummary:
    summary = RunSummary()
    lq_images = scan_images(config.input.lq_dir, config.input.exts)
    hr_images = scan_images(config.input.hr_dir, config.input.exts)
    pairs, _, _ = build_pairs(lq_images, hr_images)
    manifests_dir = config.output.out_dir / "manifests"
    write_pairs_csv(manifests_dir / "pairs.csv", pairs)
    objects_path = manifests_dir / "objects.jsonl"
    objects_path.parent.mkdir(parents=True, exist_ok=True)
    objects_path.write_text("", encoding="utf-8")

    for pair in pairs:
        try:
            lq_image = load_rgb_image(pair.lq_path)
            hr_image = load_rgb_image(pair.hr_path)
            scale_x, scale_y = read_pair_scale(pair.lq_path, pair.hr_path)
            if _scale_mismatches(config, scale_x, scale_y):
                summary.skipped += 1
                continue

            hr_crop_size = config.crop.hr_crop_size
            lq_crop_width = max(1, round(hr_crop_size / scale_x))
            lq_crop_height = max(1, round(hr_crop_size / scale_y))
            if lq_crop_width != lq_crop_height:
                raise ValueError("Non-square LQ crop sizes are not supported.")
            lq_crop_size = lq_crop_width

            processed_any = False
            for label in config.prompts.labels:
                label_dir = normalize_label(label)
                mask_path = config.output.out_dir / "masks" / label_dir / pair.relative_path.with_suffix(".png")
                mask = load_mask_image(mask_path)
                bbox = mask_bbox(mask)
                if bbox is None:
                    continue

                windows = fixed_windows_covering_box(
                    box=bbox,
                    window_size=lq_crop_size,
                    width=lq_image.width,
                    height=lq_image.height,
                )
                for x, y, w, h, row, col in windows:
                    lq_box = clamp_box((x, y, w, h), lq_image.width, lq_image.height)
                    hr_box = clamp_box(scale_box(lq_box, scale_x, scale_y), hr_image.width, hr_image.height)
                    stem = f"{pair.relative_path.stem}__r{row}_c{col}.png"
                    relative_dir = pair.relative_path.parent
                    export_bbox_crop(
                        lq_image,
                        lq_box,
                        config.output.out_dir / "crops" / label_dir / f"lq_{lq_crop_size}" / relative_dir / stem,
                    )
                    export_bbox_crop(
                        hr_image,
                        hr_box,
                        config.output.out_dir / "crops" / label_dir / f"hr_{hr_crop_size}" / relative_dir / stem,
                    )
                    export_mask_crop(
                        mask,
                        lq_box,
                        config.output.out_dir / "crops" / label_dir / f"mask_{lq_crop_size}" / relative_dir / stem,
                    )
                    append_jsonl_record(
                        objects_path,
                        {
                            "relative_path": pair.relative_path.as_posix(),
                            "label": label,
                            "tile_row": row,
                            "tile_col": col,
                            "bbox_lq": list(lq_box),
                            "bbox_hr": list(hr_box),
                        },
                    )
                    processed_any = True

            if processed_any:
                summary.processed += 1
            else:
                summary.empty += 1
        except Exception:
            summary.failed += 1

    write_summary(manifests_dir / "summary.json", summary)
    return summary


def _scale_mismatches(config: AppConfig, scale_x: float, scale_y: float) -> bool:
    expected = config.pairing.expected_scale
    tolerance = config.pairing.scale_tolerance
    mismatch = abs(scale_x - expected) > tolerance or abs(scale_y - expected) > tolerance
    return config.pairing.skip_scale_mismatch and mismatch


def resolve_label_mode(results, label_mode: str):
    if label_mode == "multi":
        return results
    if label_mode != "single":
        raise ValueError(f"Unsupported label_mode: {label_mode}")

    best_by_box = {}
    for result in results:
        current = best_by_box.get(result.bbox)
        if current is None or result.score > current.score:
            best_by_box[result.bbox] = result
    return list(best_by_box.values())


def _group_results_by_label(results, labels: list[str]) -> dict[str, list]:
    grouped = {label: [] for label in labels}
    for result in results:
        grouped.setdefault(result.label, []).append(result)
    return grouped


def _aggregate_label_mask(results, width: int, height: int) -> np.ndarray:
    aggregate = np.zeros((height, width), dtype=bool)
    for result in results:
        aggregate |= np.asarray(result.mask, dtype=bool)
    return aggregate
