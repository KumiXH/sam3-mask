import builtins
from pathlib import Path

import pytest
from PIL import Image

from sam3_mask.config.loader import load_config
from sam3_mask.config.schema import AppConfig, CropConfig, InputConfig, OutputConfig, PipelineConfig, PromptsConfig
from sam3_mask.models.base import SegmenterBackend
from sam3_mask.models.dummy_backend import DummyBackend
from sam3_mask.models.proposals import NoopProposalProvider
from sam3_mask.models.sam3_backend import SAM3Backend
from sam3_mask.models.types import SegmentationResult
from sam3_mask.pipeline.runner import resolve_label_mode, run_pipeline


def test_dummy_backend_returns_prompt_labeled_object() -> None:
    backend = DummyBackend()
    image = Image.new("RGB", (8, 8), "white")

    results = backend.segment(image=image, labels=["face"], boxes=None, score_threshold=0.2)

    assert len(results) == 1
    assert results[0].label == "face"
    assert results[0].score == 0.99
    assert results[0].bbox == (1, 1, 4, 4)
    assert results[0].mask.shape == (8, 8)
    assert results[0].mask.dtype == bool


def test_dummy_backend_filters_by_score_threshold() -> None:
    backend = DummyBackend(score=0.1)
    image = Image.new("RGB", (8, 8), "white")

    results = backend.segment(image=image, labels=["face"], boxes=None, score_threshold=0.2)

    assert results == []


def test_noop_proposal_provider_returns_no_boxes() -> None:
    provider = NoopProposalProvider()
    image = Image.new("RGB", (8, 8), "white")

    assert provider.propose(image=image, labels=["face"]) == []


def test_load_config_reads_pipeline_stage_and_crop_size(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
input:
  lq_dir: /tmp/lq
  hr_dir: /tmp/hr
prompts:
  labels: [plant]
pipeline:
  stage: crop
crop:
  hr_crop_size: 512
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path, {})

    assert config.pipeline.stage == "crop"
    assert config.crop.hr_crop_size == 512


def test_run_pipeline_default_stage_writes_masks(tmp_path: Path) -> None:
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
    assert (tmp_path / "out" / "masks" / "face" / "a" / "b" / "1.png").exists()


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

    run_pipeline(config=config, backend=DummyBackend())

    manifest = (tmp_path / "out" / "manifests" / "objects.jsonl").read_text(encoding="utf-8")
    assert manifest == ""


def test_run_pipeline_rewrites_object_manifest_per_run(tmp_path: Path) -> None:
    lq_root = tmp_path / "LQ"
    hr_root = tmp_path / "HR"
    lq_root.mkdir()
    hr_root.mkdir()
    Image.new("RGB", (8, 8), "white").save(lq_root / "1.png")
    Image.new("RGB", (16, 16), "white").save(hr_root / "1.png")
    config = AppConfig(
        input=InputConfig(lq_dir=lq_root, hr_dir=hr_root),
        output=OutputConfig(out_dir=tmp_path / "out", save_crop=True),
        prompts=PromptsConfig(labels=["face"], label_mode="single", score_threshold=0.2),
    )

    run_pipeline(config=config, backend=DummyBackend())
    run_pipeline(config=config, backend=DummyBackend())

    lines = (tmp_path / "out" / "manifests" / "objects.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 0


class AggregateMaskBackend(SegmenterBackend):
    def segment(self, image, labels, boxes, score_threshold):
        import numpy as np

        mask_left = np.zeros((image.height, image.width), dtype=bool)
        mask_left[1:3, 1:3] = True
        mask_right = np.zeros((image.height, image.width), dtype=bool)
        mask_right[4:6, 4:6] = True
        return [
            SegmentationResult(label="face", score=0.8, bbox=(1, 1, 2, 2), mask=mask_left),
            SegmentationResult(label="face", score=0.9, bbox=(4, 4, 2, 2), mask=mask_right),
        ]


class EmptyMaskBackend(SegmenterBackend):
    def segment(self, image, labels, boxes, score_threshold):
        return []


def test_run_pipeline_mask_stage_saves_aggregate_label_mask(tmp_path: Path) -> None:
    lq_root = tmp_path / "LQ"
    hr_root = tmp_path / "HR"
    lq_root.mkdir()
    hr_root.mkdir()
    Image.new("RGB", (8, 8), "white").save(lq_root / "1.png")
    Image.new("RGB", (16, 16), "white").save(hr_root / "1.png")
    config = AppConfig(
        input=InputConfig(lq_dir=lq_root, hr_dir=hr_root),
        output=OutputConfig(out_dir=tmp_path / "out"),
        prompts=PromptsConfig(labels=["face"], label_mode="single", score_threshold=0.2),
        pipeline=PipelineConfig(stage="mask"),
    )

    run_pipeline(config=config, backend=AggregateMaskBackend())

    mask_path = tmp_path / "out" / "masks" / "face" / "1.png"
    assert mask_path.exists()
    mask = Image.open(mask_path)
    assert mask.getpixel((1, 1)) == 255
    assert mask.getpixel((4, 4)) == 255


def test_run_pipeline_mask_stage_saves_empty_mask_when_no_detection(tmp_path: Path) -> None:
    lq_root = tmp_path / "LQ"
    hr_root = tmp_path / "HR"
    lq_root.mkdir()
    hr_root.mkdir()
    Image.new("RGB", (8, 8), "white").save(lq_root / "1.png")
    Image.new("RGB", (16, 16), "white").save(hr_root / "1.png")
    config = AppConfig(
        input=InputConfig(lq_dir=lq_root, hr_dir=hr_root),
        output=OutputConfig(out_dir=tmp_path / "out"),
        prompts=PromptsConfig(labels=["face"], label_mode="single", score_threshold=0.2),
        pipeline=PipelineConfig(stage="mask"),
    )

    run_pipeline(config=config, backend=EmptyMaskBackend())

    mask_path = tmp_path / "out" / "masks" / "face" / "1.png"
    assert mask_path.exists()
    mask = Image.open(mask_path)
    assert mask.getbbox() is None


def test_run_pipeline_crop_stage_centers_small_mask_region_into_fixed_windows(tmp_path: Path) -> None:
    lq_root = tmp_path / "LQ"
    hr_root = tmp_path / "HR"
    mask_root = tmp_path / "out" / "masks" / "face"
    (lq_root / "nested").mkdir(parents=True)
    (hr_root / "nested").mkdir(parents=True)
    mask_root.mkdir(parents=True)
    Image.new("RGB", (8, 8), "white").save(lq_root / "nested" / "1.png")
    Image.new("RGB", (16, 16), "white").save(hr_root / "nested" / "1.png")

    mask = Image.new("L", (8, 8), 0)
    for x in range(3, 5):
        for y in range(3, 5):
            mask.putpixel((x, y), 255)
    (mask_root / "nested").mkdir(parents=True)
    mask.save(mask_root / "nested" / "1.png")

    config = AppConfig(
        input=InputConfig(lq_dir=lq_root, hr_dir=hr_root),
        output=OutputConfig(out_dir=tmp_path / "out"),
        prompts=PromptsConfig(labels=["face"], label_mode="single", score_threshold=0.2),
        pipeline=PipelineConfig(stage="crop"),
        crop=CropConfig(hr_crop_size=8),
    )

    summary = run_pipeline(config=config, backend=DummyBackend())

    assert summary.processed == 1
    lq_crop = tmp_path / "out" / "crops" / "face" / "lq_4" / "nested" / "1__r0_c0.png"
    hr_crop = tmp_path / "out" / "crops" / "face" / "hr_8" / "nested" / "1__r0_c0.png"
    mask_crop = tmp_path / "out" / "crops" / "face" / "mask_4" / "nested" / "1__r0_c0.png"
    assert lq_crop.exists()
    assert hr_crop.exists()
    assert mask_crop.exists()
    assert Image.open(lq_crop).size == (4, 4)
    assert Image.open(hr_crop).size == (8, 8)
    assert Image.open(mask_crop).size == (4, 4)


def test_run_pipeline_crop_stage_tiles_large_mask_without_overlap(tmp_path: Path) -> None:
    lq_root = tmp_path / "LQ"
    hr_root = tmp_path / "HR"
    mask_root = tmp_path / "out" / "masks" / "face"
    lq_root.mkdir()
    hr_root.mkdir()
    mask_root.mkdir(parents=True)
    Image.new("RGB", (12, 8), "white").save(lq_root / "1.png")
    Image.new("RGB", (24, 16), "white").save(hr_root / "1.png")

    mask = Image.new("L", (12, 8), 0)
    for x in range(1, 11):
        for y in range(1, 7):
            mask.putpixel((x, y), 255)
    mask.save(mask_root / "1.png")

    config = AppConfig(
        input=InputConfig(lq_dir=lq_root, hr_dir=hr_root),
        output=OutputConfig(out_dir=tmp_path / "out"),
        prompts=PromptsConfig(labels=["face"], label_mode="single", score_threshold=0.2),
        pipeline=PipelineConfig(stage="crop"),
        crop=CropConfig(hr_crop_size=8),
    )

    summary = run_pipeline(config=config, backend=DummyBackend())

    assert summary.processed == 1
    assert (tmp_path / "out" / "crops" / "face" / "lq_4" / "1__r0_c0.png").exists()
    assert (tmp_path / "out" / "crops" / "face" / "lq_4" / "1__r0_c1.png").exists()
    assert (tmp_path / "out" / "crops" / "face" / "lq_4" / "1__r1_c0.png").exists()
    assert (tmp_path / "out" / "crops" / "face" / "lq_4" / "1__r1_c1.png").exists()


class DuplicateObjectBackend(SegmenterBackend):
    def segment(self, image, labels, boxes, score_threshold):
        import numpy as np

        mask = np.ones((image.height, image.width), dtype=bool)
        return [
            SegmentationResult(label="face", score=0.8, bbox=(1, 1, 4, 4), mask=mask),
            SegmentationResult(label="person", score=0.9, bbox=(1, 1, 4, 4), mask=mask),
        ]


def test_resolve_label_mode_single_keeps_highest_score_for_duplicate_object() -> None:
    import numpy as np

    mask = np.ones((8, 8), dtype=bool)
    results = [
        SegmentationResult(label="face", score=0.8, bbox=(1, 1, 4, 4), mask=mask),
        SegmentationResult(label="person", score=0.9, bbox=(1, 1, 4, 4), mask=mask),
    ]

    resolved = resolve_label_mode(results, "single")

    assert len(resolved) == 1
    assert resolved[0].label == "person"


def test_resolve_label_mode_multi_keeps_all_duplicate_objects() -> None:
    import numpy as np

    mask = np.ones((8, 8), dtype=bool)
    results = [
        SegmentationResult(label="face", score=0.8, bbox=(1, 1, 4, 4), mask=mask),
        SegmentationResult(label="person", score=0.9, bbox=(1, 1, 4, 4), mask=mask),
    ]

    resolved = resolve_label_mode(results, "multi")

    assert len(resolved) == 2


def test_run_pipeline_writes_summary_json(tmp_path: Path) -> None:
    lq_root = tmp_path / "LQ"
    hr_root = tmp_path / "HR"
    (lq_root / "a").mkdir(parents=True)
    (hr_root / "a").mkdir(parents=True)
    Image.new("RGB", (8, 8), "white").save(lq_root / "a" / "1.png")
    Image.new("RGB", (16, 16), "white").save(hr_root / "a" / "1.png")
    config = AppConfig(
        input=InputConfig(lq_dir=lq_root, hr_dir=hr_root),
        output=OutputConfig(out_dir=tmp_path / "out", save_crop=True),
        prompts=PromptsConfig(labels=["face"], label_mode="single", score_threshold=0.2),
    )

    run_pipeline(config=config, backend=DummyBackend())

    summary_path = tmp_path / "out" / "manifests" / "summary.json"
    assert summary_path.exists()
    assert '"processed": 1' in summary_path.read_text(encoding="utf-8")


def test_run_pipeline_writes_pairs_csv(tmp_path: Path) -> None:
    lq_root = tmp_path / "LQ"
    hr_root = tmp_path / "HR"
    lq_root.mkdir()
    hr_root.mkdir()
    Image.new("RGB", (8, 8), "white").save(lq_root / "1.png")
    Image.new("RGB", (16, 16), "white").save(hr_root / "1.png")
    config = AppConfig(
        input=InputConfig(lq_dir=lq_root, hr_dir=hr_root),
        output=OutputConfig(out_dir=tmp_path / "out", save_crop=True),
        prompts=PromptsConfig(labels=["face"], label_mode="single", score_threshold=0.2),
    )

    run_pipeline(config=config, backend=DummyBackend())

    pairs_csv = (tmp_path / "out" / "manifests" / "pairs.csv").read_text(encoding="utf-8")
    assert "relative_path,lq_path,hr_path" in pairs_csv
    assert "1.png" in pairs_csv


def test_run_pipeline_skips_scale_mismatch(tmp_path: Path) -> None:
    lq_root = tmp_path / "LQ"
    hr_root = tmp_path / "HR"
    lq_root.mkdir()
    hr_root.mkdir()
    Image.new("RGB", (8, 8), "white").save(lq_root / "1.png")
    Image.new("RGB", (12, 12), "white").save(hr_root / "1.png")
    config = AppConfig(
        input=InputConfig(lq_dir=lq_root, hr_dir=hr_root),
        output=OutputConfig(out_dir=tmp_path / "out", save_crop=True),
        prompts=PromptsConfig(labels=["face"], label_mode="single", score_threshold=0.2),
    )

    summary = run_pipeline(config=config, backend=DummyBackend())

    assert summary.processed == 0
    assert summary.skipped == 1


def test_run_pipeline_continues_after_unreadable_image(tmp_path: Path) -> None:
    lq_root = tmp_path / "LQ"
    hr_root = tmp_path / "HR"
    lq_root.mkdir()
    hr_root.mkdir()
    (lq_root / "bad.png").write_text("not an image", encoding="utf-8")
    (hr_root / "bad.png").write_text("not an image", encoding="utf-8")
    config = AppConfig(
        input=InputConfig(lq_dir=lq_root, hr_dir=hr_root),
        output=OutputConfig(out_dir=tmp_path / "out", save_crop=True),
        prompts=PromptsConfig(labels=["face"], label_mode="single", score_threshold=0.2),
    )

    summary = run_pipeline(config=config, backend=DummyBackend())

    assert summary.failed == 1
    assert (tmp_path / "out" / "manifests" / "summary.json").exists()


def test_sam3_backend_raises_clear_error_when_dependency_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "sam3.model_builder" or name == "sam3.model.sam3_image_processor":
            raise ImportError("simulated missing dependency")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="SAM3 backend is not available"):
        SAM3Backend.from_config(device="cpu", checkpoint="missing.pt")


class FakeSam3Processor:
    def set_image(self, image):
        return {"image_size": image.size}

    def set_text_prompt(self, *, state, prompt):
        assert state["image_size"] == (10, 8)
        assert prompt == "face"
        state["boxes"] = [[1.2, 2.1, 7.9, 6.8]]
        state["scores"] = [0.95]
        state["masks"] = [[[False] * 10 for _ in range(8)]]
        state["masks"][0][2][1] = True
        state["masks"][0][6][7] = True
        return state


def test_sam3_backend_adapts_official_image_processor_state() -> None:
    backend = SAM3Backend(FakeSam3Processor())
    image = Image.new("RGB", (10, 8), "white")

    results = backend.segment(image=image, labels=["face"], boxes=None, score_threshold=0.2)

    assert len(results) == 1
    assert results[0].label == "face"
    assert results[0].score == 0.95
    assert results[0].bbox == (1, 2, 6, 4)
    assert results[0].mask.shape == (8, 10)
    assert results[0].mask.dtype == bool


class FakeSam3ProcessorWithChannelMask:
    def set_image(self, image):
        return {"image_size": image.size}

    def set_text_prompt(self, *, state, prompt):
        assert state["image_size"] == (10, 8)
        assert prompt == "face"
        state["boxes"] = [[1, 2, 8, 7]]
        state["scores"] = [0.95]
        state["masks"] = [[[[False] * 10 for _ in range(8)]]]
        state["masks"][0][0][2][1] = True
        state["masks"][0][0][6][7] = True
        return state


def test_sam3_backend_squeezes_single_channel_masks_from_official_processor() -> None:
    backend = SAM3Backend(FakeSam3ProcessorWithChannelMask())
    image = Image.new("RGB", (10, 8), "white")

    results = backend.segment(image=image, labels=["face"], boxes=None, score_threshold=0.2)

    assert len(results) == 1
    assert results[0].mask.shape == (8, 10)
    assert results[0].mask.dtype == bool
