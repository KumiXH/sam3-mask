from pathlib import Path

import pytest
from PIL import Image

from sam3_mask.config.schema import AppConfig, InputConfig, OutputConfig, PromptsConfig
from sam3_mask.models.base import SegmenterBackend
from sam3_mask.models.dummy_backend import DummyBackend
from sam3_mask.models.proposals import NoopProposalProvider
from sam3_mask.models.sam3_backend import SAM3Backend
from sam3_mask.models.types import SegmentationResult
from sam3_mask.pipeline.runner import run_pipeline


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


def test_run_pipeline_exports_mirrored_lq_hr_outputs(tmp_path: Path) -> None:
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
    assert (tmp_path / "out" / "LQ" / "face" / "a_b_1_face_01.png").exists()
    assert (tmp_path / "out" / "HR" / "face" / "a_b_1_face_01.png").exists()


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
    assert '"label": "face"' in manifest
    assert '"relative_path": "a/b/1.png"' in manifest


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
    assert len(lines) == 1


class DuplicateObjectBackend(SegmenterBackend):
    def segment(self, image, labels, boxes, score_threshold):
        import numpy as np

        mask = np.ones((image.height, image.width), dtype=bool)
        return [
            SegmentationResult(label="face", score=0.8, bbox=(1, 1, 4, 4), mask=mask),
            SegmentationResult(label="person", score=0.9, bbox=(1, 1, 4, 4), mask=mask),
        ]


def test_run_pipeline_single_label_keeps_highest_score_for_duplicate_object(tmp_path: Path) -> None:
    lq_root = tmp_path / "LQ"
    hr_root = tmp_path / "HR"
    lq_root.mkdir()
    hr_root.mkdir()
    Image.new("RGB", (8, 8), "white").save(lq_root / "1.png")
    Image.new("RGB", (16, 16), "white").save(hr_root / "1.png")
    config = AppConfig(
        input=InputConfig(lq_dir=lq_root, hr_dir=hr_root),
        output=OutputConfig(out_dir=tmp_path / "out", save_crop=True),
        prompts=PromptsConfig(labels=["face", "person"], label_mode="single", score_threshold=0.2),
    )

    run_pipeline(config=config, backend=DuplicateObjectBackend())

    assert not (tmp_path / "out" / "LQ" / "face" / "1_face_01.png").exists()
    assert (tmp_path / "out" / "LQ" / "person" / "1_person_01.png").exists()


def test_run_pipeline_multi_label_exports_duplicate_object_to_each_label(tmp_path: Path) -> None:
    lq_root = tmp_path / "LQ"
    hr_root = tmp_path / "HR"
    lq_root.mkdir()
    hr_root.mkdir()
    Image.new("RGB", (8, 8), "white").save(lq_root / "1.png")
    Image.new("RGB", (16, 16), "white").save(hr_root / "1.png")
    config = AppConfig(
        input=InputConfig(lq_dir=lq_root, hr_dir=hr_root),
        output=OutputConfig(out_dir=tmp_path / "out", save_crop=True),
        prompts=PromptsConfig(labels=["face", "person"], label_mode="multi", score_threshold=0.2),
    )

    run_pipeline(config=config, backend=DuplicateObjectBackend())

    assert (tmp_path / "out" / "LQ" / "face" / "1_face_01.png").exists()
    assert (tmp_path / "out" / "LQ" / "person" / "1_person_02.png").exists()


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


def test_sam3_backend_raises_clear_error_when_dependency_missing() -> None:
    with pytest.raises(RuntimeError, match="SAM3 backend is not available"):
        SAM3Backend.from_config(device="cpu", checkpoint="missing.pt")
