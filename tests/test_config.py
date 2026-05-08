from pathlib import Path

import pytest

from sam3_mask.cli import build_overrides, parse_args
from sam3_mask.config.loader import load_config


def test_load_config_reads_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
input:
  lq_dir: LQ
  hr_dir: HR
prompts:
  labels: [face, bird]
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path, cli_overrides={})

    assert config.input.lq_dir == Path("LQ")
    assert config.input.hr_dir == Path("HR")
    assert config.prompts.labels == ["face", "bird"]


def test_load_config_applies_cli_overrides(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
input:
  lq_dir: LQ
  hr_dir: HR
prompts:
  labels: [face]
output:
  save_cutout: false
""".strip(),
        encoding="utf-8",
    )

    config = load_config(
        config_path,
        cli_overrides={
            "prompts.labels": ["face", "plant"],
            "output.save_cutout": True,
        },
    )

    assert config.prompts.labels == ["face", "plant"]
    assert config.output.save_cutout is True


def test_load_config_uses_default_output_values(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
input:
  lq_dir: ./LQ
  hr_dir: ./HR
prompts:
  labels: [texture]
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path, cli_overrides={})

    assert config.output.out_dir == Path("output")
    assert config.output.save_crop is True
    assert config.output.save_cutout is False
    assert config.output.save_mask is False
    assert config.output.save_overlay is False


def test_load_config_rejects_non_boolean_output_flags(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
input:
  lq_dir: ./LQ
  hr_dir: ./HR
output:
  save_cutout: "false"
prompts:
  labels: [texture]
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(TypeError, match="output.save_cutout"):
        load_config(config_path, cli_overrides={})


def test_load_config_rejects_non_list_extensions(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
input:
  lq_dir: ./LQ
  hr_dir: ./HR
  exts: ".png"
prompts:
  labels: [texture]
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(TypeError, match="input.exts"):
        load_config(config_path, cli_overrides={})


def test_cli_supports_bidirectional_save_cutout_override() -> None:
    args = parse_args(["--config", "config.yaml", "--no-save-cutout"])
    overrides = build_overrides(args)

    assert overrides["output.save_cutout"] is False


def test_cli_requires_at_least_one_label_value() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--config", "config.yaml", "--labels"])


def test_load_config_reads_model_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
input:
  lq_dir: ./LQ
  hr_dir: ./HR
prompts:
  labels: [face]
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path, cli_overrides={})

    assert config.model.backend == "dummy"
    assert config.model.proposal_provider == "none"
    assert config.model.device == "cpu"
    assert config.pairing.expected_scale == 2.0
    assert config.pairing.scale_tolerance == 0.05
    assert config.pairing.skip_scale_mismatch is True


def test_load_config_reads_pairing_values(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
input:
  lq_dir: ./LQ
  hr_dir: ./HR
prompts:
  labels: [face]
pairing:
  expected_scale: 4.0
  scale_tolerance: 0.1
  skip_scale_mismatch: false
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path, cli_overrides={})

    assert config.pairing.expected_scale == 4.0
    assert config.pairing.scale_tolerance == 0.1
    assert config.pairing.skip_scale_mismatch is False


def test_load_config_rejects_invalid_label_mode(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
input:
  lq_dir: ./LQ
  hr_dir: ./HR
prompts:
  labels: [face]
  label_mode: singel
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="prompts.label_mode"):
        load_config(config_path, cli_overrides={})


def test_cli_supports_backend_runtime_overrides() -> None:
    args = parse_args(
        [
            "--config",
            "config.yaml",
            "--backend",
            "sam3",
            "--device",
            "cuda:0",
            "--checkpoint",
            "model.pt",
            "--label-mode",
            "multi",
        ]
    )

    overrides = build_overrides(args)

    assert overrides["model.backend"] == "sam3"
    assert overrides["model.device"] == "cuda:0"
    assert overrides["model.checkpoint"] == "model.pt"
    assert overrides["prompts.label_mode"] == "multi"
