# sam3-mask

Prompt-driven object crop extraction for paired super-resolution datasets.

The first runnable version uses a local `dummy` backend for CPU-friendly Windows development and keeps a guarded `SAM3` backend hook for later Linux/V100 execution.

## Requirements

- Python 3.10+
- Windows or Linux

## Install

```bash
python -m pip install -e .[dev]
```

For quick source-tree execution without installing, use:

```powershell
$env:PYTHONPATH = "src"
python -m sam3_mask.main --config configs/default.yaml --backend dummy
```

## Config

The default runtime config lives at `configs/default.yaml`.

Important options:

- `input.lq_dir` and `input.hr_dir`: paired dataset roots
- `prompts.labels`: text prompts such as `face`, `bird`, `plant`, `texture`
- `output.save_crop`, `output.save_cutout`, `output.save_mask`: export toggles
- `pairing.expected_scale`: expected HR/LQ scale, default `2.0`
- `model.backend`: `dummy` locally, `sam3` when the SAM3 dependency stack is installed

CLI overrides:

```powershell
$env:PYTHONPATH = "src"
python -m sam3_mask.main --config configs/default.yaml --labels face bird plant --save-cutout
```

## Output

The pipeline writes mirrored category directories and manifests:

```text
output/
  LQ/<label>/
  HR/<label>/
  manifests/
    pairs.csv
    objects.jsonl
    summary.json
```

File stems flatten source subdirectories:

```text
a/b/1.png + face -> a_b_1_face_01.png
```

## Local Verification

```powershell
$base = Join-Path $env:TEMP ('codex-pytest-' + [guid]::NewGuid().ToString())
python -m pytest tests -v -p no:cacheprovider --basetemp=$base
```

Current expected result: all tests pass.

## SAM3 Backend

The `sam3` backend intentionally fails with a clear message when the SAM3 package is not installed. After installing the target SAM3 dependency stack and checkpoint on Linux, run:

```bash
python -m sam3_mask.main --config configs/default.yaml --backend sam3 --device cuda:0 --checkpoint /path/to/checkpoint.pt
```

## Notes

- The package uses a `src/` layout.
- Real CPU inference with SAM3 is expected to be slow; use `dummy` for local workflow checks.
- Large-scale inference is intended for the later Linux/V100 environment.
