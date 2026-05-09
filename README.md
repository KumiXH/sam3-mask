# sam3-mask

Prompt-driven mask and crop extraction for paired super-resolution datasets.

The project now supports a two-stage dataset workflow on paired `LQ/HR` images:

1. Stage 1 writes one aggregate single-channel mask per `(image, label)` under `output/masks/...`
2. Stage 2 reads the saved masks and exports fixed-size `HR/LQ/mask` crops under `output/crops/...`

Multiple detected instances for the same label are merged into one mask. If nothing is detected for a label, an empty mask is still saved.

## Two-Stage Overview

```mermaid
flowchart LR
  A[LQ image] --> B[Stage 1: prompt-driven SAM3 segmentation]
  B --> C[Single-channel mask per label]
  C --> D[Stage 2: fixed-size crop generation]
  D --> E[HR 480 crop]
  D --> F[LQ 240 crop]
  D --> G[Mask 240 crop]
```

## Visual Examples

### Face Example

<table>
  <tr>
    <th>Sample</th>
    <th>LQ Input<br /><sub>2048×1536</sub></th>
    <th>Stage 1 Mask<br /><sub>2048×1536</sub></th>
    <th>Stage 2 LQ Crop<br /><sub>240×240</sub></th>
    <th>Stage 2 HR Crop<br /><sub>480×480</sub></th>
    <th>Stage 2 Mask Crop<br /><sub>240×240</sub></th>
  </tr>
  <tr>
    <td align="center">
      <code>0013</code>
    </td>
    <td align="center">
      <img src="docs/examples/face_0013_lq.jpg" width="120" alt="Face 0013 LQ input" />
    </td>
    <td align="center">
      <img src="docs/examples/face_0013_mask.png" width="120" alt="Face 0013 stage 1 mask" />
    </td>
    <td align="center">
      <img src="docs/examples/face_0013_lq_crop.png" width="120" alt="Face 0013 LQ crop" />
    </td>
    <td align="center">
      <img src="docs/examples/face_0013_hr_crop.png" width="120" alt="Face 0013 HR crop" />
    </td>
    <td align="center">
      <img src="docs/examples/face_0013_mask_crop.png" width="120" alt="Face 0013 mask crop" />
    </td>
  </tr>
  <tr>
    <td align="center">
      <code>0036</code>
    </td>
    <td align="center">
      <img src="docs/examples/face_0036_lq.jpg" width="120" alt="Face 0036 LQ input" />
    </td>
    <td align="center">
      <img src="docs/examples/face_0036_mask.png" width="120" alt="Face 0036 stage 1 mask" />
    </td>
    <td align="center">
      <img src="docs/examples/face_0036_lq_crop.png" width="120" alt="Face 0036 LQ crop" />
    </td>
    <td align="center">
      <img src="docs/examples/face_0036_hr_crop.png" width="120" alt="Face 0036 HR crop" />
    </td>
    <td align="center">
      <img src="docs/examples/face_0036_mask_crop.png" width="120" alt="Face 0036 mask crop" />
    </td>
  </tr>
  <tr>
    <td align="center">
      <code>0041</code>
    </td>
    <td align="center">
      <img src="docs/examples/face_0041_lq.jpg" width="120" alt="Face 0041 LQ input" />
    </td>
    <td align="center">
      <img src="docs/examples/face_0041_mask.png" width="120" alt="Face 0041 stage 1 mask" />
    </td>
    <td align="center">
      <img src="docs/examples/face_0041_lq_crop.png" width="120" alt="Face 0041 LQ crop" />
    </td>
    <td align="center">
      <img src="docs/examples/face_0041_hr_crop.png" width="120" alt="Face 0041 HR crop" />
    </td>
    <td align="center">
      <img src="docs/examples/face_0041_mask_crop.png" width="120" alt="Face 0041 mask crop" />
    </td>
  </tr>
  <tr>
    <td align="center">
      <code>0060</code>
    </td>
    <td align="center">
      <img src="docs/examples/face_0060_lq.jpg" width="120" alt="Face 0060 LQ input" />
    </td>
    <td align="center">
      <img src="docs/examples/face_0060_mask.png" width="120" alt="Face 0060 stage 1 mask" />
    </td>
    <td align="center">
      <img src="docs/examples/face_0060_lq_crop.png" width="120" alt="Face 0060 LQ crop" />
    </td>
    <td align="center">
      <img src="docs/examples/face_0060_hr_crop.png" width="120" alt="Face 0060 HR crop" />
    </td>
    <td align="center">
      <img src="docs/examples/face_0060_mask_crop.png" width="120" alt="Face 0060 mask crop" />
    </td>
  </tr>
</table>

### Plant Example

<table>
  <tr>
    <th>Sample</th>
    <th>LQ Input<br /><sub>2048×1536</sub></th>
    <th>Stage 1 Mask<br /><sub>2048×1536</sub></th>
    <th>Stage 2 LQ Crop<br /><sub>240×240</sub></th>
    <th>Stage 2 HR Crop<br /><sub>480×480</sub></th>
    <th>Stage 2 Mask Crop<br /><sub>240×240</sub></th>
  </tr>
  <tr>
    <td align="center">
      <code>0000</code>
    </td>
    <td align="center">
      <img src="docs/examples/plant_0000_lq.jpg" width="120" alt="Plant 0000 LQ input" />
    </td>
    <td align="center">
      <img src="docs/examples/plant_0000_mask.png" width="120" alt="Plant 0000 stage 1 mask" />
    </td>
    <td align="center">
      <img src="docs/examples/plant_0000_lq_crop.png" width="120" alt="Plant 0000 LQ crop" />
    </td>
    <td align="center">
      <img src="docs/examples/plant_0000_hr_crop.png" width="120" alt="Plant 0000 HR crop" />
    </td>
    <td align="center">
      <img src="docs/examples/plant_0000_mask_crop.png" width="120" alt="Plant 0000 mask crop" />
    </td>
  </tr>
  <tr>
    <td align="center">
      <code>0001</code>
    </td>
    <td align="center">
      <img src="docs/examples/plant_0001_lq.jpg" width="120" alt="Plant 0001 LQ input" />
    </td>
    <td align="center">
      <img src="docs/examples/plant_0001_mask.png" width="120" alt="Plant 0001 stage 1 mask" />
    </td>
    <td align="center">
      <img src="docs/examples/plant_0001_lq_crop.png" width="120" alt="Plant 0001 LQ crop" />
    </td>
    <td align="center">
      <img src="docs/examples/plant_0001_hr_crop.png" width="120" alt="Plant 0001 HR crop" />
    </td>
    <td align="center">
      <img src="docs/examples/plant_0001_mask_crop.png" width="120" alt="Plant 0001 mask crop" />
    </td>
  </tr>
  <tr>
    <td align="center">
      <code>0005</code>
    </td>
    <td align="center">
      <img src="docs/examples/plant_0005_lq.jpg" width="120" alt="Plant 0005 LQ input" />
    </td>
    <td align="center">
      <img src="docs/examples/plant_0005_mask.png" width="120" alt="Plant 0005 stage 1 mask" />
    </td>
    <td align="center">
      <img src="docs/examples/plant_0005_lq_crop.png" width="120" alt="Plant 0005 LQ crop" />
    </td>
    <td align="center">
      <img src="docs/examples/plant_0005_hr_crop.png" width="120" alt="Plant 0005 HR crop" />
    </td>
    <td align="center">
      <img src="docs/examples/plant_0005_mask_crop.png" width="120" alt="Plant 0005 mask crop" />
    </td>
  </tr>
  <tr>
    <td align="center">
      <code>0008</code>
    </td>
    <td align="center">
      <img src="docs/examples/plant_0008_lq.jpg" width="120" alt="Plant 0008 LQ input" />
    </td>
    <td align="center">
      <img src="docs/examples/plant_0008_mask.png" width="120" alt="Plant 0008 stage 1 mask" />
    </td>
    <td align="center">
      <img src="docs/examples/plant_0008_lq_crop.png" width="120" alt="Plant 0008 LQ crop" />
    </td>
    <td align="center">
      <img src="docs/examples/plant_0008_hr_crop.png" width="120" alt="Plant 0008 HR crop" />
    </td>
    <td align="center">
      <img src="docs/examples/plant_0008_mask_crop.png" width="120" alt="Plant 0008 mask crop" />
    </td>
  </tr>
</table>

## Why Two Stages

- Stage 1 saves masks once, so crop strategy changes do not require rerunning SAM3.
- Stage 2 is deterministic from saved masks and is convenient for dataset patch generation.
- One label produces one aggregate mask per image, even when that image contains multiple instances.

## Requirements

- Python 3.10+
- Windows or Linux

## Install

```bash
python -m pip install -e .[dev]
```

For a new Windows machine that needs the full SAM3 backend, use the setup script after cloning both repositories:

```powershell
git clone https://github.com/KumiXH/sam3-mask.git
git clone https://github.com/facebookresearch/sam3.git
cd sam3-mask
powershell -ExecutionPolicy Bypass -File .\scripts\setup_sam3_env.ps1 -Python python -Sam3RepoDir ..\sam3
```

This script explicitly installs `setuptools<70.0.0` because newer `setuptools` versions can break the official `sam3` import path with `No module named pkg_resources`.

For quick source-tree execution without installing, use:

```powershell
$env:PYTHONPATH = "src"
python -m sam3_mask.main --config configs/default.yaml
```

## Config

The default runtime config lives at `configs/default.yaml`.

Important options:

- `input.lq_dir` and `input.hr_dir`: paired dataset roots
- `prompts.labels`: label prompts such as `face`, `plant`, `architecture`
- `pipeline.stage`: `mask` or `crop`
- `crop.hr_crop_size`: fixed HR crop size, default `480`
- `pairing.expected_scale`: expected HR/LQ scale, default `2.0`
- `model.backend`: `dummy` locally, `sam3` when the SAM3 dependency stack is installed
- `model.checkpoint`: local checkpoint path such as `D:/repository/sam3-mask/sam3.pt`

CLI overrides:

```powershell
$env:PYTHONPATH = "src"
python -m sam3_mask.main --config configs/default.yaml --labels face plant
```

## Output

The pipeline writes label-scoped masks and crops plus manifests:

```text
output/
  masks/<label>/<relative_dir>/<filename>.png
  crops/<label>/hr_480/<relative_dir>/<filename>__r{row}_c{col}.png
  crops/<label>/lq_240/<relative_dir>/<filename>__r{row}_c{col}.png
  crops/<label>/mask_240/<relative_dir>/<filename>__r{row}_c{col}.png
  manifests/
    pairs.csv
    objects.jsonl
    summary.json
```

Notes:

- `manifests/summary.json` and `objects.jsonl` are rewritten on each run
- shared output roots are safe for `masks/<label>` and `crops/<label>`, but manifests are not label-scoped

## Local Verification

```powershell
$base = Join-Path $env:TEMP ('codex-pytest-' + [guid]::NewGuid().ToString())
python -m pytest tests/test_pipeline_dummy.py -v -p no:cacheprovider --basetemp=$base
```

Current expected result: all tests pass.

## SAM3 Backend

After installing the SAM3 dependency stack and placing a local checkpoint, run stage 1 and stage 2 separately:

```powershell
python -m sam3_mask.main --config configs/SR_HR_.yaml
python -m sam3_mask.main --config configs/SR_HR_crop.yaml
```

### Manual Environment Fixes

If a fresh machine reports:

```text
SAM3 backend is not available in this environment
```

or a direct SAM3 import reports:

```text
No module named pkg_resources
```

install or downgrade `setuptools` first:

```powershell
python -m pip install "setuptools<70.0.0"
```

Then verify the official SAM3 imports:

```powershell
python -c "import pkg_resources; from sam3.model_builder import build_sam3_image_model; from sam3.model.sam3_image_processor import Sam3Processor; print('ok')"
```

## Example Configs

The repository includes ready-to-run configs for the current dataset:

- `configs/SR_HR_.yaml`: shared labels mask stage
- `configs/SR_HR_crop.yaml`: shared labels crop stage
- `configs/SR_HR_plant.yaml` / `configs/SR_HR_plant_crop.yaml`: plant-only output root
- `configs/SR_HR_face_shared.yaml` / `configs/SR_HR_face_shared_crop.yaml`: face in shared output root
- `configs/SR_HR_plant_shared.yaml` / `configs/SR_HR_plant_shared_crop.yaml`: plant in shared output root
- `configs/SR_HR_architecture_shared.yaml` / `configs/SR_HR_architecture_shared_crop.yaml`: architecture in shared output root

## Notes

- The package uses a `src/` layout.
- `sam3.pt` is intentionally ignored by git.
- Stage 2 reads masks from the configured output root; run stage 1 first.
- Real CPU inference with SAM3 is expected to be slow; use GPU for practical runs.
