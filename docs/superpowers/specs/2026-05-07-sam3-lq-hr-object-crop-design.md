# SAM3 LQ/HR Object Crop Design

Date: 2026-05-07

## 1. Goal

Build a lightweight Python + PyTorch project that scans paired super-resolution dataset directories (`LQ` and `HR`), detects and segments prompt-defined object categories from `LQ` images using `SAM3`, and exports aligned object crops from both `LQ` and `HR`.

The project should:

- Run on Windows and Linux
- Support nested directories under both input roots
- Pair `LQ` and `HR` images by strict relative-path matching
- Use `LQ` coordinates as the canonical detection space
- Map crops and masks to `HR` by scale ratio
- Export results into mirrored `LQ` and `HR` output trees
- Allow prompt-driven category extension without code changes
- Support CPU-friendly local development and GPU-heavy Linux execution later

## 2. Non-Goals For V1

- No model training or fine-tuning
- No web UI
- No distributed framework
- No automatic prompt engineering
- No mandatory `GroundingDINO` integration in the first runnable version
- No promise of high-throughput CPU inference

## 3. High-Level Architecture

The project will use a simple package structure instead of a heavy framework.

```text
sam3-mask/
  configs/
    default.yaml
  src/
    main.py
    config/
    scanner/
    pairing/
    models/
    pipeline/
    export/
    utils/
  tests/
  docs/
```

Core responsibilities:

- `config`: YAML and CLI argument loading, runtime options, validation
- `scanner`: recursive image discovery under `LQ` and `HR`
- `pairing`: strict relative-path pairing and ratio validation
- `models`: inference backends and proposal-provider abstractions
- `pipeline`: orchestration of detection, segmentation, label resolution, crop generation
- `export`: file naming, output directory management, image/mask/cutout saving, manifest writing
- `utils`: shared image, geometry, logging, and path helpers

## 4. Model Strategy

V1 will use `SAM3` as the primary backend for prompt-based segmentation.

The codebase will still reserve a proposal-provider abstraction so that future versions can prepend a detector such as `GroundingDINO` without rewriting the rest of the pipeline.

Interfaces:

- `ProposalProvider`
  - Optional stage
  - Produces candidate boxes for an image and prompt list
- `SegmenterBackend`
  - Produces segmentation outputs from image, prompts, and optional boxes

V1 implementations:

- `NoopProposalProvider`
- `SAM3Backend`

Future extension:

- `GroundingDINOProposalProvider`

## 5. Input Data Rules

### 5.1 Directory Inputs

The user provides:

- `lq_dir`
- `hr_dir`

Both may contain:

- images directly under the root
- nested subdirectories
- empty folders
- unmatched files

### 5.2 Pairing Rule

Pairing is strict:

- `LQ/a/b/1.png` matches only `HR/a/b/1.png`

No fuzzy name matching is allowed in V1.

### 5.3 Scale Rule

The expected relation is `HR ~= 2x LQ`, but the implementation should not hardcode exactly `2.0`.

Behavior:

- compute width and height scale from actual image sizes
- validate against configured `expected_scale`
- allow a configurable tolerance
- warn or skip if scale is inconsistent

This keeps the logic robust for future datasets that are not exactly 2x.

## 6. Detection And Labeling Flow

Detection and segmentation run on `LQ` images only.

For each paired image:

1. Load `LQ`
2. Send prompt labels to `SAM3`
3. Receive object outputs containing at least:
   - `label`
   - `score`
   - `bbox`
   - `mask`
4. Resolve label assignment according to configuration
5. Export `LQ` crop/cutout/mask
6. Map geometry to `HR`
7. Export aligned `HR` crop/cutout/mask
8. Append object record to manifest

### 6.1 Label Modes

Supported label strategies:

- `single-label`
  - assign one category per object
  - default mode
- `multi-label`
  - allow one object to be exported into multiple category directories

## 7. Crop And Export Modes

Supported object export types:

- `bbox crop`
  - crop by the bounding rectangle of the object
  - default mode
- `alpha cutout`
  - preserve the object region with transparent background

Optional extra outputs:

- binary mask
- overlay visualization

The implementation should allow these to be toggled independently to control storage usage.

## 8. Output Layout

The output directory should keep `LQ` and `HR` trees mirrored by category.

```text
output/
  LQ/
    face/
    bird/
    plant/
    texture/
  HR/
    face/
    bird/
    plant/
    texture/
  manifests/
    pairs.csv
    objects.jsonl
    summary.json
  overlays/
    LQ/
    HR/
```

### 8.1 File Naming

Each exported object file name should follow:

`各级子目录名_原文件名_对象类型_序号xx`

Example:

`a_b_1_face_01.png`

The exact implementation rule:

- flatten relative parent directory names with `_`
- use original stem only, not the full extension
- append normalized object label
- append a two-digit sequence index per source image and label

## 9. Coordinate Mapping

All object geometry is defined in `LQ` space first.

`HR` geometry is derived by scale mapping:

- `x_hr = round(x_lq * scale_x)`
- `y_hr = round(y_lq * scale_y)`
- same for width and height

Masks should also be rescaled consistently before use in `HR` cutouts.

Clamping is required so exported boxes remain inside image bounds.

## 10. Runtime Strategy

### 10.1 Local Windows Development

The project must remain usable on a local Windows machine with only CPU or weak integrated graphics.

V1 local guarantees:

- code runs
- CLI works
- directory scanning works
- pairing works
- crop/export works
- manifests are generated
- tests run without CUDA

V1 local non-guarantee:

- high-throughput real `SAM3` inference on CPU

### 10.2 Linux GPU Execution

The target production environment can use one or multiple `V100` GPUs.

V1 should support:

- single-GPU execution
- multi-GPU task sharding by worker

### 10.3 Concurrency Model

Use a practical hybrid model:

- CPU workers for scanning, loading, saving, and post-processing
- limited GPU workers for model inference

Avoid spawning many model copies by default on small machines.

## 11. Error Handling

The pipeline should be sample-fault-tolerant.

Failure on one sample must not stop the full run.

Cases to handle:

- unreadable image
- missing paired file
- inconsistent scale ratio
- inference backend unavailable
- inference failure
- export failure
- empty detection result

Final summary should separate:

- `processed`
- `skipped`
- `failed`
- `empty`

## 12. Configuration

Use `YAML` plus CLI override.

Reference shape:

```yaml
input:
  lq_dir: /data/LQ
  hr_dir: /data/HR
  exts: [".png", ".jpg", ".jpeg", ".webp"]

output:
  out_dir: /data/output
  save_crop: true
  save_cutout: false
  save_mask: false
  save_overlay: false

prompts:
  labels:
    - face
    - bird
    - plant
    - texture
  label_mode: single
  score_threshold: 0.25

pairing:
  mode: strict_relative_path
  expected_scale: 2.0
  scale_tolerance: 0.05

runtime:
  device: cuda
  num_workers: 8
  gpu_ids: [0]

model:
  backend: sam3
  proposal_provider: none
  checkpoint: /models/sam3.pt
```

CLI must allow overriding prompt labels and selected export options without editing files.

## 13. Testing Strategy

Testing should be structured so the user can validate most of the project on a Windows CPU machine.

### 13.1 Unit Tests

Cover pure logic:

- path scan filtering
- strict relative pairing
- scale calculation
- bbox mapping
- file naming
- crop generation
- alpha cutout generation
- label-mode behavior

### 13.2 Integration Tests With Dummy Backend

Use a deterministic fake backend returning fixed boxes and masks.

Validate:

- full pipeline orchestration
- mirrored `LQ`/`HR` output structure
- manifest content
- error accounting

### 13.3 Real Backend Smoke Test

Optional small test:

- load `SAM3`
- run one or two small images
- confirm the runtime path is wired correctly

This can be skipped locally if the environment is not ready.

## 14. Delivery Scope For First Implementation

The first implementation should include:

- runnable CLI
- YAML config
- recursive scan and strict pairing
- `SAM3` backend interface and stub/mock-friendly abstraction
- `NoopProposalProvider`
- crop and cutout export
- mirrored output tree generation
- manifest generation
- CPU-friendly tests
- clear environment/setup instructions for Windows and Linux

## 15. Open Risks

- `SAM3` packaging and dependency maturity may affect Windows setup
- real CPU inference may be too slow for meaningful datasets
- prompt quality will strongly affect recall, especially for abstract categories such as texture/structure
- multi-label deduplication rules may need refinement after real data trials

## 16. Recommended Next Step

After this design is approved, write an implementation plan and then build V1 in a way that keeps:

- local Windows development friction low
- production Linux GPU execution straightforward
- future `GroundingDINO` integration easy
