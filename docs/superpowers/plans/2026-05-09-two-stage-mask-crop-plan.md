# Two-Stage Mask Crop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace direct bbox export with a two-stage mask-first then crop-from-mask pipeline for dataset generation.

**Architecture:** Extend config/schema with explicit stage and crop settings, split pipeline execution into stage-specific paths, and add crop-window helpers that derive fixed-size LQ/HR windows from saved per-label aggregate masks. Keep SAM3 inference only in the mask stage and make crop stage deterministic from on-disk masks.

**Tech Stack:** Python, Pillow, NumPy, pytest, existing sam3-mask pipeline/config/export modules

---

### Task 1: Add config support for stages and crop sizing

**Files:**
- Modify: `D:\repository\sam3-mask\src\sam3_mask\config\schema.py`
- Modify: `D:\repository\sam3-mask\src\sam3_mask\config\loader.py`
- Test: `D:\repository\sam3-mask\tests\test_pipeline_dummy.py`

- [ ] Add failing tests for new config fields.
- [ ] Implement `pipeline.stage` and crop-size schema/loading.
- [ ] Run targeted config tests.

### Task 2: Add mask-stage export behavior

**Files:**
- Modify: `D:\repository\sam3-mask\src\sam3_mask\pipeline\runner.py`
- Modify: `D:\repository\sam3-mask\src\sam3_mask\export\writer.py`
- Test: `D:\repository\sam3-mask\tests\test_pipeline_dummy.py`

- [ ] Add failing tests for aggregate-mask union and empty-mask saving.
- [ ] Implement stage-1 per-label mask writing to `output/masks/<label>/...`.
- [ ] Run targeted mask-stage tests.

### Task 3: Add fixed-window crop generation from saved masks

**Files:**
- Modify: `D:\repository\sam3-mask\src\sam3_mask\pipeline\runner.py`
- Modify: `D:\repository\sam3-mask\src\sam3_mask\export\writer.py`
- Modify: `D:\repository\sam3-mask\src\sam3_mask\utils\geometry.py`
- Test: `D:\repository\sam3-mask\tests\test_pipeline_dummy.py`

- [ ] Add failing tests for centered small-object crop and tiled large-object crop.
- [ ] Implement crop-window calculation and stage-2 export layout.
- [ ] Run targeted crop-stage tests.

### Task 4: Verify end-to-end behavior

**Files:**
- Modify: `D:\repository\sam3-mask\configs\SR_HR_.yaml`
- Modify: `D:\repository\sam3-mask\configs\SR_HR_plant.yaml`
- Create: `D:\repository\sam3-mask\configs\SR_HR_stage1.yaml` (if needed)
- Create: `D:\repository\sam3-mask\configs\SR_HR_stage2.yaml` (if needed)

- [ ] Update example configs to the new schema.
- [ ] Run the full test file.
- [ ] Run a manual stage-1 config and confirm masks are written.
- [ ] Run a manual stage-2 config and confirm crops are written under `output/crops/<label>/...`.
