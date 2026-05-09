# Two-Stage Mask And Crop Design

**Goal**

Evolve the current one-pass bbox crop pipeline into a two-stage dataset generation pipeline:
- Stage 1 generates per-label aggregate masks from LQ images and writes one single-channel PNG per source image, even when the mask is empty.
- Stage 2 reads those saved masks and exports fixed-size HR/LQ/mask crops from the shared output directory.

**Confirmed Requirements**

- Work on the existing repository and current SAM3-based detection flow.
- Same image and label can contain multiple objects, but each `(image, label)` pair produces exactly one aggregate mask that merges all detected instances.
- Stage selection is controlled from YAML.
- Stage 1 output path:
  - `output/masks/<label>/<relative_dir>/<filename>.png`
- Stage 2 output paths:
  - `output/crops/<label>/hr_480/<relative_dir>/<filename>__r{row}_c{col}.png`
  - `output/crops/<label>/lq_240/<relative_dir>/<filename>__r{row}_c{col}.png`
  - `output/crops/<label>/mask_240/<relative_dir>/<filename>__r{row}_c{col}.png`
- `480` is the HR crop size and must be configurable.
- LQ and mask crop sizes are derived from the LQ/HR scale ratio. With 2x pairing, HR `480` means LQ/mask `240`.
- If no object is recognized for a label, save an empty mask in stage 1.
- Stage 2 is manual and must consume the already saved masks rather than rerunning segmentation.

**Design**

1. Configuration
- Add a pipeline stage switch, e.g. `pipeline.stage: mask|crop`.
- Add crop settings with `hr_crop_size`.
- Keep existing prompt/model/pairing settings unchanged where possible.

2. Stage 1: Mask generation
- Reuse current pairing and segmentation setup.
- For each pair and each label, build a boolean aggregate mask by OR-ing all masks returned for that label.
- Save each label mask under `masks/<label>/.../<source_name>.png`.
- Write empty masks when no detections exist.
- Summary counters remain image-oriented and should not fail when detections are absent.

3. Stage 2: Crop generation from masks
- Read the per-label saved LQ masks from stage 1.
- For each non-empty mask, compute the foreground bounding box.
- Convert crop size from HR to LQ using the measured pair scale.
- If the foreground bbox is smaller than one crop window, center a single fixed-size crop around the bbox center and clamp to image bounds.
- If the foreground bbox is larger than the crop window, tile it using non-overlapping windows anchored from the bbox min corner; when the final row/column would be undersized, shift the window to the far valid edge so the last crop is still full-sized and coverage is not wasted.
- Use the same LQ crop window for `LQ` and `mask`, and scale it to the paired HR window for `HR`.

4. Naming and structure
- Preserve the original relative subdirectories under each label bucket.
- Use stable tile names `__r{row}_c{col}` based on traversal order.

5. Testing
- Add focused tests for:
  - aggregate mask union across multiple detections
  - empty mask emission
  - stage 2 crop generation for small-object centering
  - stage 2 crop generation for large-object tiling and edge alignment
  - configured output path layout
