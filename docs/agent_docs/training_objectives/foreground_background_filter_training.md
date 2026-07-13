# Foreground/Background Crop Classifier (Post-Generation Filter)

## Purpose
Load this for the small crop classifier that validates generated layout
datasets. Goal: a generator produces an image **plus** its layout conditioning
(bbox + category). For each conditioning bbox, this classifier looks at the
generated pixels inside that box and decides object-vs-background (binary) or
exact-class-vs-background (multiclass). If the crop is classified as the expected
object, the bbox is **kept** as ground truth for downstream detector training;
otherwise the bbox is **discarded** from the layout. Images are always kept; only
annotations are removed.

Do not load this for the detector itself (`simple_yolo_detector_training.md`,
`yolo_detector_training.md`) or for the generator/RegionDiff/STAY sampling
internals unless the request also touches filtering.

## Mental model (end-to-end)
1. **Train** the classifier on **real** FLIR crops: positives = expanded GT
   boxes, negatives = random boxes with low IoU vs all GT (sampled background).
2. **Generate** a candidate dataset (image + COCO `annotations.json` carrying the
   conditioning bboxes used to drive generation).
3. **Audit**: for every annotation, expand its bbox by `context_ratio`, crop the
   generated image, resize to `input_size`, run the classifier.
4. **Filter**: keep annotations the classifier accepts (`is_positive`), drop the
   rest. Original annotations are preserved in `annotations_unfiltered.json`;
   `annotations.json` is overwritten with the kept set.
5. (Optional, RegionDiff) **Retry**: re-generate images whose invalid-instance
   ratio exceeds a threshold, up to `max_tries`.

## Primary files
- `src/models/foreground_background_classifier.py` — two compact 1-channel CNNs:
  `ForegroundBackgroundClassifier` (binary, single logit) and
  `MultiClassForegroundBackgroundClassifier` (N foreground classes + background).
- `src/core/data/foreground_background_dataset.py` — `ForegroundBackgroundCropDataset`
  (binary) and `MultiClassCropDataset`; build positive crops from GT, sample
  negatives via `_max_iou_xyxy` ≤ `negative_iou_threshold`, expand with
  `_expand_box_with_context`. Also `collate_foreground_background_batch` and
  `build_balanced_sample_weights`.
- `src/algorithms/training/foreground_background_utils.py` — metrics
  (`compute_binary_metrics`, size buckets tiny/small/medium_large via
  `size_bucket_name`), `select_best_threshold`, checkpoint IO.
- `scripts/standalone/train_fg_bg_classifier.py` — binary trainer.
- `scripts/standalone/train_multiclass_fg_bg_classifier.py` — multiclass trainer
  (per-class thresholds).
- `src/algorithms/inference/rare_layout_dataset_tools.py` — the audit engine:
  `audit_generated_layout_dataset` (core scoring), `expand_box_with_context`,
  `load_filter_from_run_or_checkpoint`, `auto_find_latest_filter_run`,
  `export_audit_results`, `resolve_filter_output_dir`.
- `scripts/standalone/filter_generated_layout_dataset.py` — standalone audit CLI
  (writes manifests; does not rewrite `annotations.json`).
- `src/algorithms/inference/regiondiff/audit_filtering.py` — production filter
  applied inside RegionDiff generation: `write_filtered_annotations_from_audit`
  (the actual keep/discard of GT), retry loops, sanity-check renders, FID/KID/MMD.

## Train the classifier

Binary:
```bash
python -m scripts.standalone.train_fg_bg_classifier \
    --dataset_id flir_private_proxy_alignment_v18 \
    --input_size 128 --context_ratio 1.25 \
    --negative_iou_threshold 0.01 --epochs 20 --batch_size 128
```

Multiclass (required for production RegionDiff filtering):
```bash
python -m scripts.standalone.train_multiclass_fg_bg_classifier \
    --dataset_id flir_private_proxy_alignment_v18 \
    --input_size 128 --context_ratio 1.25 --epochs 20
```

Key hparams (must match between train and audit):
- `input_size` — crop resize edge fed to the CNN.
- `context_ratio` — how much each GT box is expanded around its center before
  cropping (1.25 = +25% context). Used identically at train and audit time.
- `negative_iou_threshold`, `negative_max_retries` — background sampling.
- Binary threshold is **auto-selected** on val (`select_best_threshold`, F1) and
  saved as `chosen_threshold`. Multiclass saves `per_class_thresholds`.

Runs are written under:
- binary: `artifacts/checkpoints/foreground_background_filter/runs/<run>/`
- multiclass: `artifacts/checkpoints/multiclass_foreground_background_filter/runs/<run>/`

Each run has `checkpoints/best.pt` + `latest.pt`, `metrics/summary.json`,
`metrics/per_epoch.jsonl`, and `tensorboard/`. `summary.json` carries everything
the audit needs: `classifier_mode`, `input_size`, `context_ratio`,
`normalization_mode`, `chosen_threshold` (binary) or `per_class_thresholds` +
`background_class_index` + `model_index_to_category_id` (multiclass).

## Apply the filter

Standalone audit (writes manifests + stats, does **not** rewrite GT):
```bash
python -m scripts.standalone.filter_generated_layout_dataset \
    --config configs/auxiliary/generated_layout_audit/presets/multiclass_rare_layout_20260415_161641.yaml \
    --device cuda:0
```
Config keys: `generated_dataset_dir` (required), `filter_run_dir` **or**
`filter_checkpoint` (auto-finds latest run if both empty), optional `output_dir`,
`threshold` (binary override), `batch_size`. Outputs land under a resolved
`filter_audit`-style dir: `per_instance_manifest.jsonl`,
`per_image_manifest.jsonl`, and a `summary` with per-category/size stats.

Production filter (rewrites GT in place) runs inside RegionDiff generation via
`src/algorithms/inference/regiondiff/audit_filtering.py`. The keep/discard step
is `write_filtered_annotations_from_audit`:
- copies `annotations.json` → `annotations_unfiltered.json` (once),
- keeps only annotations whose audit row has `is_positive == True`,
- writes `metadata/filtered_annotation_summary.json`
  (`n_annotations_unfiltered`, `n_annotations`, `n_invalid_annotations_removed`).
Production requires a **multiclass** filter (it raises otherwise).

## Keep/discard decision (what `is_positive` means)
Computed in `audit_generated_layout_dataset`:
- **Binary**: `prob = sigmoid(logit)`; keep if `prob >= threshold`.
- **Multiclass**: `probs = softmax(logits)`; keep only if argmax equals the
  expected class **and** is not background **and**
  `prob[expected] >= per_class_thresholds[expected]`. Wrong-class or
  below-threshold or background-argmax → discard.

Per-image `valid_fraction = n_positive / n_instances` drives the optional retry:
RegionDiff regenerates images where `invalid_ratio >
retry.invalid_instance_ratio_threshold` until accepted or `max_tries` reached.

## Critical invariants / gotchas
- **`context_ratio` and `input_size` at audit time come from the run summary**,
  not from new flags — keep training/audit consistent or scores are meaningless.
- Generated images are loaded as `.npy` (grayscale, 1 channel); the audit forces
  single-channel (`_prepare_crop_batch` averages if needed). Detector docs assume
  RGB-stored thermal; here the filter pipeline is genuinely 1-channel.
- Crops are expanded around the box **center** then clipped to image bounds; tiny
  boxes near edges yield small crops — `recall_tiny`/`recall_small` buckets in the
  metrics track this weak spot.
- The standalone CLI only audits; only `write_filtered_annotations_from_audit`
  (RegionDiff path) actually mutates `annotations.json`. To filter a dataset
  produced outside RegionDiff, run the audit then call that helper, or wire it in.
- `annotations_unfiltered.json` is the source of truth for re-runs; deleting it
  loses the original layout.

## Sanity checks
`audit_filtering.py` renders, under the generated dataset dir:
- `sanity_checks/sample_*.png` — full frames, green=kept / red=discarded boxes
  with `expected -> predicted p=…` labels.
- `sanity_checks/{valid,invalid}_bbox_crops.png` — crop contact sheets.
- `layout_overlays/` — conditioning boxes that drove generation.
Use these to eyeball threshold/`context_ratio` choices before trusting the filter.
