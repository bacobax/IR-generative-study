# Handoff — FM generative augmentation for tiny YOLO (v18)

_Last updated: 2026-06-27_

## Goal

Use the trained LayoutFM model to generate synthetic images that augment the v18 YOLO
training set, then train a tiny (`simple_torch`) YOLO and compare against baselines. Two
generation strategies were built and evaluated; a held-out **test**-set comparison is the
open next step.

## TL;DR status

- **Two FM-augmentation methods implemented and fully run at 150 epochs**, plus 3 baselines.
- All runs **complete**. Best **val** mAP50: `fmaug 0.9777` > `rareaug 0.9769` > `fmbalaug 0.9721`
  > `genaug 0.9698` > `fast 0.9667`.
- **Open task (next step):** run `--action eval` on all 5 `best.pt` against the v18 **test**
  split for the final held-out verdict (all numbers so far are validation).

## The two FM-augmentation methods (both via Experiment B / `--action run_exp_b`)

Shared pipeline: offline **generate** synthetic images → **YOLO-adherence filter** → **merge**
real+synth → **train** tiny YOLO → **eval**. Only the layout source differs.

1. **`fm_aug`** (`small_v4_fmaug`): 1 synth image per real image, conditioned on that image's
   exact GT layout. Inherits the real (skewed) slice distribution.
2. **`fm_balanced_aug`** (`small_v4_fmbalaug`): builds NEW multi-box layouts that over-represent
   rare 27-slices (1 class × 3 size bins × 9 position bins) to flatten the combined real+synth
   distribution. Each synth box = a real box bootstrapped from a deficit slice, jittered in its
   3×3 cell, recombined into novel layouts; target = bring every slice up to the max real count.

Both use FM checkpoint **minmax ep570** and the **YOLO-adherence filter** (trained YOLOn detects
each synthetic box at its GT position; unmatched boxes dropped; empty images discarded).

## Results (150 epochs, validation mAP50 = `best_metric`)

| run | mAP50 | ep | mAP@[.5:.95] | mAP75 | notes |
|-----|-------|-----|------|-------|-------|
| small_v4_fmaug    | **0.9777** | 110 | 0.7787 | 0.8841 | best overall |
| small_v4_rareaug  | 0.9769 | 90 | 0.7666 | 0.8573 | rare-slice weighted sampler (no synth) |
| small_v4_fmbalaug | 0.9721 | 65 | 0.7457 | 0.8655 | balanced synth; 3rd |
| small_v4_genaug   | 0.9698 | 130 | 0.7621 | 0.8595 | generic ultralytics aug (no synth) |
| small_v4_fast     | 0.9667 | 35 | 0.7283 | 0.8317 | plain real-only |

**Key finding:** slice-balancing (`fm_balanced_aug`) underperformed plain `fm_aug`. The rare
slices targeted for balancing (tiny / central persons) are exactly what FM renders least
detectably → the YOLO filter dropped **29.6%** of balanced synth boxes (vs 6.5% for `fm_aug`),
eroding both image quality and the intended balance. Post-filter slice-distribution CV dropped
real 0.430 → merged 0.215 (pre-filter target was 0.036), so balance was real but halved by the
filter.

### Dataset composition after filter
- `fm_aug`: 6725 real + 6589 synth = **13314** train imgs (1092/16822 boxes dropped).
- `fm_balanced_aug`: 6725 real + 5328 synth = **12053** train imgs (4341/14660 boxes dropped,
  549 empty imgs discarded).

## Code map (all changes this session)

New/changed files:
- `src/algorithms/inference/yolo_adherence_filter.py` — **new.** YOLO-detection filter:
  `audit_generated_candidates_yolo()` runs the trained YOLO on each generated `.npy` (converted
  to uint8 RGB), greedy IoU-matches predictions to GT boxes, marks each GT `is_positive` if
  matched ≥ `adherence_iou`. Returns `instance_rows`/`image_rows`/`stats` matching the audit schema.
- `src/algorithms/inference/balanced_layout_generator.py` — **new.** `compute_slice_pool()` +
  `build_balanced_layouts()`: deficit pool over 27 slices → bootstrap + jitter + recombine real
  boxes into balanced multi-box `YOLOTrainSample` layouts. (Deferred import of `YOLOBox`/
  `YOLOTrainSample` from `yolo_experiment_b` to avoid a circular import.)
- `src/algorithms/training/yolo_experiment_b.py` — added: YOLO-filter dispatch
  (`audit_generated_candidates` branches on `filter.kind`), `_audit_generated_candidates_yolo`,
  `_write_filtered_annotations_from_audit(drop_empty_images=…)`, shared `_fm_generate_for_samples`
  core, `generate_fm_balanced_candidates`, `fm_balanced_aug` mode in `prepare_experiment_b_dataset`,
  validator updates.
- `src/core/configs/yolo_experiment_config.py` — `YOLOExperimentBFilterConfig` gained
  `kind`/`yolo_weights`/`adherence_iou`/`yolo_conf`/`discard_empty_images`; added
  `YOLOExperimentBBalancedConfig` (`target`/`jitter_frac`/`max_pair_iou`/`placement_tries`/`seed`)
  on `YOLOExperimentBConfig`.
- `configs/yolo/exp_v18_simple_yolo_tiny/small_v4_fmaug.yaml` — **new.**
- `configs/yolo/exp_v18_simple_yolo_tiny/small_v4_fmbalaug.yaml` — **new.**
- `scripts/run_yolo_150ep_compare.sh` — **new.** 2-GPU-lane orchestrator (≤2 GPUs).

Reused (pre-existing): `src/analysis/flir_subgroup/yolo_slice_stats.py` (27-slice machinery:
`load_yolo_slice_dataset`, `build_slice_counts_table`, size tertiles + 3×3 position bins);
`rare_layout_dataset_tools.py` (`load_sampler_from_pipeline`, `sample_layout_batch`,
`collate_layout_batch`); `export_augmented_yolo_dataset` (merge).

## Key gotchas (already worked around)

1. **YOLO trainer ignores `--device cuda:N`** — `cuda:1` lands on GPU0. Pin GPUs with
   `CUDA_VISIBLE_DEVICES=<n>` and pass `--device cuda:0`. (See `scripts/run_yolo_150ep_compare.sh`.)
2. **FM sampler preset must be the resolved `effective_config.yaml`** — `load_sampler_from_pipeline`
   uses a plain (non-`extends`) YAML loader, so the extends-based training preset lacks
   `model`/`training` keys. Both fmaug/fmbalaug configs point `fm.preset_path` at the run's
   `effective_config.yaml`.
3. **ultralytics can't read `.npy`** — generated images are `.npy`; the filter converts to uint8
   RGB (same stretch as export) before `model.predict`.
4. **`run_exp_b` action** is the correct `--action` (not `experiment_b`).

## Paths

- Configs: `configs/yolo/exp_v18_simple_yolo_tiny/{small_v4_fmaug,small_v4_fmbalaug,small_v4_fast,small_v4_genaug,small_v4_rareaug}.yaml`
- FM checkpoint: `artifacts/checkpoints/flow_matching/serious_runs/stay_layout_latent_v18_sd15ft_x8_256_minmax_reg_v2/UNET/unet_fm_epoch_570.pt`
- YOLO filter weights: `artifacts/checkpoints/yolo/exp_v18_scratch_yolo11n/default_aug/best.pt`
- Trained YOLO checkpoints: `artifacts/checkpoints/yolo/exp_v18_simple_yolo_tiny/<run>/best.pt`
- Train summaries: `artifacts/analysis/yolo/exp_v18_simple_yolo_tiny/<run>/train_summary.json`
- Merged datasets: `artifacts/generated/yolo/exp_b/augmented_yolo/<run>/<mode>/full_train_synthetic_aug.yaml`
- Real v18 YOLO data: `data/derived/yolo-test-ds_v18/{full_train,val,test}/` (6725 train imgs, 16822 boxes, 1 class `person`, 799 test)
- Logs: `artifacts/logs/cmp150_*.log`, `artifacts/logs/fmbalaug.log`

## Next step (the open task)

Held-out **test**-set comparison of all 5 runs. Each best checkpoint:
```bash
# example for one run; repeat for all 5 experiment_names
CUDA_VISIBLE_DEVICES=0 conda run -n diffusers-dev python -m src.cli.train_yolo \
  --action eval --device cuda:0 \
  --config configs/yolo/exp_v18_simple_yolo_tiny/<run>.yaml
```
`run_eval` uses `evaluation.dataset_yaml` (= v18 test split, already set in `_base_v4.yaml`) and the
run's `best.pt`. Collect test mAP/mAP50/mAP75 per run and rebuild the comparison table to confirm
whether the val ranking (fmaug > rareaug > fmbalaug > genaug > fast) holds on test.

## Possible follow-ups (not started)
- Over-generate rare-slice boxes in `fm_balanced_aug` to compensate for the ~30% filter drop, so
  post-filter balance approaches target.
- Loosen the YOLO filter (`adherence_iou`/`yolo_conf`) for rare slices, or skip filtering tiny boxes.
- Per-slice test-mAP breakdown to see if balancing helped specifically on rare slices even though
  global mAP didn't improve.
