# STAY-Style Pixel FM v2

## Summary

This version upgrades the working v1 bbox-conditioned pixel FM path toward a
more STAY-like architecture while keeping the same training loop, dataset path,
checkpointing, fixed-validation sampling, and TensorBoard workflow.

The main changes from v1 are:

- explicit per-object embeddings from class, bbox geometry, and optional style;
- learned per-object masks instead of only raw box raster maps;
- overlap-aware semantic assembly with smaller-object priority;
- edge-aware spatial maps;
- multi-scale Edge-Aware Normalization inside the pixel UNet;
- a lightweight masked object-context residual adapter as a practical
  approximation of Styled Mask Attention.

## Files Changed

- Model wrapper:
  [`stay_layout_conditioned_unet.py`](/projets/Fbassignana/diffusers_try/flow_matching_trial/src/models/stay_layout_conditioned_unet.py)
- Trainer integration and TensorBoard logging:
  [`layout_flow_matching_trainer.py`](/projets/Fbassignana/diffusers_try/flow_matching_trial/src/algorithms/training/layout_flow_matching_trainer.py)
- Fixed-style validation sampling:
  [`layout_flow_matching_sampler.py`](/projets/Fbassignana/diffusers_try/flow_matching_trial/src/algorithms/inference/layout_flow_matching_sampler.py)
- Layout config fields:
  [`fm_config.py`](/projets/Fbassignana/diffusers_try/flow_matching_trial/src/core/configs/fm_config.py)
- Visualization helpers:
  [`layout_debug.py`](/projets/Fbassignana/diffusers_try/flow_matching_trial/src/core/visualization/layout_debug.py)
- V2 presets:
  [`stay_layout_pixel_flir_v2_tiny.yaml`](/projets/Fbassignana/diffusers_try/flow_matching_trial/configs/fm/train/presets/stay_layout_pixel_flir_v2_tiny.yaml)
  and
  [`stay_layout_pixel_flir_v2_smoke200.yaml`](/projets/Fbassignana/diffusers_try/flow_matching_trial/configs/fm/train/presets/stay_layout_pixel_flir_v2_smoke200.yaml)
- V2 smoke launcher:
  [`stay_layout_pixel_flir_v2_smoke200.sh`](/projets/Fbassignana/diffusers_try/flow_matching_trial/scripts/train/stay_layout_pixel_flir_v2_smoke200.sh)
- V2 tests:
  [`test_stay_layout_fm.py`](/projets/Fbassignana/diffusers_try/flow_matching_trial/tests/test_stay_layout_fm.py)

## What Changed From v1

- v1 keeps `layout_conditioning.variant: "raster_v1"` and still uses
  bbox-filled raster conditioning concatenated to the pixel input.
- v2 adds `layout_conditioning.variant: "stay_v2"` and switches to:
  - explicit object embeddings;
  - learned object masks in local coordinates;
  - full-image soft and hard masks assembled from object boxes;
  - semantic and edge-aware maps;
  - multi-scale Edge-Aware Normalization instead of raw input concatenation.

## STAY Concepts Implemented

Implemented faithfully enough for this repo stage:

- explicit object representations from layout annotations;
- optional per-object style latents;
- self-supervised learned object masks;
- overlap-aware semantic ownership maps with smaller objects taking priority;
- edge-aware maps derived from mask borders;
- multi-scale Edge-Aware Normalization conditioning.

Approximations or deferred parts:

- the repo still uses flow matching rather than the paper's exact ADM training;
- mask learning is stabilized with lightweight self-supervised regularizers
  because segmentation labels are unavailable here;
- Styled Mask Attention is approximated with a masked object-context residual
  adapter instead of rewriting all backbone attention blocks.

## TensorBoard Additions

V1 visuals are preserved, and v2 adds:

- predicted soft mask composites;
- thresholded hard mask composites;
- non-overlap owner maps;
- semantic map previews;
- edge-aware map previews;
- masked context map previews;
- generated images with mask overlays;
- fixed validation panels comparing layout, masks, generated overlays, and
  ground truth overlays.

Scalars now also include:

- `layout_fm/fm_loss_step`
- `layout_fm/aux_loss_step`
- `layout_fm/mask_overlap_loss`
- `layout_fm/mask_sharpness_loss`
- `layout_fm/mask_activation_loss`
- `layout_fm/mask_mean`
- `layout_fm/overlap_ratio`
- `layout_fm/edge_map_energy`

## Commands

Tiny v2 verification run:

```bash
conda run -n diffusers-dev python -m src.cli.train \
  --config configs/fm/train/presets/stay_layout_pixel_flir_v2_tiny.yaml \
  --device cpu
```

Short CUDA smoke run:

```bash
conda run -n diffusers-dev python -m src.cli.train \
  --config configs/fm/train/presets/stay_layout_pixel_flir_v2_tiny.yaml \
  --max_train_samples 2 \
  --max_val_samples 2 \
  --epochs 1 \
  --device cuda
```

200-image / 50-epoch debug run:

```bash
bash scripts/train/stay_layout_pixel_flir_v2_smoke200.sh
```

TensorBoard:

```bash
tensorboard --logdir ./artifacts/runs/test/flow_matching/stay_layout_pixel_flir_v2_smoke200
```
