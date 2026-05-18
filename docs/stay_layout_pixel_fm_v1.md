# STAY-Style Pixel FM v1

## Summary

This first version adds a bbox-conditioned pixel-space flow-matching path on top
of the existing modular FM stack. It is intentionally simple and debuggable:

- generation happens directly in pixel space;
- conditioning comes from resized `boxes_xyxy` plus category labels;
- per-object class and bbox features are rasterized into dense spatial maps;
- those maps are concatenated to the noisy FM input before a pixel `UNet2DModel`;
- fixed validation layouts are sampled periodically and logged to TensorBoard.

## Main Architecture Choices

- Dataset path: [`AnnotationLayoutDataset`](/projets/Fbassignana/diffusers_try/flow_matching_trial/src/core/data/datasets.py)
  returns aligned `pixel_values`, `boxes_xyxy`, `labels`, and `label_names`.
- Batch path: [`layout_batching.py`](/projets/Fbassignana/diffusers_try/flow_matching_trial/src/core/data/layout_batching.py)
  pads variable numbers of objects and adds `boxes_xyxy_norm` plus `object_mask`.
- Model path: [`layout_conditioned_unet.py`](/projets/Fbassignana/diffusers_try/flow_matching_trial/src/models/layout_conditioned_unet.py)
  learns class embeddings and bbox features, rasterizes them into dense spatial
  conditioning maps, and concatenates them to the pixel FM input.
- Trainer path: [`layout_flow_matching_trainer.py`](/projets/Fbassignana/diffusers_try/flow_matching_trial/src/algorithms/training/layout_flow_matching_trainer.py)
  handles dict batches, fixed-layout validation sampling, TensorBoard image
  logging, and periodic debug panel saves.

## Conditioning Representation

- Input boxes are resized-image `xyxy` pixels from the dataset.
- The collate path also emits normalized boxes in `[0, 1]`.
- Each object uses:
  - a trainable class embedding,
  - a bbox MLP over `xyxy + cxcywh`,
  - fusion into a per-object feature vector.
- Rasterization fills each bbox region with the object feature, averages overlap
  regions, appends an objectness map, and projects the result with a small conv
  head into the final conditioning tensor.

## TensorBoard Logging

Scalars:

- `layout_fm/loss_step`
- `layout_fm/loss_epoch`
- `layout_fm/eval_loss_epoch`
- `layout_fm/lr`
- `layout_fm/grad_norm`
- `layout_fm/mean_objects`
- `layout_fm/max_objects`
- `layout_fm/empty_layout_fraction`
- `layout_fm/layout_coverage`

Images:

- training inputs
- training inputs with bbox overlays
- class-layout visualization
- objectness map
- conditioning feature map preview
- conditioning feature energy
- fixed validation generated images
- fixed validation generated images with bbox overlays
- fixed validation ground-truth with bbox overlays
- side-by-side panels:
  - class layout
  - generated image
  - generated image + boxes
  - ground truth + boxes

## Commands

Tiny verification run:

```bash
conda run -n diffusers-dev python -m src.cli.train_flow_matching \
  --config configs/fm/train/presets/stay_layout_pixel_flir_tiny.yaml \
  --device cpu
```

The tiny preset uses a dedicated lightweight 64x64 pixel UNet so it can serve as
a practical CPU verification pass.

200-image / 50-epoch debug run:

```bash
bash scripts/train/stay_layout_pixel_flir_smoke200.sh
```

TensorBoard:

```bash
tensorboard --logdir ./artifacts/runs/test/flow_matching/stay_layout_pixel_flir_smoke200
```
