# SSDLite Detector Training

## Overview

`backend: ssdlite` — multi-scale anchor-based detector using a MobileNetV3-Small
backbone with depthwise-separable (SSDLite-style) prediction heads.

Key files:
- `src/models/ssdlite.py` — `SSDLiteConfig`, `SSDLiteDetector`, `generate_ssdlite_anchors`
- `src/algorithms/training/ssdlite_detector.py` — loss, decode, train/eval entry points
- `src/core/configs/yolo_experiment_config.py` — `SSDLiteModelConfig`, `YOLOModelConfig.ssdlite`
- `configs/yolo/exp_v18_ssdlite/` — base config + 9 variant configs

## Selecting the backend

```yaml
model:
  backend: ssdlite
  ssdlite:
    input_channels: 3
    ...
```

CLI override: `--model_backend ssdlite`

## Anchor / default box parameters (`model.ssdlite.*`)

| Key | Default | Meaning |
|---|---|---|
| `n_feature_maps` | `3` | Number of feature map scales (3–5). 3 → 32×32, 16×16, 8×8. |
| `anchor_min_sizes` | `[0.07, 0.15, 0.33]` | Normalized minimum anchor size per FM scale. |
| `anchor_max_sizes` | `[0.15, 0.33, 0.60]` | Normalized max size for the `sqrt(min*max)` anchor. |
| `anchor_aspect_ratios` | `[2.0]` | Aspect ratios `r`. Each `r` generates a wide (r:1) and tall (1:r) anchor. Total per cell = 2 + 2 × len(ratios). |
| `iou_pos_threshold` | `0.50` | IoU ≥ this → positive anchor match. |
| `iou_neg_threshold` | `0.40` | IoU < this → negative. Anchors between thresholds are ignored. |

With defaults (3 FMs, `anchor_aspect_ratios: [2.0]`):
- 4 anchors per cell: 1:1 at min_size, 1:1 at sqrt(min×max), 2:1, 1:2
- Total anchors: 32×32×4 + 16×16×4 + 8×8×4 = **5376**

## Loss parameters (`loss.*`)

| Key | Default | Used by SSDLite |
|---|---|---|
| `box_weight` | `5.0` | Scales Smooth L1 localization loss. |
| `class_weight` | `1.0` | Scales confidence BCE loss. |
| `neg_pos_ratio` | `3.0` | Hard negative mining ratio (negatives per positive). |
| `giou_weight` | `2.0` | **Ignored** by SSDLite. |
| `objectness_weight` | `1.0` | **Ignored** by SSDLite. |
| `no_object_weight` | `2.0` | **Ignored** by SSDLite. |

## Inference / NMS parameters (`model.ssdlite.*`)

| Key | Default | Meaning |
|---|---|---|
| `conf_threshold` | `0.25` | Minimum class confidence to keep a box during inference. |
| `nms_iou_threshold` | `0.45` | IoU threshold for per-class greedy NMS. |

During training-time validation, `conf_threshold=0.001` is used to maximize recall
(matches Tiny YOLO trainer behavior). The YAML `conf` and `iou` values under `evaluation:`
control the final eval and TensorBoard overlays.

## Launch commands

```bash
# Plain (no aug)
CUDA_VISIBLE_DEVICES=0 python -m src.cli.train_yolo \
  --action train --device cuda:0 --epochs 150 \
  --config configs/yolo/exp_v18_ssdlite/ssdlite_plain_no_aug.yaml

# Default augmentation
CUDA_VISIBLE_DEVICES=0 python -m src.cli.train_yolo \
  --action train --device cuda:0 --epochs 150 \
  --config configs/yolo/exp_v18_ssdlite/ssdlite_default_aug.yaml

# Rare-slice augmentation (baseline_b)
CUDA_VISIBLE_DEVICES=0 python -m src.cli.train_yolo \
  --action train --device cuda:0 --epochs 150 \
  --config configs/yolo/exp_v18_ssdlite/ssdlite_rare_slices_aug.yaml

# FM generative aug variants (action=run_exp_b)
CUDA_VISIBLE_DEVICES=0 python -m src.cli.train_yolo \
  --action run_exp_b --device cuda:0 --epochs 150 \
  --config configs/yolo/exp_v18_ssdlite/ssdlite_fm_1to1_no_filter.yaml

CUDA_VISIBLE_DEVICES=0 python -m src.cli.train_yolo \
  --action run_exp_b --device cuda:0 --epochs 150 \
  --config configs/yolo/exp_v18_ssdlite/ssdlite_fm_1to1_yolo_filter.yaml

CUDA_VISIBLE_DEVICES=0 python -m src.cli.train_yolo \
  --action run_exp_b --device cuda:0 --epochs 150 \
  --config configs/yolo/exp_v18_ssdlite/ssdlite_fm_1to1_classifier_filter.yaml

CUDA_VISIBLE_DEVICES=0 python -m src.cli.train_yolo \
  --action run_exp_b --device cuda:0 --epochs 150 \
  --config configs/yolo/exp_v18_ssdlite/ssdlite_fm_rare_balancing_no_filter.yaml

CUDA_VISIBLE_DEVICES=0 python -m src.cli.train_yolo \
  --action run_exp_b --device cuda:0 --epochs 150 \
  --config configs/yolo/exp_v18_ssdlite/ssdlite_fm_rare_balancing_yolo_filter.yaml

CUDA_VISIBLE_DEVICES=0 python -m src.cli.train_yolo \
  --action run_exp_b --device cuda:0 --epochs 150 \
  --config configs/yolo/exp_v18_ssdlite/ssdlite_fm_rare_balancing_classifier_filter.yaml

# Eval only
CUDA_VISIBLE_DEVICES=0 python -m src.cli.train_yolo \
  --action eval --device cuda:0 \
  --config configs/yolo/exp_v18_ssdlite/ssdlite_plain_no_aug.yaml
```

GPU convention: always pin with `CUDA_VISIBLE_DEVICES=N --device cuda:0`.

## Checkpoint format

Saved as `best.pt` and `last.pt` under `artifacts/checkpoints/yolo/exp_v18_ssdlite/<experiment_name>/`.

Checkpoint dict keys: `format="ssdlite_detector_v1"`, `epoch`, `model_config`,
`nc`, `names`, `model_state_dict`, `optimizer_state_dict`, `metrics`, `training_config`.

Load via `load_ssdlite_checkpoint(path, map_location=device)` → `(model, payload)`.

## Differences from TorchVision SSDLite

1. **Single backbone split**: MobileNetV3-Small features are split into 3 stages by
   spatial resolution (stride 8/16/32) using a dummy forward pass during `__init__`.
   TorchVision SSDLite uses `IntermediateLayerGetter`; this codebase uses
   `nn.Sequential` slicing for simplicity and no torchvision internals dependency.

2. **Channel projection**: A 1×1 conv + BN + ReLU6 per backbone stage normalizes
   all feature maps to 256 channels before the prediction heads, instead of adapting
   each head separately to the backbone channel count.

3. **Confidence loss**: Binary cross-entropy per class (sigmoid) rather than
   softmax over background + classes. Aligns with the single-class regime and
   avoids introducing an explicit background logit.

4. **Hard negative mining**: Mines `neg_pos_ratio × n_pos` negatives (sorted by
   BCE loss), with a floor of `min(30, available_negatives)` to ensure gradient
   flow during early training when anchors rarely match GT boxes.
