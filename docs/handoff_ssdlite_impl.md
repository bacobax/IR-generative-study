# Handoff: SSDLite Detector Implementation

## Context

You are working inside a PyTorch object-detection research codebase.
The current detector is a custom single-scale anchor-free Tiny YOLO implemented from scratch.
Your task is to add SSDLite as a **second detector option** that plugs into the same training pipeline.

Dataset: **v18** — thermal FLIR-style images, 256×256, single class (`person`).
- Train: 6 725 images (full_train split)
- Test: 799 images
- Labels: YOLO-format `.txt` files (class cx cy w h, normalized)

GPU convention: always pin with `CUDA_VISIBLE_DEVICES=N --device cuda:0`
(the simple_torch trainer ignores the physical index; use the env var).

---

## Key files to read first

| File | Purpose |
|---|---|
| `src/models/simple_yolo.py` | Tiny YOLO model + `SimpleYOLOConfig` — the template to mirror for SSDLite |
| `src/algorithms/training/simple_yolo_detector.py` | Full trainer, loss, dataset, collate, decode, eval — nearly everything to reuse |
| `src/core/configs/yolo_experiment_config.py` | All config dataclasses — add `SSDLiteModelConfig` and update `YOLOModelConfig` here |
| `src/cli/train_yolo.py` | CLI entrypoint — add `backend == "ssdlite"` dispatch at four marked spots |
| `src/evaluation/detection_metrics.py` | `DetectionPrediction`, `DetectionGroundTruth`, `evaluate_detections`, `nms_numpy` |
| `configs/yolo/exp_v18_simple_yolo_tiny/_base_v4.yaml` | Base config to `extends:` from for new SSD configs |

---

## Architecture of the existing simple_torch backend

### Training dispatch chain

```
src/cli/train_yolo.py: run_train(cfg)
  if cfg.model.backend == "simple_torch":          # line 1068
    → train_simple_yolo(cfg, ...)                  # simple_yolo_detector.py:945

src/cli/train_yolo.py: run_eval(cfg)
  if cfg.model.backend == "simple_torch":          # line 1211
    → eval_simple_yolo(cfg, ...)

src/cli/train_yolo.py: run_eval_slices(cfg)
  if cfg.model.backend == "simple_torch":          # line 1344
    → eval_simple_yolo_slices(cfg, ...)

src/cli/train_yolo.py: _validate_model_backend(cfg)  # line 791
  allowed = {"ultralytics", "simple_torch"}            # ← add "ssdlite" here
```

`run_exp_b` (FM generative augmentation) builds the augmented dataset first then calls
`run_train`, so it automatically inherits the backend dispatch.

### Model interface

```python
# src/models/simple_yolo.py

class SimpleYOLODetector(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x:    [B, C, H, W]  float32 in [0,1]
        # out:  [B, boxes_per_cell, 5 + nc, grid_h, grid_w]
        #        channels 0-3: raw tx,ty,tw,th logits (anchor-free)
        #        channel 4:    objectness logit
        #        channels 5+:  class logits

def count_trainable_parameters(model: nn.Module) -> int: ...
```

The SSDLite model forward output can differ from this shape — but it must be consumed
entirely by its own loss and decode functions that you write. The only shared contract
is the **batch dict** and the **DetectionPrediction/DetectionGroundTruth** types.

### Batch dict format (from `SimpleYoloDataset.__getitem__` + `simple_yolo_collate`)

```python
batch = {
    "images":      torch.Tensor,          # [B, C, H, W] float32 in [0,1]
    "boxes_xywh":  list[torch.Tensor],    # each (N_i, 4), normalized cx cy w h
    "class_ids":   list[torch.Tensor],    # each (N_i,),  int64
    "image_ids":   list[str],
    "image_paths": list[str],
}
```

`SimpleYoloDataset` is backend-agnostic — reuse it as-is for SSDLite.
Same for `simple_yolo_collate`. Import both.

### Evaluation contract

```python
# Outputs your model must produce for eval
from src.evaluation.detection_metrics import DetectionPrediction, DetectionGroundTruth

pred = DetectionPrediction(
    image_id="stem_name",
    boxes_xyxy=np.ndarray,   # (N, 4) normalized, float32
    scores=np.ndarray,       # (N,)   float32
    class_ids=np.ndarray,    # (N,)   int64
)
gt = DetectionGroundTruth(
    image_id="stem_name",
    boxes_xyxy=np.ndarray,   # (N, 4) normalized, float32
    class_ids=np.ndarray,    # (N,)   int64
)

metrics = evaluate_detections(predictions=preds, ground_truths=gts, names=names)
# metrics["summary"] → dict with map50, map, precision, recall
```

### Loss interface

```python
class SimpleYoloLoss(nn.Module):
    def forward(
        self,
        output: torch.Tensor,          # raw model output
        *,
        boxes_xywh: list[torch.Tensor],
        class_ids: list[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # returns (total_loss, {loss_key: float, ...})
```

The dict keys are logged to TensorBoard and the stdout progress line.
Mirror this signature for the SSDLite loss.

### Training loop (reuse entirely)

`train_simple_yolo` in `simple_yolo_detector.py:945` does:
1. Build `SimpleYoloDataset` (train + val)
2. Build dataloaders (with optional `WeightedRandomSampler`)
3. Instantiate model via `build_simple_yolo_model(cfg, nc=nc)`
4. `SimpleYoloLoss(cfg.loss)`
5. Optimizer + cosine scheduler
6. Epoch loop: forward → loss → backward → checkpoint → early-stop
7. Val every `cfg.training.val_interval` epochs via `evaluate_simple_yolo_model`

For SSDLite, write a `train_ssdlite(cfg, ...)` that does the same loop but with:
- `build_ssdlite_model(cfg, nc=nc)` instead of the YOLO model builder
- `SSDLiteLoss(cfg.loss)` for your SSD-style loss (focal/smoothL1 + hard-neg mining)
- `collect_ssdlite_predictions(model, ...)` that handles anchor decode + NMS

Alternatively, extract the shared training skeleton into a helper and call it from both.

---

## Config system

### How YAML keys map to config dataclasses

```
model.backend         → YOLOModelConfig.backend   (str)
model.simple.*        → SimpleYOLOModelConfig      (add new SSDLiteModelConfig here)
training.*            → YOLOTrainConfig
loss.*                → YOLOLossConfig
data.*                → YOLODataConfig
baseline.*            → YOLOBaselineConfig
augment.*             → inline dataclass (aug_cfg)
experiment_b.*        → YOLOExperimentBConfig
evaluation.*          → YOLOEvalConfig
output.*              → YOLOOutputConfig
```

All located in `src/core/configs/yolo_experiment_config.py`.

**Unknown YAML keys cause a hard error** (`_validate_config_yaml_keys` at line 341 in
`train_yolo.py`). Every new key you add to a YAML must have a corresponding dataclass field.

### What to add to `yolo_experiment_config.py`

```python
@dataclass
class SSDLiteModelConfig:
    input_channels: int = 3
    # anchor scales, aspect ratios, feature map strides, etc.
    # — design these to match what SSDLite.from_config() expects
    ...

@dataclass
class YOLOModelConfig:
    weights: str = "yolov8n.pt"
    task: str = "detect"
    backend: str = "ultralytics"
    simple: SimpleYOLOModelConfig = field(default_factory=SimpleYOLOModelConfig)
    ssdlite: SSDLiteModelConfig = field(default_factory=SSDLiteModelConfig)  # ← add
```

The existing `YOLOLossConfig` keys (`box_weight`, `giou_weight`, `objectness_weight`,
`no_object_weight`, `class_weight`) may not all apply to SSD.
Option A: reuse/ignore irrelevant keys.
Option B: add new keys for `localization_weight`, `neg_pos_ratio`, etc.
Either way: every key in `loss:` section must exist in `YOLOLossConfig`.

### Existing `_base_v4.yaml` (the base all variant configs extend)

```yaml
data:
  image_size: 256
  batch_size: 16
  workers: 8
  cache_images: true
  dataset_yaml: data/derived/yolo-test-ds_v18/full_train.yaml
  test_dataset_yaml: data/derived/yolo-test-ds_v18/test.yaml

model:
  backend: simple_torch          # ← change to "ssdlite" in new base
  simple:
    input_channels: 3
    base_channels: 32
    output_stride: 16
    boxes_per_cell: 2
    ...

training:
  epochs: 75
  lr0: 0.001
  optimizer: AdamW
  patience: 20
  cos_lr: true
  mixed_precision: auto
  grad_clip_norm: 10.0
  val_interval: 5

loss:
  box_weight: 5.0
  giou_weight: 2.0
  objectness_weight: 1.0
  no_object_weight: 2.0
  class_weight: 1.0

augment:
  enabled: false

output:
  runs_root: artifacts/runs/yolo/exp_v18_simple_yolo_tiny
  checkpoints_root: artifacts/checkpoints/yolo/exp_v18_simple_yolo_tiny
  analysis_root: artifacts/analysis/yolo/exp_v18_simple_yolo_tiny
```

Recommend: create a separate `configs/yolo/exp_v18_ssdlite/` directory with its own
`_base_ssdlite.yaml` that extends nothing and sets `model.backend: ssdlite` and SSD-
appropriate output paths. Then variant configs extend `_base_ssdlite.yaml`.

---

## Existing Tiny YOLO configs to mirror (all in `configs/yolo/exp_v18_simple_yolo_tiny/`)

| Existing YOLO config | SSDLite equivalent to create |
|---|---|
| `small_v4.yaml` (no aug) | `ssdlite_plain_no_aug.yaml` |
| `small_v4_genaug.yaml` (default aug) | `ssdlite_default_aug.yaml` |
| `small_v4_rareaug.yaml` (rare-slice baseline_b aug) | `ssdlite_rare_slices_aug.yaml` |
| `small_v4_fmaug_nofilter.yaml` | `ssdlite_fm_1to1_no_filter.yaml` |
| `small_v4_fmaug.yaml` (YOLO filter) | `ssdlite_fm_1to1_yolo_filter.yaml` |
| `small_v4_fmaug_fgbg.yaml` (classifier filter) | `ssdlite_fm_1to1_classifier_filter.yaml` |
| `small_v4_fmbalaug_nofilter.yaml` | `ssdlite_fm_rare_balancing_no_filter.yaml` |
| `small_v4_fmbalaug.yaml` | `ssdlite_fm_rare_balancing_yolo_filter.yaml` |
| *(does not exist yet)* | `ssdlite_fm_rare_balancing_classifier_filter.yaml` |

For the `fm_*` variants the only change vs Tiny YOLO is `model.backend: ssdlite` and
`model.ssdlite.*`. The `experiment_b:`, `augment:`, `filter:` sections are identical.

FM checkpoint for all variants:
```yaml
experiment_b:
  fm:
    checkpoint_path: artifacts/checkpoints/flow_matching/serious_runs/stay_layout_latent_v18_sd15ft_x8_256_minmax_reg_v2/UNET/unet_fm_epoch_570.pt
    preset_path: artifacts/checkpoints/flow_matching/serious_runs/stay_layout_latent_v18_sd15ft_x8_256_minmax_reg_v2/effective_config.yaml
    steps: 50
    batch_size: 8
```

YOLO filter weights: `artifacts/checkpoints/yolo/exp_v18_scratch_yolo11n/default_aug/best.pt`

---

## Checkpoint save/load

`_save_checkpoint` in `simple_yolo_detector.py` writes:
```python
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "config": cfg_dict,
    "split_info": dataclasses.asdict(split_info),
    "epoch": epoch,
    "metrics": metrics,
}, path)
```

Mirror the same format for SSDLite so `eval_simple_yolo` (or your `eval_ssdlite`) can
load it: read `checkpoint["config"]` to rebuild the model, load `model_state_dict`.

---

## Tiny YOLO experiment results (reference baseline)

All on v18 test set (799 images, 2026 GT boxes, single class `person`):

| Config | mAP50 (test) | mAP (test) | Best val ep | Stop ep |
|---|---|---|---|---|
| `small_v4_fmaug` (YOLO filter, baseline) | 0.9777 | — | 110 | 150 |
| `small_v4_fmbalaug` (balanced + YOLO filter) | 0.9721 | — | ~90 | ~110 |
| `small_v4_fmbalaug_nofilter` | 0.9706 | — | ~90 | ~110 |
| `small_v4_fmaug_nofilter` | **0.8433** | 0.6199 | 45 | 65 (early stop) |

Note: `fmaug_nofilter` test mAP50 (0.8433) is much lower than val mAP50 (0.9641) — the
model overfit to easy val images. Indicates the YOLO filter provides meaningful training signal.

---

## CLI launch pattern

```bash
# Train (action=train for plain/aug runs)
CUDA_VISIBLE_DEVICES=1 python -m src.cli.train_yolo \
  --action train \
  --device cuda:0 \
  --epochs 150 \
  --config configs/yolo/exp_v18_ssdlite/ssdlite_default_aug.yaml

# Train with FM generative aug (action=run_exp_b)
CUDA_VISIBLE_DEVICES=1 python -m src.cli.train_yolo \
  --action run_exp_b \
  --device cuda:0 \
  --epochs 150 \
  --config configs/yolo/exp_v18_ssdlite/ssdlite_fm_1to1_no_filter.yaml

# Eval only
CUDA_VISIBLE_DEVICES=1 python -m src.cli.train_yolo \
  --action eval \
  --device cuda:0 \
  --config configs/yolo/exp_v18_ssdlite/ssdlite_default_aug.yaml
```

---

## Where NOT to touch

- `src/algorithms/training/yolo_experiment_b.py` — FM generation pipeline; fully backend-agnostic
- `src/evaluation/detection_metrics.py` — metric code; do not modify
- `src/analysis/flir_subgroup/yolo_slice_eval.py` — per-slice eval; do not modify
- Any existing YOLO configs or model files
- Augmentation primitives in `src/algorithms/training/yolo_slice_baselines.py`

---

## Minimal change checklist

1. `src/models/ssdlite.py` — new file: `SSDLiteConfig`, `SSDLiteDetector`
2. `src/algorithms/training/ssdlite_detector.py` — new file: loss, decode, `train_ssdlite`, `eval_ssdlite`, `collect_ssdlite_predictions`
3. `src/core/configs/yolo_experiment_config.py` — add `SSDLiteModelConfig`; add `ssdlite` field to `YOLOModelConfig`
4. `src/cli/train_yolo.py` — four spots:
   - line ~791: add `"ssdlite"` to `_validate_model_backend` allowed set
   - line ~1068: add `elif backend == "ssdlite": train_ssdlite(...)`
   - line ~1211: add `elif backend == "ssdlite": eval_ssdlite(...)`
   - line ~1344: add `elif backend == "ssdlite": eval_ssdlite_slices(...)`
5. `configs/yolo/exp_v18_ssdlite/_base_ssdlite.yaml` + 9 variant configs
