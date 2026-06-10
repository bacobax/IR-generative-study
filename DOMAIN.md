# Domain Map for AI Coding Agents

This repository is an experiment codebase for single-channel infrared and
remote-sensing generative modeling, layout-conditioned generation, and YOLO
subgroup evaluation. This file summarizes only concepts that are represented in
the active code, configs, docs, and tests. Prefer the referenced source files
over this summary when behavior is ambiguous.

## Canonical Data Identities

Named datasets are registered in `src/core/data/dataset_targets.py`.

- `flir_private_proxy_alignment_v18`: root `data/raw/flir_private_proxy_alignment_v18`, normalization mode `uint8_linear`, COCO-style annotations per split. This is the default multi-class FLIR proxy dataset for analysis (`src/analysis/flir_subgroup/datasets.py`).
- `v18`: root `data/raw/v18`, normalization mode `raw_uint16_percentile`, COCO-style annotations per split. The subgroup app describes it as a single-class dataset with only the person category.
- `bigearthnet_s2_b08_5x5_stride3`: root `data/derived/bigearthnet_s2_b08_5x5_stride3`, normalization mode `sentinel2_reflectance`, manifest-backed TIFF mosaics and no COCO annotations.

Path helpers live in `src/core/paths.py`. Do not hard-code these roots in new
Python code when a helper exists.

## Image Normalization

The repo has multiple image domains that must not be merged:

- `raw_uint16_percentile` (`src/core/normalization.py`): raw uint16 IR values
  use global constants from `src/core/constants.py` and map to `[-1, 1]`.
  Inverse helpers include `fm_output_to_uint16`, `sd_output_to_uint16`, and
  `sd_output_to_npy`.
- `uint8_linear`: uint8-like FLIR proxy values map linearly from `[0, 255]` to
  `[-1, 1]`.
- `sentinel2_reflectance`: Sentinel-2 values map from `[0, 10000]` to
  `[-1, 1]`.
- `per_image_minmax` exists for per-image normalization but is a distinct
  family and should not replace the dataset target modes.

Stable Diffusion/SDXL adaptation paths convert single-channel data to RGB PIL
images before token/image preprocessing (`src/algorithms/stable_diffusion/data.py`).
Latent FM and unconditional latent SD presets using the SD 1.5 VAE explicitly
note that single-channel IR is replicated to RGB before VAE encoding, for
example `configs/fm/train/presets/uncond_latent_flir_sd15_512_b64.yaml` and
`configs/sd_uncond/train/presets/uncond_latent_flir_sd15_512_b64.yaml`.

## Canonical Sample and Layout Schemas

The canonical dict keys are documented in `src/core/data/schema.py`:

- Required image key: `pixel_values`.
- Optional text/token keys: `text`, `input_ids`, `attention_mask`.
- Optional layout keys: `boxes_xyxy`, `boxes_xyxy_norm`, `labels`,
  `label_names`, `object_mask`, `n_objects`.
- Optional provenance: `metadata`.

`src/core/data/adapters.py` normalizes legacy samples and copies aliases such as
`image_id`, `file_name`, `label_names`, `prompt_text`, `caption_text`, `split`,
`dataset_id`, `width`, and `height` into `metadata`.

Layout batching uses `src/core/data/layout_batching.py`:

- Inputs have variable object counts per image.
- The collate pads boxes to `(B, max_objects, 4)` and labels to
  `(B, max_objects)`.
- `object_mask` is the only reliable way to distinguish real objects from
  padding; label id `0` can be valid.
- `boxes_xyxy_norm` is derived from resized pixel-space `boxes_xyxy` and
  clamped to `[0, 1]`.

COCO-style bbox annotations are loaded through `src/core/data/annotations.py`
and consumed by `AnnotationFMDataset`, `AnnotationLayoutDataset`,
`StableDiffusionLayoutDataset`, ControlNet bbox masks, subgroup analysis, and
YOLO export.

## FLIR and v18 Data Flow

Active FLIR/v18 paths use split directories with `.npy` images and
`annotations.json`:

- `SingleChannelImageDataset` loads `.npy`, `.tif`, or `.tiff` files and can
  return metadata from JSONL manifests (`src/core/data/datasets.py`).
- `AnnotationFMDataset` loads `.npy` images plus COCO annotations for
  unconditional or text-conditioned FM (`src/core/data/annotation_dataset.py`).
  Text captions in this path are derived from annotations via person counts,
  not from `captions.json`.
- `AnnotationLayoutDataset` returns normalized images, resized `boxes_xyxy`,
  labels, label names, image ids, and file names. It can horizontally flip
  images and boxes together.
- `StableDiffusionLayoutDataset` applies the same square-pad-plus-resize
  geometry to both SD images and boxes, then emits `input_ids` plus layout
  fields (`src/algorithms/stable_diffusion/layout_data.py`).

Common pitfall: do not assume `root_dir/images` exists for every path. Some
loaders use the split directory directly; `StableDiffusionLayoutDataset` checks
for a nested `images` directory and falls back to the root.

## BigEarthNet S2 B08 5x5 Stride3

The active BigEarthNet target is `bigearthnet_s2_b08_5x5_stride3`
(`src/core/data/dataset_targets.py`). It resolves:

- split directories: `data/derived/bigearthnet_s2_b08_5x5_stride3/images/<split>`
- manifests: `data/derived/bigearthnet_s2_b08_5x5_stride3/manifests/<split>.jsonl`
- canonical `val` alias: `validation`
- annotations: no COCO categories; `category_metadata()` returns `{}` and
  `has_coco_annotations()` is false.

The protocol documentation in `docs/bigearthnet_s2_b08_5x5_protocol.md`
establishes the Sentinel-2 facts: B08 patches are 120x120 uint16 at 10 m
resolution; a 5x5 mosaic is 600x600 and covers 6 km x 6 km; all 25 patches in a
window must come from one source-tile directory and one official metadata split.
The active creation notebook records the stride-3 variant and output root in
`docs/notebooks/bigearthnet_s2_b08_5x5_creation.ipynb`.

Manifest records are expected to include an `image_path` or compatible alias.
`SingleChannelImageDataset` resolves relative manifest paths against the repo
root, split root, or file basename (`src/core/data/datasets.py`).

BigEarthNet presets include:

- FM: `configs/datasets/bigearthnet_s2_b08_5x5_stride3/flow_matching/...`
- SD 1.5 LoRA: `configs/datasets/bigearthnet_s2_b08_5x5_stride3/sd_adaptation/...`
- SDXL LoRA: `configs/sdxl/train/presets/bigearthnet_s2_b08_5x5_stride3_lora_stage1_r8.yaml`
- evaluation: `configs/eval/publication_single_runs/bigearthnet_s2_b08_5x5_stride3/...`

## Flow Matching

Core FM math is in `src/algorithms/tasks/flow_matching.py`.

- `z0` is sampled as Gaussian noise.
- `t` is sampled uniformly per batch item.
- `zt = (1 - t) * z0 + t * x_target`.
- Velocity target is `v_target = x_target - z0`.
- `training.train_target` can be `v` or `x0`; `x0` predictions are converted
  back to velocity for the MSE loss.
- `path.mode` can be `independent`, `minibatch_ot`, or `conditional_ot`; OT uses
  `src/core/ot.py`. When matching permutes targets, batch-aligned conditioning
  tensors/lists are permuted with the target.

`src/algorithms/training/flow_matching_trainer.py` is the main trainer for
pixel-space and latent-space FM. Latent-space FM uses a frozen VAE and writes
pipeline-style artifacts under `model_dir/UNET` and optionally `model_dir/VAE`.
Sampling is in `src/algorithms/inference/flow_matching_sampler.py`; Euler
sampling uses `t * t_scale` as the UNet timestep input.

Important FM config anchors:

- Unconditional latent FLIR: `configs/fm/train/presets/uncond_latent_flir_sd15_512_b64.yaml`
- Unconditional latent BigEarthNet with hflip+OT:
  `configs/datasets/bigearthnet_s2_b08_5x5_stride3/flow_matching/uncond_latent_bigearthnet_s2_b08_5x5_stride3_sd15_512_b64_hflip_ot.yaml`
- RegionDiff latent FLIR initialized from uncond FM:
  `configs/fm/train/presets/regiondiff_latent_flir_sd15_512_from_uncond_ot_b64_hflip.yaml`
- STAY layout latent FLIR:
  `configs/fm/train/presets/stay_layout_latent_flir_sd15_512.yaml`

Common pitfalls:

- `data.image_size` must be a positive multiple of 32 (`src/core/configs/fm_config.py`).
- If a VAE is configured, `image_size` must be divisible by the VAE downsample
  factor (`_resolve_unet_sample_size` in `flow_matching_trainer.py`).
- `training.t_scale` must be finite and positive.
- RegionDiff area loss requires `boxes_xyxy_norm` and `object_mask`.

## Latent and Unconditional Stable Diffusion

There are two diffusion families:

- Stage-1 text-conditioned SD/SDXL adaptation through diffusers pipelines.
- Unconditional latent diffusion using a repo UNet/DiT plus a DDPM scheduler.

Unconditional latent SD is implemented by
`src/algorithms/training/unconditional_sd_trainer.py` and configured by
`src/core/configs/sd_uncond_config.py`.

- It subclasses the FM trainer for runtime/checkpoint structure, but the loss
  is diffusion noise prediction against a `DDPMScheduler`.
- `diffusion.prediction_type` supports `epsilon` or `v_prediction`.
- Backbones can be `unet` or `dit`; DiT is only supported for unconditioned
  latent diffusion and rejects layout conditioning.
- RegionDiff can wrap the uncond UNet when `layout_conditioning.enabled` and
  `variant=regiondiff_v1`.

Main presets:

- `configs/sd_uncond/train/presets/uncond_latent_flir_sd15_512_b64.yaml`
- `configs/sd_uncond/train/presets/regiondiff_latent_flir_sd15_512_from_uncond_b64_hflip.yaml`
- BigEarthNet counterparts under
  `configs/datasets/bigearthnet_s2_b08_5x5_stride3/diffusion/`

## SD 1.5 and SDXL LoRA/UNet Adaptation

SD 1.5 stage-1 adaptation is configured in
`src/algorithms/stable_diffusion/config.py` and trained by
`src/algorithms/stable_diffusion/training.py`.

Supported SD 1.5 baseline modes:

- `sd_ir_lora`: LoRA-only adaptation. VAE and text encoder must be frozen.
- `sd_ir_unet`: UNet adaptation with `unet_train_mode` of `full` or `partial`.

The default constant prompt is `thermal image`; `generic_prompt` switches to
the legacy generic IR prompt. Dataset selection can be via `dataset_id`,
Hugging Face `dataset_name`, or local `train_data_dir`. `subset_manifest` is
only supported for local repo datasets, not Hugging Face datasets.

SD 1.5 examples:

- FLIR LoRA r8 sampling/training config:
  `configs/datasets/flir/sd_adaptation/flir_lora_stage1_r8.yaml`
- FLIR LoRA rank variants:
  `configs/sd/train/presets/flir_lora_stage1_r16.yaml`,
  `flir_lora_stage1_r32.yaml`, `flir_lora_stage1_r64.yaml`,
  `flir_lora_stage1_r128.yaml`
- DomainStudio full-UNet adaptation:
  `configs/sd/train/presets/flir_unet_full_domainstudio_512.yaml`

DomainStudio losses are in `src/algorithms/stable_diffusion/domainstudio.py`.
They use a frozen source teacher UNet, source-prior images, clean-latent
reconstruction, pairwise image/Haar high-frequency KL losses, and HF MSE.
Current config validation forbids DomainStudio with layout conditioning and
supports only epsilon prediction.

SDXL adaptation is configured in
`src/algorithms/stable_diffusion_xl/config.py` and trained by
`src/algorithms/stable_diffusion_xl/training.py`.

- Supported baseline mode is `sdxl_ir_lora`.
- VAE stays frozen.
- Text encoders stay frozen unless `text_encoder_lora_enabled=true`.
- Default model is `stabilityai/stable-diffusion-xl-base-1.0`.
- FLIR preset: `configs/sdxl/train/presets/flir_lora_stage1_r8.yaml`
- BigEarthNet preset:
  `configs/sdxl/train/presets/bigearthnet_s2_b08_5x5_stride3_lora_stage1_r8.yaml`

## RegionDiff and Layout Conditioning

RegionDiff-style modules are in `src/models/regiondiffusion.py` with shared
construction/trainability helpers in `src/models/regiondiffusion_factory.py`.

Represented concepts:

- Layout tokens combine class text/identity features, Fourier bbox features,
  and same-class occurrence position embeddings.
- Optional background token covers visual tokens not inside any object box.
- `build_region_token_mask()` maps normalized boxes to token accessibility
  masks at active latent resolutions such as `[64, 32, 16]`.
- `RegionSelfAttentionAdapter` performs masked attention from visual tokens to
  layout tokens.
- `RegionSpatialAdapter` can inject spatial layout residuals when enabled.
- Area loss uses bbox areas to weight object regions and normalize mean weight.

RegionDiff can wrap:

- SD 1.5 conditional UNet for layout stage-2 (`src/algorithms/stable_diffusion/layout_models.py`)
- FM UNet (`backbone_kind="fm_unet2d"`)
- uncond SD UNet (`backbone_kind="sd_uncond_unet2d"`)

Trainability modes:

- `adapters_only`: freeze base model, train RegionDiff adapters.
- `adapters_plus_partial_unet` / `adapters_plus_partial_backbone`: also unfreeze
  configured base prefixes such as `mid_block` and `up_blocks`.

SD layout stage-2 is configured by `src/core/configs/sd_layout_config.py` and
uses:

- data/prompt construction in `src/algorithms/stable_diffusion/layout_data.py`
- model initialization from a stage-1 LoRA or UNet run in
  `src/algorithms/stable_diffusion/layout_models.py`
- example preset:
  `configs/sd_layout/train/presets/flir_regiondiff_sd15_lora_stage2_r8.yaml`

Common pitfalls:

- SD 1.5 layout conditioning only supports `layout_conditioning_variant=regiondiff_v1`.
- If `dataset_id` is absent in SD layout, explicit layout annotation paths are
  required.
- Stage-2 initialization expects a valid stage-1 manifest or checkpoint under
  the configured `stage1_dir`.

## STAY Layout Flow Matching

The STAY-inspired layout FM path is implemented in
`src/models/stay_layout_conditioned_unet.py` and trained by
`src/algorithms/training/layout_flow_matching_trainer.py`.

The active STAY variant is `layout_conditioning.variant: stay_v2`. It encodes
per-object representations from:

- class embeddings
- bbox features `[x1, y1, x2, y2, center_x, center_y, width, height]`
- optional per-object style noise

It predicts local masks, places them into full spatial maps according to boxes,
derives semantic/objectness/edge/occupancy style maps, and injects conditioning
through modes such as `ea_norm`. Auxiliary losses include mask overlap,
sharpness, and activation terms.

The main latent FLIR preset is
`configs/fm/train/presets/stay_layout_latent_flir_sd15_512.yaml`. Pixel-space
STAY presets also exist under `configs/fm/train/presets/stay_layout_pixel_*`.

Common pitfall: STAY style noise must remain batch/object aligned. The sampler
and trainer generate fixed or random `style_noise` when `use_style_latent` is
enabled and none is provided.

## VAE

VAE training is launched by `train_vae.py`, which delegates to
`src/cli/train_vae.py`. Model helpers are in `src/models/vae.py`.

Supported VAE backends:

- in-repo MONAI/generative `AutoencoderKL` configs such as
  `configs/models/fm/vae_config_x4.json` and `vae_config_x8.json`
- diffusers `AutoencoderKL` adapter, marked in config metadata with
  `_backend: diffusers_autoencoder_kl`

VAE training datasets return both normalized inputs and raw resized targets.
The default reconstruction family is `l1_mse` plus KL warmup. Monitoring
includes PSNR/SSIM-like metrics and raw-domain reconstruction metrics when the
normalization mode supports an inverse.

Example presets:

- `configs/vae/train/presets/vae_4x.yaml`
- `configs/vae/train/presets/vae_8x.yaml`
- FLIR proxy alignment VAE presets under `configs/vae/train/presets/flir_private_proxy_alignment_v18_vae_*`

Common pitfall: diffusers SD VAE expects 3 channels. The adapter expands
1-channel input to 3 channels and averages 3-channel reconstructions back to
1-channel when needed (`_match_channel_count` in `src/models/vae.py`).

## ControlNet

ControlNet training is a stage-2 FM path built on a frozen stage-1 FM UNet and
frozen VAE:

- CLI/config: `src/cli/train_controlnet.py`, `src/core/configs/controlnet_config.py`
- trainer: `src/algorithms/training/controlnet_trainer.py`
- model: `src/models/controlnet.py`
- preset: `configs/controlnet/train/presets/bbox_controlnet.yaml`

The dataset is `BBoxConditioningDataset`, which loads `.npy` images and
COCO-format boxes, rasterizes `[x, y, w, h]` annotations into a binary bbox
mask, resizes image and mask to 256x256, and optionally drops the conditioning
mask during training.

ControlNet details represented in code:

- The ControlNet copies the frozen UNet encoder/down blocks and mid block.
- A conditioning encoder downsamples the bbox mask to latent resolution.
- Zero-initialized 1x1 convolutions make initial ControlNet residuals zero.
- Training objective is FM velocity MSE; only ControlNet parameters are trained.

Common pitfall: `stage1.stage1_pipeline_dir` is required and must contain
`UNET/config.json`, `UNET/unet_fm_best.pt` or epoch weights, and matching VAE
artifacts.

## YOLO Evaluation, Augmentation, and Subgroup Analysis

YOLO experiment config dataclasses are in
`src/core/configs/yolo_experiment_config.py`; the training CLI is
`src/cli/train_yolo.py`.

Experiment A:

- Uses exported YOLO datasets under `data/derived/yolo-test-ds/`.
- Baselines include balanced, unbalanced, full train, and slice-aware baseline
  modes.
- Example config family: `configs/yolo/exp_a/flir_yolov8m/`.
- Slice rarity and targeted geometry augmentation are implemented in
  `src/algorithms/training/yolo_slice_baselines.py`.
- Slice stats use `(class_label, size_bin, position_bin)` where size bins are
  `small`, `medium`, `large` and position bins are a 3x3 grid
  (`src/analysis/flir_subgroup/yolo_slice_stats.py`).

Experiment B:

- Full-train synthetic augmentation helpers live in
  `src/algorithms/training/yolo_experiment_b.py`.
- Modes are `plain`, `fm_aug`, `sd_aug`, and `precomputed_aug`.
- `fm_aug` requires an FM checkpoint path and preset; `sd_aug` requires exactly
  one of SD stage1 dir or LoRA dir; `precomputed_aug` requires a precomputed
  dataset dir.
- Generated candidates are written as `.npy` images plus COCO metadata before
  export/augmentation.
- Example config:
  `configs/yolo/exp_b/flir_yolov8m/exp_fm_aug.yaml`.

YOLO export:

- `src/analysis/flir_subgroup/yolo_export.py` converts subgroup partitions to
  standard Ultralytics directories: `images/<split>`, `labels/<split>`, plus a
  YAML with `path`, `train`, `val`, `test`, `names`, and `nc`.
- Exported labels use normalized YOLO columns:
  `class x_center y_center width height`.
- Source images are display-normalized and saved as RGB PNGs.

Subgroup analysis app:

- Backend modules: `src/analysis/flir_subgroup/`.
- Frontend: `frontend/flir-subgroup-analysis/`.
- API wrapper: `serve_flir_analysis.py` delegates to `src/cli/serve_flir_analysis.py`.
- Supported API dataset ids are `flir_private_proxy_alignment_v18` and `v18`
  (`src/analysis/flir_subgroup/schemas.py`).
- Analysis phases are `phase1` and `phase2`.
- `phase1` subgroup labels are `class=<label> | size=<bin>`.
- `phase2` subgroup labels add `pos=<bin>`.
- Default subgroup size bins are quantiles labeled `small`, `medium`, `large`.
- Default position mode in subgroup analysis is horizontal
  (`left`, `center`, `right`), while YOLO slice stats use a 3x3 grid.

Common pitfalls:

- The subgroup API request schemas require a `dataset` field for analysis POST
  payloads.
- `tau` must be in `[0, 1]`; example counts are limited by schema.
- YOLO rare-slice weighted sampling is single-process only; distributed rank is
  rejected in `build_weighted_train_dataloader()`.
- Experiment B validation rejects incompatible generation source combinations.

## Checkpoint Selection and Generative Metrics

Checkpoint-selection logic lives in `src/evaluation/checkpoint_selection/` and
is exposed in the subgroup web app as a separate view. Configs under
`configs/eval/` define run discovery, sampling configs, reference datasets, and
metric output roots. The frontend can browse roots such as
`artifacts/generated/checkpoint_selection` or scratch roots, but preview serving
is constrained under the selected root (`frontend/flir-subgroup-analysis/README.md`).

Generative metric helpers include:

- `src/evaluation/generative_metrics.py`
- `src/evaluation/feature_extractors.py`
- `src/evaluation/intra_lpips.py`
- `src/evaluation/mmd.py`

Do not launch expensive generation or checkpoint-selection runs as routine
validation unless explicitly asked.

## Repository Boundaries and Pitfalls for Agents

- Active code belongs in top-level `src/`, `configs/`, `scripts/`, `slurm/`,
  `tests/`, and `frontend/`. Treat `archive/`, `ControlNet/`,
  `src/diffusers/`, and nested checkout-like trees as reference/legacy unless
  explicitly asked.
- Root scripts are wrappers. Add behavior under `src/cli/` or relevant
  `src/algorithms/` modules, not in root wrappers.
- Prefer config presets over hard-coded experiment behavior. Existing launchers
  generally point at YAML via a clear config path.
- Generated outputs, checkpoints, TensorBoard events, generated datasets, and
  local `.env` files should stay out of tracked source.
- `subset_manifest` is a local repo dataset feature. Config validation rejects
  it for Hugging Face `dataset_name` SD/SDXL inputs.
- Layout labels, boxes, masks, and target images must remain aligned across
  flips, padding/resizing, OT matching, and batching.
- `src/diffusers/` is a nested diffusers checkout; do not modify it for normal
  repo behavior.

