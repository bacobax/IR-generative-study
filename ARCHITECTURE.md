# Architecture Guide for AI Coding Agents

This repository is a config-first research codebase for infrared/thermal image generation, adaptation, evaluation, and subgroup analysis. This document describes the active architecture as it exists in code and configs. Do not treat it as a feature roadmap.

## Active Ownership Boundaries

Work from the repository root. Active Python code lives under `src/`; root scripts are compatibility entry points and should stay thin.

- `src/cli/`: canonical Python CLI entry points. Root files such as `train_flow_matching.py`, `adapt_stable_diffusion.py`, `train_latent_diffusion.py`, `train_vae.py`, `train_controlnet.py`, `train_sdxl.py`, `generate_datasets.py`, and `serve_flir_analysis.py` import and call these modules.
- `src/core/`: shared path, config, registry, data, normalization, artifact, GPU, and training-runtime utilities.
- `src/algorithms/`: training, inference, task, and Stable Diffusion/SDXL orchestration logic.
- `src/models/`: native model definitions/builders and model-adapter contracts.
- `src/conditioning/` and `src/guidance/`: optional conditioning/guidance components registered for FM-style flows.
- `src/evaluation/`: metric helpers and checkpoint-selection wrappers.
- `src/analysis/`: analysis backends, including the FLIR subgroup FastAPI API and checkpoint-selection viewer.
- `configs/`: YAML/JSON experiment presets and model architecture configs.
- `scripts/`: reusable scripts and standalone evaluation/generation/training utilities.
- `slurm/<cluster>/`: cluster launchers.
- `frontend/flir-subgroup-analysis/`: React UI for the subgroup/checkpoint-selection analysis service.

Treat `archive/`, `ControlNet/`, `src/diffusers/`, and nested checkout-like trees such as `src/flow-matching-trial/` as legacy/external reference material unless the user explicitly asks to touch them.

## Config Loading

Most runnable flows are config-first. The active config loader is `src/core/configs/config_loader.py`.

- `load_yaml()` supports one relative `extends:` parent chain and deep-merges child values over parent values.
- `merge_config_and_cli()` builds nested dataclass configs with precedence `dataclass default < YAML < explicitly provided CLI flags`.
- `apply_yaml_defaults()` is used by argparse-style CLIs such as `src/cli/generate.py` and `src/cli/train_vae.py` to apply YAML defaults before re-parsing CLI flags.
- `dict_to_dataclass_strict()` and `load_experimental_config()` validate the experimental future config shape in `src/core/configs/future_spec.py`.

FM training config dataclasses are in `src/core/configs/fm_config.py`; text-FM config is in `src/core/configs/text_fm_config.py`; unconditional latent diffusion config is in `src/core/configs/sd_uncond_config.py`; SD layout config is in `src/core/configs/sd_layout_config.py`; ControlNet config is in `src/core/configs/controlnet_config.py`; YOLO config is in `src/core/configs/yolo_experiment_config.py`.

When adding a config key, wire it through the dataclass/defaults, CLI flat-to-nested mapping if present, loader/tests/check scripts, and the consuming trainer/sampler. Do not rely on argparse defaults overriding YAML; `merge_config_and_cli()` intentionally keeps only explicit CLI overrides.

## Registries and Adapters

The lightweight registry container is `src/core/registry.py`. It has named registries for `model_builder`, `trainer`, `sampler`, `guidance`, `conditioning`, `model_adapter`, `dataset_adapter`, `task_adapter`, and `artifact_loader`.

Important behavior: there is no dynamic import discovery. A component registers only when its module is imported. CLIs import default component modules near the top for this reason.

Known registrations include:

- Model builders: `default_unet` in `src/models/fm_unet.py`, `text_fm_unet` in `src/models/fm_text_unet.py`, and `dit_transformer_2d` in `src/models/dit.py`.
- Trainers: `default_fm` in `src/algorithms/training/flow_matching_trainer.py`, `layout_fm` in `src/algorithms/training/layout_flow_matching_trainer.py`, `text_fm_cfg` in `src/algorithms/training/text_fm_trainer.py`, `sd_uncond` in `src/algorithms/training/unconditional_sd_trainer.py`, and `controlnet` in `src/algorithms/training/controlnet_trainer.py`.
- Samplers: `default_fm` in `src/algorithms/inference/flow_matching_sampler.py`, `cfg_fm` in `src/algorithms/inference/cfg_flow_matching_sampler.py`, and `sd_uncond` in `src/algorithms/inference/unconditional_sd_sampler.py`.
- Conditioning/guidance defaults: `NoConditioner` in `src/conditioning/no_conditioner.py`, `TextConditioner` in `src/conditioning/text_conditioner.py`, and `NoGuidance` in `src/guidance/no_guidance.py`.

Dataset adapter contracts live in `src/core/data/adapters.py`; model adapter contracts live in `src/models/adapters/base.py`; task adapter contracts live in `src/core/tasks/adapters.py`. The adapter APIs are additive and coexist with the legacy-style registry and trainer constructors.

## Data and Path Architecture

Canonical repository paths are centralized in `src/core/paths.py`. Use these helpers rather than hard-coded repo-relative strings when adding shared code.

Named dataset targets are in `src/core/data/dataset_targets.py`:

- `v18` resolves to `data/raw/v18/` and uses raw uint16 percentile normalization.
- `flir_private_proxy_alignment_v18` resolves to `data/raw/flir_private_proxy_alignment_v18/` and uses uint8 linear normalization.
- `bigearthnet_s2_b08_5x5_stride3` resolves to `data/derived/bigearthnet_s2_b08_5x5_stride3/`, uses manifest-backed Sentinel-2 B08 TIFF mosaics, and has no COCO annotations.

General training data resolution is in `src/core/data/training_data.py`. It resolves `dataset_id`, split directories, manifests, annotations, and normalization mode, then builds dataloaders for non-layout paths. The active datasets include:

- `SingleChannelImageDataset` and `NPYImageDataset` in `src/core/data/datasets.py` for one-channel `.npy`/TIFF data.
- `AnnotationFMDataset` in `src/core/data/annotation_dataset.py` for annotation-aware FM, curriculum crops, count filtering, and annotation-derived captions.
- `AnnotationLayoutDataset` in `src/core/data/datasets.py` for layout batches with boxes/labels and optional horizontal flip schedules.
- `BBoxConditioningDataset` in `src/core/data/datasets.py` for ControlNet bbox masks.

Batch canonicalization helpers are in `src/core/data/adapters.py`; layout collate behavior is in `src/core/data/layout_batching.py`.

## Training Data Flow

### Flow Matching

Root `train_flow_matching.py` forwards to `src/cli/train_flow_matching.py`.

The FM CLI:

1. Detects text-FM configs via text-specific CLI flags or YAML shape.
2. Builds either `FMTrainConfig` or `TextFMTrainConfig` with `merge_config_and_cli()`.
3. Resolves data via `resolve_training_data()`.
4. Chooses one of three paths:
   - non-layout FM: `build_non_layout_dataloaders()` builds `SingleChannelImageDataset` or `AnnotationFMDataset` depending on curriculum/annotations.
   - layout-conditioned FM: directly builds `AnnotationLayoutDataset`, uses `collate_layout_batch`, fills category metadata into `cfg.layout_conditioning`, and chooses `layout_fm` or `default_fm` depending on layout variant.
   - text-conditioned FM: requires COCO `annotations_path`, builds `AnnotationFMDataset(text_mode=True)`, and uses a simple text collate.
5. Resolves the trainer from `REGISTRIES.trainer` unless `architecture_mode="adapter_v1"` for non-layout FM, which uses `src/models/adapters/fm.py`.
6. Calls `trainer.train_from_config(cfg, train_loader, eval_loader)`.

Core FM target construction is in `src/algorithms/tasks/flow_matching.py`. It samples `z0`, `t`, `zt`, velocity targets, optional OT matching, and MSE loss for `v` or `x0` prediction. The main trainer is `FlowMatchingTrainer` in `src/algorithms/training/flow_matching_trainer.py`; it handles pixel-space and VAE latent-space training, EMA/scheduler/precision utilities, checkpoint manifests, TensorBoard logging, optional RegionDiff wrapping, area-weighted losses, and attention distillation. Sampling is in `src/algorithms/inference/flow_matching_sampler.py`; CFG text sampling is in `src/algorithms/inference/cfg_flow_matching_sampler.py`.

### From-Scratch Latent Diffusion

Root `train_latent_diffusion.py` forwards to `src/cli/train_latent_diffusion.py`. It parses `sd_uncond` config, reuses `resolve_training_data()` and the same non-layout/layout dataset split as FM, imports the UNet/DiT builders and `sd_uncond` trainer/sampler registrations, then resolves `REGISTRIES.trainer.get(cfg.trainer_name)`.

The registered trainer is `src/algorithms/training/unconditional_sd_trainer.py`; sampling is `src/algorithms/inference/unconditional_sd_sampler.py`.

### Stable Diffusion 1.5 Adaptation

Root `adapt_stable_diffusion.py` forwards to `src/cli/adapt_stable_diffusion.py`, which delegates by `--stage` or config path/shape:

- Stage 1 (`configs/sd/**`) uses `src/cli/adapt_stable_diffusion_stage1.py`. It parses `src/algorithms/stable_diffusion/config.py`, sets up Accelerate, loads models via `src/algorithms/stable_diffusion/models.py`, builds dataloaders via `src/algorithms/stable_diffusion/data.py` or layout data via `src/algorithms/stable_diffusion/layout_data.py`, configures trainable components, and trains with `src/algorithms/stable_diffusion/training.py`.
- RegionDiff stage 2 (`configs/sd_layout/**`) uses `src/cli/adapt_stable_diffusion_regiondiff_stage2.py`. It parses `src/core/configs/sd_layout_config.py`, builds layout dataloaders, loads layout model components from `src/algorithms/stable_diffusion/layout_models.py`, configures layout trainability, and trains with `src/algorithms/stable_diffusion/layout_training.py`.

### Stable Diffusion XL Adaptation

Root `train_sdxl.py` forwards to `src/cli/train_sdxl.py`. It parses `src/algorithms/stable_diffusion_xl/config.py`, sets up Accelerate, loads SDXL model components in `src/algorithms/stable_diffusion_xl/models.py`, creates dataloaders in `src/algorithms/stable_diffusion_xl/data.py`, configures trainable components, and trains with `src/algorithms/stable_diffusion_xl/training.py`.

### VAE

Root `train_vae.py` forwards to `src/cli/train_vae.py`. This CLI uses `apply_yaml_defaults()`, builds `_VAEReconstructionDataset` over `NPYImageDataset`, loads native or diffusers VAE configs via `src/models/vae.py`, and runs its training loop in the CLI module. It saves VAE configs/weights through `src/models/vae.py`.

### ControlNet

Root `train_controlnet.py` forwards to `src/cli/train_controlnet.py`. It uses `CNTrainConfig`, builds `BBoxConditioningDataset` train/val loaders, loads and freezes a stage-1 FM UNet/VAE from `stage1_pipeline_dir`, wraps the frozen UNet with `src/models/controlnet.py`, saves `CONTROLNET/config.json`, and trains with `src/algorithms/training/controlnet_trainer.py`.

## Generation and Sampling Flow

FM-only standalone sampling is `src/cli/sample.py`. It builds `FMSampleConfig`, resolves the sampler registry, loads `UNET/` and `VAE/` from a pipeline directory in `FlowMatchingSampler.from_config()`, samples with Euler steps, decodes through the VAE when present, and writes `.npy` plus preview `.png` outputs.

Multi-backend synthetic generation is `src/cli/generate.py`, reached from root `generate_datasets.py`. Supported modes are `sd15`, `sdxl`, and `fm`.

- `sd15` loads either a stage-1 artifact through `src/algorithms/stable_diffusion/models.py` or legacy LoRA weights into a diffusers `StableDiffusionPipeline`.
- `sdxl` loads either a stage-1 SDXL artifact through `src/algorithms/stable_diffusion_xl/models.py` or legacy LoRA weights into `StableDiffusionXLPipeline`.
- `fm` builds a registered FM sampler from `fm_pipeline_dir`.

All generation modes save generated samples under the configured `output_dir` as `.npy` files with `.png` previews and write `metadata.jsonl`.

## Evaluation and Analysis Flow

Low-level metrics live in `src/evaluation/`: FID/KID in `generative_metrics.py`, RBF MMD in `mmd.py`, feature extraction in `feature_extractors.py`, and LPIPS in `intra_lpips.py`.

Checkpoint selection is currently split between the standalone script `scripts/select_best_checkpoint_and_compute_metrics.py` and thin wrappers in `src/evaluation/checkpoint_selection/`. The script discovers native/diffusers checkpoints, generates samples, validates generated arrays, computes metrics, writes summaries, and optionally cleans generated images or non-selected checkpoints. `src/evaluation/checkpoint_selection/pipelines.py` delegates to the script for `legacy_staged_kid_fid` and `clean_fid_selection_publication` modes. The viewer API for these outputs is `src/analysis/checkpoint_selection_viewer.py`.

## Web Service and Frontend

Root `serve_flir_analysis.py` forwards to `src/cli/serve_flir_analysis.py`, which runs uvicorn over `src.analysis.flir_subgroup.app:create_app()`.

`src/analysis/flir_subgroup/app.py` builds the FastAPI app, configures CORS, mounts the FLIR subgroup router, mounts the checkpoint-selection viewer router, and exposes `/health`.

The FLIR subgroup router in `src/analysis/flir_subgroup/api.py` serves:

- `/api/flir-analysis/datasets`
- `/api/flir-analysis/options`
- `/api/flir-analysis/holdout-curves`
- `/api/flir-analysis/collateral`
- `/api/flir-analysis/partition-comparisons`
- `/api/flir-analysis/examples`
- `/api/flir-analysis/images/{dataset}/{image_key:path}`

Analysis contexts are cached in `src/analysis/flir_subgroup/context.py`; dataset registry defaults are in `src/analysis/flir_subgroup/datasets.py`; table loading/preview rendering is in `src/analysis/flir_subgroup/data.py`; analysis calculations are in `src/analysis/flir_subgroup/analysis.py`; request/response schemas are in `src/analysis/flir_subgroup/schemas.py`.

The React frontend lives in `frontend/flir-subgroup-analysis/` and talks to these APIs through `frontend/flir-subgroup-analysis/src/api.ts`.

## Common Architectural Pitfalls

- Do not add logic to root wrappers. Put behavior in the corresponding `src/cli/*` module or deeper source module.
- Do not edit or import from `archive/`, `ControlNet/`, `src/diffusers/`, or nested checkout-like trees for active code unless explicitly requested.
- Do not assume registry entries exist unless the registering module is imported. If a CLI resolves `REGISTRIES.trainer.get(...)`, make sure the module with the `REGISTRIES.trainer.register(...)` call has been imported.
- Do not hard-code dataset roots or artifact paths in shared code. Prefer `src/core/paths.py`, YAML presets, and `dataset_id` resolution.
- Do not add new generated outputs at the repository root. Defaults should go under `artifacts/`, `data/derived/`, `data/cache/`, or `logs/`.
- Do not mix layout and non-layout batch shapes casually. Layout flows expect `AnnotationLayoutDataset` plus `collate_layout_batch` or SD layout collate; non-layout FM/SD paths generally consume tensors or `pixel_values`.
- Do not use text-conditioned FM without COCO annotations. `src/cli/train_flow_matching.py` requires `data.annotations_path` for text-FM.
- Do not enable `architecture_mode="adapter_v1"` for layout FM. The CLI explicitly rejects adapter v1 with `layout_conditioning.enabled`.
- Do not forget normalization mode. Dataset targets carry `raw_uint16_percentile`, `uint8_linear`, or `sentinel2_reflectance`, and generation/evaluation conversions depend on it.
- Do not bypass config tests/checks when changing loader, path, registry, launcher, or legacy-boundary behavior. Relevant checks live under `scripts/checks/`, and targeted tests live under `tests/`.
- Slurm launchers should keep cluster/runtime details in `.slurm` files and experiment behavior in YAML configs.
