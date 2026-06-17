# FLUX Adaptation Routing

## Purpose
Load this for FLUX.1-dev QLoRA fine-tuning as a high-level task: FLUX LoRA training,
stage-1 manifests, dataset-specific FLUX presets, subsampling, and FLUX Slurm launchers.

Do not load SD15, SDXL, or STAY docs unless the user explicitly mentions them.

## Primary files/directories to inspect
- `src/cli/train_flux.py`
- `src/algorithms/flux/`
  - `config.py` — TrainingConfig dataclass, argparse, parse_args
  - `data.py` — FluxImageDataset, create_dataloader (wraps shared SD data helpers)
  - `models.py` — load_models, precompute_prompt_embeds, configure_trainable_components,
    save/load hooks, build_stage1_manifest, save_stage1_manifest, load_flux_stage1_pipeline
  - `training.py` — Trainer class with FLUX flow-matching training loop
- `configs/flux/train/default.yaml`
- `configs/flux/train/presets/`
- `configs/datasets/flir/flux_adaptation/`
- `configs/datasets/bigearthnet_s2_b08_5x5_stride3/flux_adaptation/`
- `slurm/killarney/flir/flux_adaptation/train_flir_flux_lora_stage1_r8_full_kl.slurm`
- `tests/test_flux_stage1.py`

## Key design notes
- **QLoRA**: transformer loaded in 4-bit NF4 (`quantize_4bit: true`, `bnb_4bit_quant_type: nf4`),
  LoRA adapters on attention projections, 8-bit AdamW optimizer.
- **Fixed prompt**: one prompt string per dataset encoded once via `precompute_prompt_embeds`;
  text encoders (CLIP + T5-XXL) are deleted before training begins.
- **Latent caching**: when `cache_latents: true` (default), all images are VAE-encoded once
  and the VAE is freed; `false` encodes on-the-fly (for very large splits).
- **Dataset + subsampling**: `dataset_id` + `subset_manifest` keys work identically to the
  SDXL path — any existing `train_<N>.json` manifest works automatically.
- **Flow-matching loss**: sigma interpolation + velocity target (`noise - latent`), not
  DDPM epsilon/v-prediction. Ported from `flux_lora_quant_blogpost.py`.
- **Output**: `pytorch_lora_weights.safetensors` + `stage1_manifest.json` +
  `artifact_manifest.json` under `artifacts/checkpoints/flux/lora_runs/`.

## Model access prerequisite
FLUX.1-dev is a gated model.  Before any real training run:
1. Accept the license at https://huggingface.co/black-forest-labs/FLUX.1-dev
2. Set `HF_TOKEN` or run `huggingface-cli login`.
Alternatively, set `pretrained_model_name_or_path` to a local snapshot path.

## Modification guidance
FLUX changes belong in `src/algorithms/flux/`, `src/cli/train_flux.py`, and
`configs/flux/` or dataset-specific flux configs.  Root `train_flux.py` stays thin.

Dataset and subsampling logic (shared with SDXL/SD15) lives in:
- `src/algorithms/stable_diffusion/data.py` (resolve_training_data_source, load_training_dataset,
  get_transforms, ir_npy_to_normalized_rgb)
- `src/core/data/dataset_targets.py` (supported_dataset_ids, resolve_dataset_target)

## Validation guidance
Run `python -m pytest tests/test_flux_stage1.py -v` (CPU, no model download).
Config loading: `conda run -n diffusers-dev python scripts/checks/check_config_loading.py`.
Avoid FLUX training unless explicitly requested (requires FLUX.1-dev access + a GPU).
