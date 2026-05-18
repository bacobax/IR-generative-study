# RegionDiff Attention Distillation

This experiment distills only RegionDiff adapter attention maps from a frozen
SD1.5 + LoRA + RegionDiff teacher into the scratch latent FM + RegionDiff model.
It does not distill SD noise predictions, FM velocity targets, full images, or
intermediate latent features.

## Main Runs

Baseline scratch FM + RegionDiff remains:

```bash
python -m src.cli.train_flow_matching --config configs/fm/train/presets/regiondiff_latent_flir_sd15_512_from_uncond_ot_b64_hflip.yaml
```

Attention-KD ablations:

```bash
python -m src.cli.train_flow_matching --config configs/fm/train/presets/regiondiff_attention_kd_latent_flir_sd15_512_l001.yaml
python -m src.cli.train_flow_matching --config configs/fm/train/presets/regiondiff_attention_kd_latent_flir_sd15_512_l005.yaml
python -m src.cli.train_flow_matching --config configs/fm/train/presets/regiondiff_attention_kd_latent_flir_sd15_512_l010.yaml
python -m src.cli.train_flow_matching --config configs/fm/train/presets/regiondiff_attention_kd_latent_flir_sd15_512_selected_person_car_truck_l005.yaml
```

The default teacher path in the KD presets is:

```text
artifacts/checkpoints/stable_diffusion/layout_runs/flir_sd15_regiondiff_stage2_from_lora_r8_fm_comparable
```

This workspace also has an older local fallback teacher artifact:

```text
artifacts/checkpoints/stable_diffusion/layout_runs/flir_sd15_regiondiff_stage2
```

Use a CLI override if needed:

```bash
python -m src.cli.train_flow_matching \
  --config configs/fm/train/presets/regiondiff_attention_kd_latent_flir_sd15_512_l005.yaml \
  --distillation_teacher_checkpoint artifacts/checkpoints/stable_diffusion/layout_runs/flir_sd15_regiondiff_stage2
```

## Config Fields

`distillation.enabled` gates the whole feature. With `false`, old FM and FM +
RegionDiff behavior is unchanged.

Important fields:

- `teacher_checkpoint`: final stage-2 artifact dir, a `checkpoint-*` dir, or a RegionDiff weights file.
- `loss_type`: `attention_kl` or `attention_l2`.
- `lambda_attn`: KD weight added to the normal FM loss.
- `warmup_epochs`: epoch threshold before KD is active.
- `selected_categories`: names or ids; empty means all object categories.
- `selected_region_layers`: exact names, aliases, or substrings; empty means all captured RegionDiff layers.
- `timestep_range`: FM `t` interval where KD is active.
- `bbox_mask_only`: restrict each object attention-map loss to its bbox region.

## Logs

TensorBoard scalars include:

- `fm/base_loss_step`: normal flow-matching objective.
- `fm/attention_kd_loss_step`: raw attention KD loss.
- `fm/attention_kd_weighted_step`: `lambda_attn * attention_kd_loss`.
- `fm/total_loss_step`: training loss used for backward.
- `fm/attention_kd_matched_layers`: layers contributing to KD.
- `fm/attention_kd_selected_instances`: object instances used after category and timestep filtering.
- `fm/attention_kd_skipped_shape` and `fm/attention_kd_skipped_missing`: non-fatal layer matching/skipping diagnostics.

## Debug Visualization

```bash
python scripts/debug_regiondiff_attention_distillation.py \
  --config configs/fm/train/presets/regiondiff_attention_kd_latent_flir_sd15_512_l005.yaml \
  --teacher_checkpoint artifacts/checkpoints/stable_diffusion/layout_runs/flir_sd15_regiondiff_stage2 \
  --student_checkpoint artifacts/checkpoints/flow_matching/serious_runs/regiondiff_latent_flir_sd15_512_from_uncond_ot_hflip/UNET/unet_fm_epoch_80.pt \
  --output_dir artifacts/debug/regiondiff_attention_kd_probe \
  --max_batches 1 \
  --max_images 2
```

Heatmap panels are teacher, student, and absolute difference. Bright regions
show where an object token receives more RegionDiff attention. The decoded
sample overlays are for sanity-checking bbox alignment, not generation quality.

## First Ablations

Run the baseline, then KD with `lambda_attn` `0.01`, `0.05`, and `0.1`.
Compare all-category KD against selected `person`, `car`, `truck` KD.

## Assumptions

The active student path is `src.cli.train_flow_matching` with `FlowMatchingTrainer` and
`layout_conditioning.variant: regiondiff_v1`.

FM and teacher latents are treated as compatible because the latent presets use
the SD1.5 VAE scaling convention. FM time runs noise-to-data, while SD diffusion
timesteps run data-to-noise, so teacher timesteps use:

```text
sd_timestep = round((1 - fm_t) * (num_train_timesteps - 1))
```

Teacher prompts are reconstructed from the teacher manifest prompt mode and the
batch layout categories.
