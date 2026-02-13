# Flow Matching Trial Scripts

This repo includes shell scripts in the root folder. They resolve paths relative to their own location, so you can run them from any working directory.

## Basic usage (from any folder)

```bash
bash /path/to/flow_matching_trial/run_stable_training.sh
```

You can also make scripts executable once and then run them directly:

```bash
chmod +x /path/to/flow_matching_trial/*.sh
/path/to/flow_matching_trial/run_stable_training.sh
```

## Examples

Generate datasets:

```bash
bash /path/to/flow_matching_trial/generate_dataset_fm.sh
bash /path/to/flow_matching_trial/generate_dataset_sdr4.sh
bash /path/to/flow_matching_trial/generate_dataset_sdr8.sh
bash /path/to/flow_matching_trial/generate_dataset_sdr16.sh
```

Train models:

```bash
bash /path/to/flow_matching_trial/train_vae_4x.sh
bash /path/to/flow_matching_trial/train_vae_8x.sh
bash /path/to/flow_matching_trial/run_stable_training.sh
bash /path/to/flow_matching_trial/run_controlnet_training.sh
```

Train LoRA variants:

```bash
bash /path/to/flow_matching_trial/train_lora_sd_1-5r4.sh
bash /path/to/flow_matching_trial/train_lora_sd_1-5r4_generic.sh
bash /path/to/flow_matching_trial/train_lora_sd_1-5r8.sh
bash /path/to/flow_matching_trial/train_lora_sd_1-5r8_generic.sh
bash /path/to/flow_matching_trial/train_lora_sd_1-5r16.sh
bash /path/to/flow_matching_trial/train_lora_sd_1-5r16_generic.sh
```

Analyze outputs:

```bash
bash /path/to/flow_matching_trial/analysis.sh
```
