#!/usr/bin/env python
"""Static checks for Slurm launcher helper usage."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

EXPECTED_HEADERS = {'slurm/killarney/generate_qcmp_stay_layout_fm_hflip_kl.slurm': '#!/bin/bash\n'
                                                                '#SBATCH --job-name=qcmp-stay-fm-hf\n'
                                                                '#SBATCH --account=aip-mpederso\n'
                                                                '#SBATCH --time=00:15:00\n'
                                                                '#SBATCH --cpus-per-task=10\n'
                                                                '#SBATCH --mem=15G\n'
                                                                '#SBATCH --gpus-per-node=h100:1\n'
                                                                '#SBATCH '
                                                                '--output=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.out\n'
                                                                '#SBATCH '
                                                                '--error=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.err\n',
 'slurm/killarney/generate_qcmp_uncond_fm_hflip_ot_kl.slurm': '#!/bin/bash\n'
                                                              '#SBATCH --job-name=qcmp-fm-hf-ot\n'
                                                              '#SBATCH --account=aip-mpederso\n'
                                                              '#SBATCH --time=00:15:00\n'
                                                              '#SBATCH --cpus-per-task=10\n'
                                                              '#SBATCH --mem=15G\n'
                                                              '#SBATCH --gpus-per-node=h100:1\n'
                                                              '#SBATCH '
                                                              '--output=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.out\n'
                                                              '#SBATCH '
                                                              '--error=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.err\n',
 'slurm/killarney/bigearthnet_s2_b08_5x5_stride3/flow_matching/train_bigearthnet_s2_b08_5x5_stride3_fm_kl.slurm': '#!/bin/bash\n'
                                                                     '#SBATCH --job-name=ben-fm-5x5s3\n'
                                                                     '#SBATCH --account=aip-mpederso\n'
                                                                     '#SBATCH --time=24:00:00\n'
                                                                     '#SBATCH --cpus-per-task=10\n'
                                                                     '#SBATCH --mem=32G\n'
                                                                     '#SBATCH --gpus-per-node=h100:1\n'
                                                                     '#SBATCH --array=0-2\n'
                                                                     '#SBATCH '
                                                                     '--output=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%A_%a.out\n'
                                                                     '#SBATCH '
                                                                     '--error=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%A_%a.err\n',
 'slurm/killarney/bigearthnet_s2_b08_5x5_stride3/sd_adaptation/train_bigearthnet_s2_b08_5x5_stride3_lora_kl.slurm': '#!/bin/bash\n'
                                                                       '#SBATCH --job-name=ben-lora-5x5s3\n'
                                                                       '#SBATCH --account=aip-mpederso\n'
                                                                       '#SBATCH --time=24:00:00\n'
                                                                       '#SBATCH --cpus-per-task=10\n'
                                                                       '#SBATCH --mem=32G\n'
                                                                       '#SBATCH --gpus-per-node=h100:1\n'
                                                                       '#SBATCH --array=0-2\n'
                                                                       '#SBATCH '
                                                                       '--output=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%A_%a.out\n'
                                                                       '#SBATCH '
                                                                       '--error=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%A_%a.err\n',
 'slurm/killarney/bigearthnet_s2_b08_5x5_stride3/diffusion/train_bigearthnet_s2_b08_5x5_stride3_sd_uncond_kl.slurm': '#!/bin/bash\n'
                                                                            '#SBATCH --job-name=ben-sd-5x5s3\n'
                                                                            '#SBATCH --account=aip-mpederso\n'
                                                                            '#SBATCH --time=24:00:00\n'
                                                                            '#SBATCH --cpus-per-task=10\n'
                                                                            '#SBATCH --mem=32G\n'
                                                                            '#SBATCH --gpus-per-node=h100:1\n'
                                                                            '#SBATCH --array=0-2\n'
                                                                            '#SBATCH '
                                                                            '--output=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%A_%a.out\n'
                                                                            '#SBATCH '
                                                                            '--error=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%A_%a.err\n',
 'slurm/killarney/generate_qcmp_uncond_sd_hflip_kl.slurm': '#!/bin/bash\n'
                                                           '#SBATCH --job-name=qcmp-sd-hf\n'
                                                           '#SBATCH --account=aip-mpederso\n'
                                                           '#SBATCH --time=00:15:00\n'
                                                           '#SBATCH --cpus-per-task=10\n'
                                                           '#SBATCH --mem=15G\n'
                                                           '#SBATCH --gpus-per-node=h100:1\n'
                                                           '#SBATCH '
                                                           '--output=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.out\n'
                                                           '#SBATCH '
                                                           '--error=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.err\n',
 'slurm/killarney/select_first_stage_flir_ds_checkpoints_kl.slurm': '#!/bin/bash\n'
                                                                    '#SBATCH --job-name=sel-flir-stage1\n'
                                                                    '#SBATCH --account=aip-mpederso\n'
                                                                    '#SBATCH --time=00:05:00\n'
                                                                    '#SBATCH --cpus-per-task=10\n'
                                                                    '#SBATCH --mem=48G\n'
                                                                    '#SBATCH --gpus-per-node=h100:1\n'
                                                                    '#SBATCH '
                                                                    '--output=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.out\n'
                                                                    '#SBATCH '
                                                                    '--error=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.err\n',
 'slurm/killarney/train_flir_lora_r8_then_regiondiff_kl.slurm': '#!/bin/bash\n'
                                                                '#SBATCH --job-name=sd-lora-rdiff\n'
                                                                '#SBATCH --account=aip-mpederso\n'
                                                                '#SBATCH --time=12:00:00\n'
                                                                '#SBATCH --cpus-per-task=20\n'
                                                                '#SBATCH --mem=32G\n'
                                                                '#SBATCH --gpus-per-node=h100:1\n'
                                                                '#SBATCH '
                                                                '--output=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.out\n'
                                                                '#SBATCH '
                                                                '--error=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.err\n',
 'slurm/killarney/flir/sd_adaptation/train_flir_unet_full_domainstudio_512_kl.slurm': '#!/bin/bash\n'
                                                                   '#SBATCH --job-name=sd-dstudio-unet\n'
                                                                   '#SBATCH --account=aip-mpederso\n'
                                                                   '#SBATCH --time=24:00:00\n'
                                                                   '#SBATCH --cpus-per-task=10\n'
                                                                   '#SBATCH --mem=48G\n'
                                                                   '#SBATCH --gpus-per-node=h100:1\n'
                                                                   '#SBATCH '
                                                                   '--output=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.out\n'
                                                                   '#SBATCH '
                                                                   '--error=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.err\n',
 'slurm/killarney/train_flir_unet_full_then_regiondiff_kl.slurm': '#!/bin/bash\n'
                                                                  '#SBATCH --job-name=sd-unet-rdiff\n'
                                                                  '#SBATCH --account=aip-mpederso\n'
                                                                  '#SBATCH --time=12:00:00\n'
                                                                  '#SBATCH --cpus-per-task=10\n'
                                                                  '#SBATCH --mem=32G\n'
                                                                  '#SBATCH --gpus-per-node=h100:1\n'
                                                                  '#SBATCH '
                                                                  '--output=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.out\n'
                                                                  '#SBATCH '
                                                                  '--error=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.err\n',
 'slurm/killarney/train_regiondiff_attention_kd_selected_person_car_truck_l005_kl.slurm': '#!/bin/bash\n'
                                                                                          '#SBATCH '
                                                                                          '--job-name=rdiff-kd-sel-l005\n'
                                                                                          '#SBATCH '
                                                                                          '--account=aip-mpederso\n'
                                                                                          '#SBATCH --time=00:06:00\n'
                                                                                          '#SBATCH --cpus-per-task=10\n'
                                                                                          '#SBATCH --mem=40G\n'
                                                                                          '#SBATCH '
                                                                                          '--gpus-per-node=h100:1\n'
                                                                                          '#SBATCH '
                                                                                          '--output=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.out\n'
                                                                                          '#SBATCH '
                                                                                          '--error=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.err\n',
 'slurm/killarney/train_regiondiff_fm_from_uncond_hflip_kl.slurm': '#!/bin/bash\n'
                                                                   '#SBATCH --job-name=rdiff-fm-hf\n'
                                                                   '#SBATCH --account=aip-mpederso\n'
                                                                   '#SBATCH --time=24:00:00\n'
                                                                   '#SBATCH --cpus-per-task=10\n'
                                                                   '#SBATCH --mem=32G\n'
                                                                   '#SBATCH --gpus-per-node=h100:1\n'
                                                                   '#SBATCH '
                                                                   '--output=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.out\n'
                                                                   '#SBATCH '
                                                                   '--error=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.err\n',
 'slurm/killarney/train_regiondiff_fm_from_uncond_hflip_ot_kl.slurm': '#!/bin/bash\n'
                                                                      '#SBATCH --job-name=rdiff-fm-ot-hf\n'
                                                                      '#SBATCH --account=aip-mpederso\n'
                                                                      '#SBATCH --time=24:00:00\n'
                                                                      '#SBATCH --cpus-per-task=10\n'
                                                                      '#SBATCH --mem=32G\n'
                                                                      '#SBATCH --gpus-per-node=h100:1\n'
                                                                      '#SBATCH '
                                                                      '--output=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.out\n'
                                                                      '#SBATCH '
                                                                      '--error=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.err\n',
 'slurm/killarney/train_regiondiff_fm_from_uncond_kl.slurm': '#!/bin/bash\n'
                                                             '#SBATCH --job-name=rdiff-fm\n'
                                                             '#SBATCH --account=aip-mpederso\n'
                                                             '#SBATCH --time=24:00:00\n'
                                                             '#SBATCH --cpus-per-task=10\n'
                                                             '#SBATCH --mem=32G\n'
                                                             '#SBATCH --gpus-per-node=h100:1\n'
                                                             '#SBATCH '
                                                             '--output=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.out\n'
                                                             '#SBATCH '
                                                             '--error=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.err\n',
 'slurm/killarney/train_regiondiff_sd15_lora_kl.slurm': '#!/bin/bash\n'
                                                        '#SBATCH --job-name=rdiff-sd-lora\n'
                                                        '#SBATCH --account=aip-mpederso\n'
                                                        '#SBATCH --time=24:00:00\n'
                                                        '#SBATCH --cpus-per-task=10\n'
                                                        '#SBATCH --mem=32G\n'
                                                        '#SBATCH --gpus-per-node=h100:1\n'
                                                        '#SBATCH '
                                                        '--output=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.out\n'
                                                        '#SBATCH '
                                                        '--error=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.err\n',
 'slurm/killarney/train_regiondiff_sd_from_lora_r8_fm_comparable_kl.slurm': '#!/bin/bash\n'
                                                                            '#SBATCH --job-name=rdiff-lora-r8\n'
                                                                            '#SBATCH --account=aip-mpederso\n'
                                                                            '#SBATCH --time=24:00:00\n'
                                                                            '#SBATCH --cpus-per-task=10\n'
                                                                            '#SBATCH --mem=32G\n'
                                                                            '#SBATCH --gpus-per-node=h100:1\n'
                                                                            '#SBATCH '
                                                                            '--output=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.out\n'
                                                                            '#SBATCH '
                                                                            '--error=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.err\n',
 'slurm/killarney/train_regiondiff_sd_from_uncond_hflip_kl.slurm': '#!/bin/bash\n'
                                                                   '#SBATCH --job-name=rdiff-sd-hf\n'
                                                                   '#SBATCH --account=aip-mpederso\n'
                                                                   '#SBATCH --time=24:00:00\n'
                                                                   '#SBATCH --cpus-per-task=10\n'
                                                                   '#SBATCH --mem=32G\n'
                                                                   '#SBATCH --gpus-per-node=h100:1\n'
                                                                   '#SBATCH '
                                                                   '--output=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.out\n'
                                                                   '#SBATCH '
                                                                   '--error=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.err\n',
 'slurm/killarney/train_regiondiff_sd_from_uncond_kl.slurm': '#!/bin/bash\n'
                                                             '#SBATCH --job-name=rdiff-sd\n'
                                                             '#SBATCH --account=aip-mpederso\n'
                                                             '#SBATCH --time=24:00:00\n'
                                                             '#SBATCH --cpus-per-task=10\n'
                                                             '#SBATCH --mem=32G\n'
                                                             '#SBATCH --gpus-per-node=h100:1\n'
                                                             '#SBATCH '
                                                             '--output=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.out\n'
                                                             '#SBATCH '
                                                             '--error=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.err\n',
 'slurm/killarney/train_stable_fm_hflip_kl.slurm': '#!/bin/bash\n'
                                                   '#SBATCH --job-name=stable-fm-hf\n'
                                                   '#SBATCH --account=aip-mpederso\n'
                                                   '#SBATCH --time=24:00:00\n'
                                                   '#SBATCH --cpus-per-task=10\n'
                                                   '#SBATCH --mem=32G\n'
                                                   '#SBATCH --gpus-per-node=h100:1\n'
                                                   '#SBATCH '
                                                   '--output=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.out\n'
                                                   '#SBATCH '
                                                   '--error=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.err\n',
 'slurm/killarney/flir/flow_matching/train_stable_fm_hflip_ot_kl.slurm': '#!/bin/bash\n'
                                                      '#SBATCH --job-name=stable-fm-ot-hf\n'
                                                      '#SBATCH --account=aip-mpederso\n'
                                                      '#SBATCH --time=24:00:00\n'
                                                      '#SBATCH --cpus-per-task=10\n'
                                                      '#SBATCH --mem=32G\n'
                                                      '#SBATCH --gpus-per-node=h100:1\n'
                                                      '#SBATCH '
                                                      '--output=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.out\n'
                                                      '#SBATCH '
                                                      '--error=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.err\n',
 'slurm/killarney/train_stable_fm_kl.slurm': '#!/bin/bash\n'
                                             '#SBATCH --job-name=stable-fm\n'
                                             '#SBATCH --account=aip-mpederso\n'
                                             '#SBATCH --time=24:00:00\n'
                                             '#SBATCH --cpus-per-task=10\n'
                                             '#SBATCH --mem=32G\n'
                                             '#SBATCH --gpus-per-node=h100:1\n'
                                             '#SBATCH '
                                             '--output=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.out\n'
                                             '#SBATCH '
                                             '--error=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.err\n',
 'slurm/killarney/flir/diffusion/train_stable_sd_hflip_kl.slurm': '#!/bin/bash\n'
                                                   '#SBATCH --job-name=stable-sd-hf\n'
                                                   '#SBATCH --account=aip-mpederso\n'
                                                   '#SBATCH --time=24:00:00\n'
                                                   '#SBATCH --cpus-per-task=10\n'
                                                   '#SBATCH --mem=32G\n'
                                                   '#SBATCH --gpus-per-node=h100:1\n'
                                                   '#SBATCH '
                                                   '--output=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.out\n'
                                                   '#SBATCH '
                                                   '--error=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.err\n',
 'slurm/killarney/train_stable_sd_kl.slurm': '#!/bin/bash\n'
                                             '#SBATCH --job-name=stable-sd\n'
                                             '#SBATCH --account=aip-mpederso\n'
                                             '#SBATCH --time=24:00:00\n'
                                             '#SBATCH --cpus-per-task=10\n'
                                             '#SBATCH --mem=32G\n'
                                             '#SBATCH --gpus-per-node=h100:1\n'
                                             '#SBATCH '
                                             '--output=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.out\n'
                                             '#SBATCH '
                                             '--error=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.err\n',
 'slurm/killarney/train_stay_layout_fm_hflip_kl.slurm': '#!/bin/bash\n'
                                                        '#SBATCH --job-name=stay-fm-hf\n'
                                                        '#SBATCH --account=aip-mpederso\n'
                                                        '#SBATCH --time=24:00:00\n'
                                                        '#SBATCH --cpus-per-task=20\n'
                                                        '#SBATCH --mem=40G\n'
                                                        '#SBATCH --gpus-per-node=h100:1\n'
                                                        '#SBATCH '
                                                        '--output=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.out\n'
                                                        '#SBATCH '
                                                        '--error=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.err\n',
 'slurm/tamia/train_stable_fm_tamia.slurm': '#!/bin/bash\n'
                                            '#SBATCH --job-name=stable-fm\n'
                                            '#SBATCH --account=aip-mpederso\n'
                                            '#SBATCH --time=24:00:00\n'
                                            '#SBATCH --cpus-per-task=10\n'
                                            '#SBATCH --mem=32G\n'
                                            '#SBATCH --gpus=h100:1\n'
                                            '#SBATCH --output=logs/%x-%j.out\n'
                                            '#SBATCH --error=logs/%x-%j.err\n',
 'slurm/tamia/train_stable_sd_tamia.slurm': '#!/bin/bash\n'
                                            '#SBATCH --job-name=stable-sd\n'
                                            '#SBATCH --account=aip-mpederso\n'
                                            '#SBATCH --time=24:00:00\n'
                                            '#SBATCH --cpus-per-task=10\n'
                                            '#SBATCH --mem=32G\n'
                                            '#SBATCH --gpus=h100:1\n'
                                            '#SBATCH --output=logs/%x-%j.out\n'
                                            '#SBATCH --error=logs/%x-%j.err\n'}

EXPECTED_HEADERS.update({
    'slurm/killarney/flir/sd_adaptation/train_flir_sdxl_lora_stage1_r8_full_kl.slurm': '#!/bin/bash\n'
                                                                                       '#SBATCH --job-name=flir-sdxl-r8-full\n'
                                                                                       '#SBATCH --account=aip-mpederso\n'
                                                                                       '#SBATCH --time=48:00:00\n'
                                                                                       '#SBATCH --cpus-per-task=10\n'
                                                                                       '#SBATCH --mem=80G\n'
                                                                                       '#SBATCH --gpus-per-node=h100:1\n'
                                                                                       '#SBATCH --output=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.out\n'
                                                                                       '#SBATCH --error=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.err\n',
    'slurm/killarney/flir/sd_adaptation/train_flir_sdxl_lora_stage1_r8_train_2000_kl.slurm': '#!/bin/bash\n'
                                                                                             '#SBATCH --job-name=flir-sdxl-r8-2k\n'
                                                                                             '#SBATCH --account=aip-mpederso\n'
                                                                                             '#SBATCH --time=24:00:00\n'
                                                                                             '#SBATCH --cpus-per-task=10\n'
                                                                                             '#SBATCH --mem=80G\n'
                                                                                             '#SBATCH --gpus-per-node=h100:1\n'
                                                                                             '#SBATCH --output=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.out\n'
                                                                                             '#SBATCH --error=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.err\n',
    'slurm/killarney/flir/sd_adaptation/train_flir_sdxl_lora_stage1_r8_train_5000_kl.slurm': '#!/bin/bash\n'
                                                                                             '#SBATCH --job-name=flir-sdxl-r8-5k\n'
                                                                                             '#SBATCH --account=aip-mpederso\n'
                                                                                             '#SBATCH --time=36:00:00\n'
                                                                                             '#SBATCH --cpus-per-task=10\n'
                                                                                             '#SBATCH --mem=80G\n'
                                                                                             '#SBATCH --gpus-per-node=h100:1\n'
                                                                                             '#SBATCH --output=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.out\n'
                                                                                             '#SBATCH --error=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.err\n',
    'slurm/killarney/bigearthnet_s2_b08_5x5_stride3/sd_adaptation/train_bigearthnet_s2_b08_5x5_stride3_sdxl_lora_stage1_r8_full_kl.slurm': '#!/bin/bash\n'
                                                                                                                                          '#SBATCH --job-name=ben-sdxl-r8-full\n'
                                                                                                                                          '#SBATCH --account=aip-mpederso\n'
                                                                                                                                          '#SBATCH --time=48:00:00\n'
                                                                                                                                          '#SBATCH --cpus-per-task=10\n'
                                                                                                                                          '#SBATCH --mem=80G\n'
                                                                                                                                          '#SBATCH --gpus-per-node=h100:1\n'
                                                                                                                                          '#SBATCH --output=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.out\n'
                                                                                                                                          '#SBATCH --error=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.err\n',
    'slurm/killarney/bigearthnet_s2_b08_5x5_stride3/sd_adaptation/train_bigearthnet_s2_b08_5x5_stride3_sdxl_lora_stage1_r8_train_2040_kl.slurm': '#!/bin/bash\n'
                                                                                                                                                '#SBATCH --job-name=ben-sdxl-r8-2040\n'
                                                                                                                                                '#SBATCH --account=aip-mpederso\n'
                                                                                                                                                '#SBATCH --time=24:00:00\n'
                                                                                                                                                '#SBATCH --cpus-per-task=10\n'
                                                                                                                                                '#SBATCH --mem=80G\n'
                                                                                                                                                '#SBATCH --gpus-per-node=h100:1\n'
                                                                                                                                                '#SBATCH --output=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.out\n'
                                                                                                                                                '#SBATCH --error=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.err\n',
    'slurm/killarney/bigearthnet_s2_b08_5x5_stride3/sd_adaptation/train_bigearthnet_s2_b08_5x5_stride3_sdxl_lora_stage1_r8_train_5100_kl.slurm': '#!/bin/bash\n'
                                                                                                                                                '#SBATCH --job-name=ben-sdxl-r8-5100\n'
                                                                                                                                                '#SBATCH --account=aip-mpederso\n'
                                                                                                                                                '#SBATCH --time=36:00:00\n'
                                                                                                                                                '#SBATCH --cpus-per-task=10\n'
                                                                                                                                                '#SBATCH --mem=80G\n'
                                                                                                                                                '#SBATCH --gpus-per-node=h100:1\n'
                                                                                                                                                '#SBATCH --output=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.out\n'
                                                                                                                                                '#SBATCH --error=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.err\n',
})

ok = fail = 0


def check(condition: bool, message: str) -> None:
    global ok, fail
    if condition:
        ok += 1
        print(f"  [PASS] {message}")
    else:
        fail += 1
        print(f"  [FAIL] {message}")


def slurm_header(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if line.startswith("#!") or line.startswith("#SBATCH") or line == "":
            lines.append(line)
            continue
        break
    return "\n".join(lines)


def config_refs(text: str) -> set[str]:
    refs: set[str] = set()
    patterns = [
        r'CONFIG_REL="(?:\$\{CONFIG_REL:-)?([^"}]+)',
        r'PRESET_PATH="\$\{PRESET_PATH:-([^"}]+)',
        r'--stage1-config\s+([^\s\\]+)',
        r'--stage2-config\s+([^\s\\]+)',
        r'\b(configs/[^\s")}\']+)',
    ]
    for pattern in patterns:
        refs.update(match.group(1) for match in re.finditer(pattern, text))
    return {ref for ref in refs if ref.startswith("configs/")}


print("\n=== Helper policy ===")
helper = ROOT / "slurm/lib/common.sh"
check(not helper.exists(), "Slurm launchers do not rely on slurm/lib/common.sh")

print("\n=== Launcher set ===")
actual = {str(path.relative_to(ROOT)) for path in ROOT.glob("slurm/**/*.slurm")}
expected = set(EXPECTED_HEADERS)
check(actual == expected, "Slurm launcher file set is unchanged")
if actual != expected:
    print(f"    extra={sorted(actual - expected)}")
    print(f"    missing={sorted(expected - actual)}")

print("\n=== Migrated launchers ===")
for rel_path in sorted(expected):
    path = ROOT / rel_path
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    check(path.is_file(), f"{rel_path} exists")
    check(slurm_header(text) == EXPECTED_HEADERS[rel_path], f"{rel_path} preserves #SBATCH header")
    check("common.sh" not in text, f"{rel_path} is self-contained")
    check("slurm_" not in text, f"{rel_path} avoids custom Slurm helper calls")
    check("set -euo pipefail" in text, f"{rel_path} enables strict shell mode")
    check("conda activate" in text, f"{rel_path} activates the Conda environment directly")
    if "CONFIG_REL" in text or "PRESET_PATH" in text:
        check(
            '[[ ! -f "${CONFIG}" ]]' in text
            or '[[ ! -f "${PRESET_PATH}" ]]' in text
            or '[[ ! -f "${PROJECT_ROOT}/${CONFIG}" ]]' in text,
            f"{rel_path} resolves/checks config-like paths",
        )
    for ref in sorted(config_refs(text)):
        check((ROOT / ref).is_file(), f"{rel_path} referenced config exists: {ref}")

print(f"\nSlurm launcher checks: {ok} passed, {fail} failed, {ok + fail} total")
if fail:
    raise SystemExit(1)
