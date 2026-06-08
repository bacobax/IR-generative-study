#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CLUSTER="${CLUSTER:-}"
DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-latest}"

if [[ -z "${CLUSTER}" ]]; then
  case "$(hostname)" in
    *fir*)
      CLUSTER="fir"
      ;;
    *kl*|*killarney*)
      CLUSTER="killarney"
      ;;
    *)
      CLUSTER="killarney"
      ;;
  esac
fi

if [[ "${CLUSTER}" != "fir" && "${CLUSTER}" != "killarney" ]]; then
  echo "ERROR: CLUSTER must be 'fir' or 'killarney'." >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
mkdir -p logs

if [[ "${CLUSTER}" == "fir" ]]; then
  RUN_SPECS=(
    # "flir_train_2000|artifacts/checkpoints/stable_diffusion_xl/lora_runs/flir_sdxl_lora_stage1_r8_train_2000|slurm/fir/flir/sd_adaptation/train_flir_sdxl_lora_stage1_r8_train_2000_fir.slurm"
    "flir_train_5000|artifacts/checkpoints/stable_diffusion_xl/lora_runs/flir_sdxl_lora_stage1_r8_train_5000|slurm/fir/flir/sd_adaptation/train_flir_sdxl_lora_stage1_r8_train_5000_fir.slurm"
    "flir_full|artifacts/checkpoints/stable_diffusion_xl/lora_runs/flir_sdxl_lora_stage1_r8|slurm/fir/flir/sd_adaptation/train_flir_sdxl_lora_stage1_r8_full_fir.slurm"
    # "bigearthnet_train_2040|artifacts/checkpoints/stable_diffusion_xl/lora_runs/bigearthnet_s2_b08_5x5_stride3_sdxl_lora_stage1_r8_train_2040|slurm/fir/bigearthnet_s2_b08_5x5_stride3/sd_adaptation/train_bigearthnet_s2_b08_5x5_stride3_sdxl_lora_stage1_r8_train_2040_fir.slurm"
    "bigearthnet_train_5100|artifacts/checkpoints/stable_diffusion_xl/lora_runs/bigearthnet_s2_b08_5x5_stride3_sdxl_lora_stage1_r8_train_5100|slurm/fir/bigearthnet_s2_b08_5x5_stride3/sd_adaptation/train_bigearthnet_s2_b08_5x5_stride3_sdxl_lora_stage1_r8_train_5100_fir.slurm"
    "bigearthnet_full|artifacts/checkpoints/stable_diffusion_xl/lora_runs/bigearthnet_s2_b08_5x5_stride3_sdxl_lora_stage1_r8|slurm/fir/bigearthnet_s2_b08_5x5_stride3/sd_adaptation/train_bigearthnet_s2_b08_5x5_stride3_sdxl_lora_stage1_r8_full_fir.slurm"
  )
else
  RUN_SPECS=(
    # "flir_train_2000|artifacts/checkpoints/stable_diffusion_xl/lora_runs/flir_sdxl_lora_stage1_r8_train_2000|slurm/killarney/flir/sd_adaptation/train_flir_sdxl_lora_stage1_r8_train_2000_kl.slurm"
    "flir_train_5000|artifacts/checkpoints/stable_diffusion_xl/lora_runs/flir_sdxl_lora_stage1_r8_train_5000|slurm/killarney/flir/sd_adaptation/train_flir_sdxl_lora_stage1_r8_train_5000_kl.slurm"
    "flir_full|artifacts/checkpoints/stable_diffusion_xl/lora_runs/flir_sdxl_lora_stage1_r8|slurm/killarney/flir/sd_adaptation/train_flir_sdxl_lora_stage1_r8_full_kl.slurm"
    # "bigearthnet_train_2040|artifacts/checkpoints/stable_diffusion_xl/lora_runs/bigearthnet_s2_b08_5x5_stride3_sdxl_lora_stage1_r8_train_2040|slurm/killarney/bigearthnet_s2_b08_5x5_stride3/sd_adaptation/train_bigearthnet_s2_b08_5x5_stride3_sdxl_lora_stage1_r8_train_2040_kl.slurm"
    "bigearthnet_train_5100|artifacts/checkpoints/stable_diffusion_xl/lora_runs/bigearthnet_s2_b08_5x5_stride3_sdxl_lora_stage1_r8_train_5100|slurm/killarney/bigearthnet_s2_b08_5x5_stride3/sd_adaptation/train_bigearthnet_s2_b08_5x5_stride3_sdxl_lora_stage1_r8_train_5100_kl.slurm"
    "bigearthnet_full|artifacts/checkpoints/stable_diffusion_xl/lora_runs/bigearthnet_s2_b08_5x5_stride3_sdxl_lora_stage1_r8|slurm/killarney/bigearthnet_s2_b08_5x5_stride3/sd_adaptation/train_bigearthnet_s2_b08_5x5_stride3_sdxl_lora_stage1_r8_full_kl.slurm"
  )
fi

echo "Project root: ${PROJECT_ROOT}"
echo "Cluster: ${CLUSTER}"
echo "Resume checkpoint: ${RESUME_FROM_CHECKPOINT}"
echo "Dry run: ${DRY_RUN}"
echo "Force completed runs: ${FORCE}"

submitted=0
skipped=0

for spec in "${RUN_SPECS[@]}"; do
  IFS='|' read -r label output_rel worker_rel <<< "${spec}"
  output_dir="${PROJECT_ROOT}/${output_rel}"
  worker="${PROJECT_ROOT}/${worker_rel}"
  final_weights="${output_dir}/pytorch_lora_weights.safetensors"

  if [[ ! -f "${worker}" ]]; then
    echo "ERROR: missing worker for ${label}: ${worker}" >&2
    exit 1
  fi

  if [[ "${FORCE}" != "1" && -f "${final_weights}" ]]; then
    echo "SKIP ${label}: final weights already exist at ${final_weights}"
    skipped=$((skipped + 1))
    continue
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo sbatch --chdir="${PROJECT_ROOT}" --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT}" "${worker}"
  else
    sbatch --chdir="${PROJECT_ROOT}" --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT}" "${worker}"
  fi
  submitted=$((submitted + 1))
done

echo "Submitted: ${submitted}"
echo "Skipped: ${skipped}"
