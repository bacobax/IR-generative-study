#!/usr/bin/env bash
set -euo pipefail

# ----------------------------
# Config
# ----------------------------

CLUSTER="${CLUSTER:-}"
DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-latest}"

# ----------------------------
# Resolve project root safely
# ----------------------------

if [[ -z "${PROJECT_ROOT:-}" ]]; then
  PROJECT_ROOT="$(pwd -P)"
else
  PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd -P)"
fi

# If launched from ~/projects/<repo>, transparently map to /project/<repo>
if [[ "${PROJECT_ROOT}" == "${HOME}/projects/"* ]]; then
  PROJECT_CANDIDATE="/project/${PROJECT_ROOT#"${HOME}/projects/"}"
  if [[ -d "${PROJECT_CANDIDATE}" ]]; then
    PROJECT_ROOT="$(cd "${PROJECT_CANDIDATE}" && pwd -P)"
  fi
fi

# Killarney rejects sbatch submissions from /home
if [[ "${PROJECT_ROOT}" == /home/* ]]; then
  echo "ERROR: PROJECT_ROOT is under /home, but sbatch submission from /home is not allowed." >&2
  echo "Current PROJECT_ROOT: ${PROJECT_ROOT}" >&2
  echo "Run this script from the repo under /project or /scratch, or use:" >&2
  echo "  PROJECT_ROOT=/project/<repo> bash $0" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
mkdir -p logs

# ----------------------------
# Detect cluster
# ----------------------------

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

# ----------------------------
# Jobs
# ----------------------------

if [[ "${CLUSTER}" == "fir" ]]; then
  RUN_SPECS=(
    "flir_train_5000|artifacts/checkpoints/stable_diffusion_xl/lora_runs/flir_sdxl_lora_stage1_r8_train_5000|slurm/fir/flir/sd_adaptation/train_flir_sdxl_lora_stage1_r8_train_5000_fir.slurm"
    "flir_full|artifacts/checkpoints/stable_diffusion_xl/lora_runs/flir_sdxl_lora_stage1_r8|slurm/fir/flir/sd_adaptation/train_flir_sdxl_lora_stage1_r8_full_fir.slurm"
    "bigearthnet_train_5100|artifacts/checkpoints/stable_diffusion_xl/lora_runs/bigearthnet_s2_b08_5x5_stride3_sdxl_lora_stage1_r8_train_5100|slurm/fir/bigearthnet_s2_b08_5x5_stride3/sd_adaptation/train_bigearthnet_s2_b08_5x5_stride3_sdxl_lora_stage1_r8_train_5100_fir.slurm"
    "bigearthnet_full|artifacts/checkpoints/stable_diffusion_xl/lora_runs/bigearthnet_s2_b08_5x5_stride3_sdxl_lora_stage1_r8|slurm/fir/bigearthnet_s2_b08_5x5_stride3/sd_adaptation/train_bigearthnet_s2_b08_5x5_stride3_sdxl_lora_stage1_r8_full_fir.slurm"
  )
else
  RUN_SPECS=(
    "flir_train_5000|artifacts/checkpoints/stable_diffusion_xl/lora_runs/flir_sdxl_lora_stage1_r8_train_5000|slurm/killarney/flir/sd_adaptation/train_flir_sdxl_lora_stage1_r8_train_5000_kl.slurm"
    "flir_full|artifacts/checkpoints/stable_diffusion_xl/lora_runs/flir_sdxl_lora_stage1_r8|slurm/killarney/flir/sd_adaptation/train_flir_sdxl_lora_stage1_r8_full_kl.slurm"
    "bigearthnet_train_5100|artifacts/checkpoints/stable_diffusion_xl/lora_runs/bigearthnet_s2_b08_5x5_stride3_sdxl_lora_stage1_r8_train_5100|slurm/killarney/bigearthnet_s2_b08_5x5_stride3/sd_adaptation/train_bigearthnet_s2_b08_5x5_stride3_sdxl_lora_stage1_r8_train_5100_kl.slurm"
    "bigearthnet_full|artifacts/checkpoints/stable_diffusion_xl/lora_runs/bigearthnet_s2_b08_5x5_stride3_sdxl_lora_stage1_r8|slurm/killarney/bigearthnet_s2_b08_5x5_stride3/sd_adaptation/train_bigearthnet_s2_b08_5x5_stride3_sdxl_lora_stage1_r8_full_kl.slurm"
  )
fi

echo "Project root: ${PROJECT_ROOT}"
echo "Cluster: ${CLUSTER}"
echo "Resume checkpoint: ${RESUME_FROM_CHECKPOINT}"
echo "Dry run: ${DRY_RUN}"
echo "Force completed runs: ${FORCE}"
echo

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
    echo "SKIP ${label}: final weights already exist"
    skipped=$((skipped + 1))
    continue
  fi

  cmd=(
    sbatch
    --chdir="${PROJECT_ROOT}"
    --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT}"
    "${worker}"
  )

  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'DRY RUN: '
    printf '%q ' "${cmd[@]}"
    echo
  else
    echo "SUBMIT ${label}"
    "${cmd[@]}"
  fi

  submitted=$((submitted + 1))
done

echo
echo "Submitted: ${submitted}"
echo "Skipped: ${skipped}"