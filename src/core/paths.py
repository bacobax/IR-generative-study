"""Canonical path helpers for the flow-matching-trial repository.

This module is the **single source of truth** for every well-known path in
the repository.  Prefer importing helpers from here over hard-coding paths.

All helpers resolve relative to the repository root so that scripts work
regardless of the current working directory.
"""

from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════
# Root helpers
# ═══════════════════════════════════════════════════════════════════════════

def repo_root() -> Path:
    """Absolute path to the repository root.

    Determined by walking up from ``src/core/paths.py``.
    """
    return Path(__file__).resolve().parents[2]


# ── layout roots ──────────────────────────────────────────────────────────

def data_root() -> Path:
    """``<repo>/data/`` — top-level data directory."""
    return repo_root() / "data"


def raw_data_root() -> Path:
    """``<repo>/data/raw/`` — original / unprocessed datasets."""
    return data_root() / "raw"


def derived_data_root() -> Path:
    """``<repo>/data/derived/`` — preprocessed / feature datasets."""
    return data_root() / "derived"


def cache_root() -> Path:
    """``<repo>/data/cache/`` — ephemeral caches."""
    return data_root() / "cache"


def artifacts_root() -> Path:
    """``<repo>/artifacts/`` — all training outputs."""
    return repo_root() / "artifacts"


def archive_root() -> Path:
    """``<repo>/archive/`` — retired material."""
    return repo_root() / "archive"


def legacy_code_root() -> Path:
    """``<repo>/archive/legacy_code/`` — retired source trees."""
    return archive_root() / "legacy_code"


def configs_root() -> Path:
    """``<repo>/configs/`` — YAML / JSON configuration files."""
    return repo_root() / "configs"


# ═══════════════════════════════════════════════════════════════════════════
# Specific dataset helpers
# ═══════════════════════════════════════════════════════════════════════════

def v18_root() -> Path:
    """``data/raw/v18/`` — the main IR image dataset."""
    return raw_data_root() / "v18"


def default_data_dir(split: str = "train") -> Path:
    """Return ``data/raw/v18/<split>/`` for a train/val split."""
    return v18_root() / split


def surprise_pred_dataset_root() -> Path:
    """``data/derived/surprise_pred_dataset/``."""
    return derived_data_root() / "surprise_pred_dataset"


def dino_cache_dir() -> Path:
    """``data/cache/dino_cache/`` — cached DINOv2 features."""
    return cache_root() / "dino_cache"


# ═══════════════════════════════════════════════════════════════════════════
# Model config helpers
# ═══════════════════════════════════════════════════════════════════════════

def fm_model_configs_dir() -> Path:
    """``configs/models/fm/`` — FM UNet + VAE architecture configs."""
    return configs_root() / "models" / "fm"


def stable_unet_config_path() -> Path:
    """Canonical path to the stable UNet architecture JSON."""
    return fm_model_configs_dir() / "stable_unet_config.json"


def non_stable_unet_config_path() -> Path:
    """Canonical path to the non-stable (pixel-space) UNet JSON."""
    return fm_model_configs_dir() / "non_stable_unet_config.json"


def vae_config_path() -> Path:
    """Canonical path to the VAE architecture JSON."""
    return fm_model_configs_dir() / "vae_config.json"


def vae_config_x8_path() -> Path:
    """Canonical path to the VAE x8 architecture JSON."""
    return fm_model_configs_dir() / "vae_config_x8.json"


# ═══════════════════════════════════════════════════════════════════════════
# Generated / debug output helpers
# ═══════════════════════════════════════════════════════════════════════════

def generated_root() -> Path:
    """``artifacts/generated/`` — all generated image datasets."""
    return artifacts_root() / "generated"


def default_outputs_dir() -> Path:
    """``artifacts/generated/main/`` — primary generated image datasets."""
    return generated_root() / "main"


def generated_test_dir() -> Path:
    """``artifacts/generated/test/`` — test-time generated datasets."""
    return generated_root() / "test"


def debug_root() -> Path:
    """``artifacts/debug/`` — debug output root."""
    return artifacts_root() / "debug"


def debug_samples_dir() -> Path:
    """``artifacts/debug/debug_samples/`` — saved debug images."""
    return debug_root() / "debug_samples"


# ═══════════════════════════════════════════════════════════════════════════
# Analysis output helpers
# ═══════════════════════════════════════════════════════════════════════════

def analysis_root() -> Path:
    """``artifacts/analysis/`` — all analysis outputs."""
    return artifacts_root() / "analysis"


def default_analysis_dir() -> Path:
    """``artifacts/analysis/main/`` — primary analysis results."""
    return analysis_root() / "main"


def analysis_test_dir() -> Path:
    """``artifacts/analysis/test/`` — test analysis results."""
    return analysis_root() / "test"


# ═══════════════════════════════════════════════════════════════════════════
# Checkpoint helpers
# ═══════════════════════════════════════════════════════════════════════════

def checkpoints_root() -> Path:
    """``artifacts/checkpoints/`` — all model checkpoints."""
    return artifacts_root() / "checkpoints"


def fm_checkpoints_root() -> Path:
    """``artifacts/checkpoints/flow_matching/`` — flow matching checkpoints."""
    return checkpoints_root() / "flow_matching"


def default_models_dir() -> Path:
    """``artifacts/checkpoints/flow_matching/serious_runs/`` — persisted FM runs."""
    return fm_checkpoints_root() / "serious_runs"


def vae_checkpoints_root() -> Path:
    """``artifacts/checkpoints/vae/`` — VAE checkpoints."""
    return checkpoints_root() / "vae"


def vae_runs_dir() -> Path:
    """``artifacts/checkpoints/vae/vae_runs/`` — VAE experiment runs."""
    return vae_checkpoints_root() / "vae_runs"


def sd_checkpoints_root() -> Path:
    """``artifacts/checkpoints/stable_diffusion/`` — SD checkpoints."""
    return checkpoints_root() / "stable_diffusion"


def sd_lora_runs_dir() -> Path:
    """``artifacts/checkpoints/stable_diffusion/lora_runs/`` — LoRA runs."""
    return sd_checkpoints_root() / "lora_runs"


def count_adapter_checkpoints_root() -> Path:
    """``artifacts/checkpoints/count_adapter/`` — count adapter checkpoints."""
    return checkpoints_root() / "count_adapter"


def count_adapter_runs_dir() -> Path:
    """``artifacts/checkpoints/count_adapter/runs/`` — count adapter runs."""
    return count_adapter_checkpoints_root() / "runs"


def legacy_checkpoints_root() -> Path:
    """``artifacts/checkpoints/legacy/`` — legacy pipeline_model, UNET, VAE."""
    return checkpoints_root() / "legacy"


# ═══════════════════════════════════════════════════════════════════════════
# Run / log helpers
# ═══════════════════════════════════════════════════════════════════════════

def runs_root() -> Path:
    """``artifacts/runs/`` — TensorBoard / experiment logs."""
    return artifacts_root() / "runs"


def default_runs_dir() -> Path:
    """``artifacts/runs/main/`` — primary experiment logs."""
    return runs_root() / "main"


def runs_test_dir() -> Path:
    """``artifacts/runs/test/`` — test experiment logs."""
    return runs_root() / "test"
