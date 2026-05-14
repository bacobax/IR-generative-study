from __future__ import annotations

from pathlib import Path

import yaml

from src.core.configs.config_loader import load_yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
ACTIVE_FM_GENERATE_PRESETS = REPO_ROOT / "configs" / "fm" / "generate" / "presets"
RETIRED_GUIDED_KEYS = {
    "fm_guidance_method",
    "fm_surprise_ckpt",
    "fm_energy_mode",
    "fm_w_surprise",
    "fm_w_gmm",
}


def _active_fm_generation_presets() -> list[Path]:
    return sorted(ACTIVE_FM_GENERATE_PRESETS.glob("*.yaml"))


def _read_raw_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def test_active_fm_generation_presets_resolve_identically_without_extends() -> None:
    presets = _active_fm_generation_presets()
    assert presets, "Expected at least one active FM generation preset."

    for preset in presets:
        raw = _read_raw_yaml(preset)
        if "extends" in raw:
            continue
        assert load_yaml(preset) == raw


def test_active_fm_generation_presets_do_not_reintroduce_retired_guided_keys() -> None:
    presets = _active_fm_generation_presets()
    assert presets, "Expected at least one active FM generation preset."

    for preset in presets:
        resolved = load_yaml(preset)
        assert not (set(resolved) & RETIRED_GUIDED_KEYS), preset


def test_plain_100_steps_generation_preset_still_has_expected_core_fields() -> None:
    preset = ACTIVE_FM_GENERATE_PRESETS / "plain_100_steps.yaml"
    resolved = load_yaml(preset)

    assert resolved["mode"] == "fm"
    assert resolved["fm_steps"] == 100
    assert resolved["max_samples"] == 200
    assert resolved["output_dir"] == "./artifacts/generated/main/fm_100_steps"
    assert resolved["fm_pipeline_dir"].endswith("stable_training_no_norm")
    assert resolved["fm_vae_weights"].endswith("VAE/vae_best.pt")
