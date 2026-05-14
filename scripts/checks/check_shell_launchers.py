#!/usr/bin/env python
"""Check simple shell launchers use the shared helper safely."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

MIGRATED = {
    "scripts/analyze/distribution_shift.sh": "configs/analysis/presets/distribution_shift.yaml",
    "scripts/analyze/distribution_shift_test.sh": "configs/analysis/presets/distribution_shift.yaml",
    "scripts/generate/fm_plain.sh": "configs/fm/generate/presets/plain_100_steps.yaml",
    "scripts/generate/fm_test.sh": "configs/fm/generate/presets/plain_100_steps.yaml",
    "scripts/generate/sd_r16.sh": "configs/sd/generate/presets/r16.yaml",
    "scripts/generate/sd_r16_generic.sh": "configs/sd/generate/presets/r16_generic.yaml",
    "scripts/generate/sd_r4.sh": "configs/sd/generate/presets/r4.yaml",
    "scripts/generate/sd_r4_generic.sh": "configs/sd/generate/presets/r4_generic.yaml",
    "scripts/generate/sd_r8.sh": "configs/sd/generate/presets/r8.yaml",
    "scripts/generate/sd_r8_generic.sh": "configs/sd/generate/presets/r8_generic.yaml",
    "scripts/generate/text_fm_cfg.sh": "configs/fm/sample/presets/text_cfg.yaml",
    "scripts/train/controlnet.sh": "configs/controlnet/train/presets/bbox_controlnet.yaml",
    "scripts/train/count_adapter.sh": "configs/auxiliary/count_adapter/presets/run_07.yaml",
    "scripts/train/sd_flir_regiondiff_from_lora_r8.sh": "configs/sd_layout/train/presets/flir_regiondiff_sd15_lora_stage2_r8.yaml",
    "scripts/train/sd_flir_unet_full_stage1.sh": "configs/sd/train/presets/flir_unet_full_stage1.yaml",
    "scripts/train/sd_flir_unet_partial_stage1.sh": "configs/sd/train/presets/flir_unet_partial_stage1.yaml",
    "scripts/train/sd_lora_r16.sh": "configs/sd/train/presets/lora_r16.yaml",
    "scripts/train/sd_lora_r16_generic.sh": "configs/sd/train/presets/lora_r16_generic.yaml",
    "scripts/train/sd_lora_r4.sh": "configs/sd/train/presets/lora_r4.yaml",
    "scripts/train/sd_lora_r4_generic.sh": "configs/sd/train/presets/lora_r4_generic.yaml",
    "scripts/train/sd_lora_r8.sh": "configs/sd/train/presets/lora_r8.yaml",
    "scripts/train/sd_lora_r8_generic.sh": "configs/sd/train/presets/lora_r8_generic.yaml",
    "scripts/train/stable_fm.sh": "configs/fm/train/presets/stable_latent.yaml",
    "scripts/train/stay_layout_latent_flir_sd15_512.sh": "configs/fm/train/presets/stay_layout_latent_flir_sd15_512.yaml",
    "scripts/train/text_fm.sh": "configs/fm/train/presets/text_cfg.yaml",
    "scripts/train/uncond_fm_latent_flir_sd15_512.sh": "configs/fm/train/presets/uncond_latent_flir_sd15_512.yaml",
    "scripts/train/uncond_latent_flir_sd15_512.sh": "configs/sd_uncond/train/presets/uncond_latent_flir_sd15_512.yaml",
    "scripts/train/unstable_fm.sh": "configs/fm/train/presets/pixel_x0.yaml",
    "scripts/train/vae_4x.sh": "configs/vae/train/presets/flir_private_proxy_alignment_v18_vae_x4_512.yaml",
    "scripts/train/vae_8x.sh": "configs/vae/train/presets/vae_8x.yaml",
    "scripts/train/vae_sd15_v18_256.sh": "configs/vae/train/presets/v18_sd15_vae_x8_256.yaml",
}

BESPOKE = {
    "scripts/analyze/run_generated_layout_audits_rare_layout_20260415_161641.sh",
    "scripts/generate/qcmp_stay_layout_fm_hflip.sh",
    "scripts/generate/qcmp_uncond_fm_hflip_ot.sh",
    "scripts/generate/qcmp_uncond_sd_hflip.sh",
    "scripts/train/sd_flir_lora_stage1.sh",  # Existing wrapper points at a missing legacy preset.
    "scripts/train/stay_layout_pixel_flir_smoke200.sh",
    "scripts/train/stay_layout_pixel_flir_v2.sh",
    "scripts/train/stay_layout_pixel_flir_v2_smoke200.sh",
}


ok = fail = 0


def check(condition: bool, message: str) -> None:
    global ok, fail
    if condition:
        ok += 1
        print(f"  [PASS] {message}")
    else:
        fail += 1
        print(f"  [FAIL] {message}")


print("\n=== Shared helper ===")
helper = ROOT / "scripts/lib/common.sh"
helper_src = helper.read_text(encoding="utf-8") if helper.is_file() else ""
check(helper.is_file(), "scripts/lib/common.sh exists")
for needle in ("set -euo pipefail", "enter_repo_root()", "require_config()", "run_python_module_config()", "run_python_script_config()", "run_accelerate_module_config()"):
    check(needle in helper_src, f"common.sh defines {needle}")

print("\n=== Launcher coverage ===")
all_shell = {
    str(path.relative_to(ROOT))
    for folder in ("scripts/train", "scripts/generate", "scripts/analyze")
    for path in (ROOT / folder).glob("*.sh")
}
expected = set(MIGRATED) | BESPOKE
check(all_shell == expected, "all shell launchers are migrated or explicitly allowlisted")
if all_shell != expected:
    print(f"    extra={sorted(all_shell - expected)}")
    print(f"    missing={sorted(expected - all_shell)}")

print("\n=== Migrated wrappers ===")
for rel_path, config_rel in sorted(MIGRATED.items()):
    path = ROOT / rel_path
    if not path.is_file():
        check(False, f"{rel_path} exists")
        continue
    src = path.read_text(encoding="utf-8")
    check("source \"${SCRIPT_DIR}/../lib/common.sh\"" in src, f"{rel_path} sources common.sh")
    check("enter_repo_root \"${SCRIPT_DIR}\"" in src, f"{rel_path} enters repo root")
    check(("run_python_module_config" in src or "run_python_script_config" in src or "run_accelerate_module_config" in src or "require_config" in src), f"{rel_path} checks config before launch")
    check("\"$@\"" in src, f"{rel_path} preserves CLI passthrough")
    check(config_rel in src, f"{rel_path} references expected config")
    check((ROOT / config_rel).is_file(), f"{rel_path} config exists")

print("\n=== Bespoke wrappers ===")
for rel_path in sorted(BESPOKE):
    check((ROOT / rel_path).is_file(), f"{rel_path} exists")

print(f"\nShell launcher checks: {ok} passed, {fail} failed, {ok + fail} total")
if fail:
    raise SystemExit(1)
