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
 'slurm/killarney/train_flir_unet_full_domainstudio_512_kl.slurm': '#!/bin/bash\n'
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
 'slurm/killarney/train_stable_fm_hflip_ot_kl.slurm': '#!/bin/bash\n'
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
 'slurm/killarney/train_stable_sd_hflip_kl.slurm': '#!/bin/bash\n'
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
    ]
    for pattern in patterns:
        refs.update(match.group(1) for match in re.finditer(pattern, text))
    return {ref for ref in refs if ref.startswith("configs/")}


print("\n=== Helper ===")
helper = ROOT / "slurm/lib/common.sh"
helper_text = helper.read_text(encoding="utf-8") if helper.is_file() else ""
check(helper.is_file(), "slurm/lib/common.sh exists")
for needle in (
    "set -euo pipefail",
    "slurm_init_runtime()",
    "slurm_activate_conda()",
    "slurm_require_file()",
    "slurm_require_path()",
    "slurm_config_path()",
    "slurm_print_python_diagnostics()",
    "slurm_print_gpu_diagnostics()",
    "slurm_grep_config_keys()",
    "slurm_run_timed()",
    "/usr/bin/time -v",
):
    check(needle in helper_text, f"common helper defines {needle}")

print("\n=== Launcher set ===")
actual = {str(path.relative_to(ROOT)) for path in ROOT.glob("slurm/*/*.slurm")}
expected = set(EXPECTED_HEADERS)
check(actual == expected, "Slurm launcher file set is unchanged")
if actual != expected:
    print(f"    extra={sorted(actual - expected)}")
    print(f"    missing={sorted(expected - actual)}")

print("\n=== Migrated launchers ===")
DIRECT_RUNTIME_LAUNCHERS = {
    "slurm/killarney/train_flir_unet_full_domainstudio_512_kl.slurm",
    "slurm/killarney/train_stable_fm_hflip_ot_kl.slurm",
    "slurm/killarney/train_stable_sd_hflip_kl.slurm",
}
for rel_path in sorted(expected):
    path = ROOT / rel_path
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    check(path.is_file(), f"{rel_path} exists")
    check(slurm_header(text) == EXPECTED_HEADERS[rel_path], f"{rel_path} preserves #SBATCH header")
    if rel_path in DIRECT_RUNTIME_LAUNCHERS:
        check('source "${SCRIPT_DIR}/../lib/common.sh"' not in text, f"{rel_path} is self-contained")
        check("slurm_" not in text, f"{rel_path} avoids custom Slurm helper calls")
        check("set -euo pipefail" in text, f"{rel_path} enables strict shell mode")
        check("conda activate" in text, f"{rel_path} activates the Conda environment directly")
        check("/usr/bin/time -v" not in text, f"{rel_path} uses Slurm logs directly")
    else:
        check('source "${SCRIPT_DIR}/../lib/common.sh"' in text, f"{rel_path} sources Slurm common helper")
        check("slurm_init_runtime" in text, f"{rel_path} initializes shared runtime")
        check("slurm_run_timed" in text, f"{rel_path} uses timed command helper")
        check("/usr/bin/time -v" not in text, f"{rel_path} does not duplicate raw time invocation")
    if "CONFIG_REL" in text or "PRESET_PATH" in text:
        check(
            "slurm_config_path" in text or "slurm_require_file" in text or '[[ ! -f "${CONFIG}" ]]' in text,
            f"{rel_path} resolves/checks config-like paths",
        )
    for ref in sorted(config_refs(text)):
        check((ROOT / ref).is_file(), f"{rel_path} referenced config exists: {ref}")

print(f"\nSlurm launcher checks: {ok} passed, {fail} failed, {ok + fail} total")
if fail:
    raise SystemExit(1)
