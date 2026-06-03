#!/usr/bin/env python
"""Static checks for Slurm launcher organization and helper usage."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

ok = fail = 0


def check(condition: bool, message: str) -> None:
    global ok, fail
    if condition:
        ok += 1
        print(f"  [PASS] {message}")
    else:
        fail += 1
        print(f"  [FAIL] {message}")


def config_refs(text: str) -> set[str]:
    refs: set[str] = set()
    patterns = [
        r'CONFIG_REL="(?:\$\{CONFIG_REL:-)?([^"}]+)',
        r'PRESET_PATH="\$\{PRESET_PATH:-([^"}]+)',
        r"--stage1-config\s+([^\s\\]+)",
        r"--stage2-config\s+([^\s\\]+)",
        r"\b(configs/[^\s\")}'<>]+)",
    ]
    for pattern in patterns:
        refs.update(match.group(1) for match in re.finditer(pattern, text))
    return {ref for ref in refs if ref.startswith("configs/")}


def sbatch_job_name(text: str) -> str | None:
    match = re.search(r"^#SBATCH --job-name=([^\n]+)", text, re.MULTILINE)
    return match.group(1).strip() if match else None


print("\n=== Helper policy ===")
helper = ROOT / "slurm/lib/common.sh"
check(not helper.exists(), "Slurm launchers do not rely on slurm/lib/common.sh")

print("\n=== Directory organization ===")
for directory in sorted(path for path in (ROOT / "slurm").rglob("*") if path.is_dir()):
    files = [path for path in directory.iterdir() if path.is_file()]
    suffixes = {path.suffix for path in files}
    if suffixes:
        check(
            suffixes <= {".slurm"} or suffixes <= {".sh"},
            f"{directory.relative_to(ROOT)} has one launcher file type",
        )

print("\n=== Slurm launchers ===")
for path in sorted((ROOT / "slurm").rglob("*.slurm")):
    rel_path = path.relative_to(ROOT)
    text = path.read_text(encoding="utf-8")
    check(text.startswith("#!/bin/bash"), f"{rel_path} has Bash shebang")
    check("common.sh" not in text, f"{rel_path} is self-contained")
    check("slurm_" not in text, f"{rel_path} avoids custom Slurm helper calls")
    check("set -euo pipefail" in text, f"{rel_path} enables strict shell mode")
    check("conda activate" in text, f"{rel_path} activates the Conda environment directly")
    if rel_path.parts[-2] in {"fm_scratch", "diff_scratch", "lora_sd15", "lora_sdxl"}:
        check(path.stem == sbatch_job_name(text), f"{rel_path} filename matches #SBATCH job name")
    if "CONFIG_REL" in text or "PRESET_PATH" in text:
        check(
            '[[ ! -f "${CONFIG}" ]]' in text
            or '[[ ! -f "${PRESET_PATH}" ]]' in text
            or '[[ ! -f "${PROJECT_ROOT}/${CONFIG}" ]]' in text,
            f"{rel_path} resolves/checks config-like paths",
        )
    for ref in sorted(config_refs(text)):
        check((ROOT / ref).is_file(), f"{rel_path} referenced config exists: {ref}")

print("\n=== Dataset launchers ===")
launchers = [
    path
    for dataset in ("ben", "flir")
    for path in (ROOT / "slurm").glob(f"*/{dataset}/launch_*.sh")
]
for path in sorted(launchers):
    rel_path = path.relative_to(ROOT)
    dataset = path.parent.name
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("sbatch "):
            continue
        target = line.split(maxsplit=1)[1]
        check((ROOT / target).is_file(), f"{rel_path} references existing launcher: {target}")
        check(f"/{dataset}/" in f"/{target}", f"{rel_path} submits only {dataset} jobs: {target}")

print(f"\nSlurm launcher checks: {ok} passed, {fail} failed, {ok + fail} total")
if fail:
    raise SystemExit(1)
