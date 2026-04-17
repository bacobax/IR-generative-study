#!/usr/bin/env python3
"""Export clean pip requirements for a Conda environment.

The exporter is designed for Conda environments where some packages were
installed by Conda and others by pip. It keeps only the pip-managed packages
recorded by ``conda list --json`` as ``channel == "pypi"`` and emits a clean,
sorted requirements file that is portable across Linux VMs.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PYTORCH_EXTRA_INDEXES = {
    "cu118": "https://download.pytorch.org/whl/cu118",
    "cu121": "https://download.pytorch.org/whl/cu121",
    "cu124": "https://download.pytorch.org/whl/cu124",
    "cu126": "https://download.pytorch.org/whl/cu126",
    "cu128": "https://download.pytorch.org/whl/cu128",
    "cpu": "https://download.pytorch.org/whl/cpu",
}

PORTABLE_URL_PREFIXES = ("git+", "https://", "http://", "ssh://")
PORTABLE_DIRECT_URL_PREFIXES = ("https://", "http://", "git+", "ssh://")
TORCH_FAMILY = {"torch", "torchaudio", "torchvision"}


class ExportError(RuntimeError):
    """Raised when the export cannot be completed cleanly."""


@dataclass(frozen=True)
class CondaPackage:
    name: str
    version: str
    channel: str


def canonicalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def run_checked(cmd: list[str], *, text: bool = True) -> str:
    try:
        proc = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=text,
        )
    except FileNotFoundError as exc:
        raise ExportError(f"Required command was not found: {cmd[0]}") from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        detail = stderr or stdout or "no output"
        raise ExportError(f"Command failed: {' '.join(cmd)}\n{detail}") from exc
    return proc.stdout


def ensure_conda_available() -> str:
    conda = shutil.which("conda")
    if not conda:
        raise ExportError("Conda was not found on PATH.")
    return conda


def load_conda_env_names(conda: str) -> set[str]:
    raw = run_checked([conda, "env", "list", "--json"])
    data = json.loads(raw)
    env_names: set[str] = set()
    for env_path in data.get("envs", []):
        env_names.add(Path(env_path).name)
    return env_names


def ensure_env_exists(conda: str, env_name: str) -> None:
    env_names = load_conda_env_names(conda)
    if env_name not in env_names:
        raise ExportError(f"Conda environment '{env_name}' was not found.")


def ensure_pip_available(conda: str, env_name: str) -> None:
    run_checked([conda, "run", "-n", env_name, "python", "-m", "pip", "--version"])


def load_conda_packages(conda: str, env_name: str) -> list[CondaPackage]:
    raw = run_checked([conda, "list", "-n", env_name, "--json"])
    data = json.loads(raw)
    packages: list[CondaPackage] = []
    for row in data:
        packages.append(
            CondaPackage(
                name=row["name"],
                version=row["version"],
                channel=row.get("channel", ""),
            )
        )
    return packages


def load_pip_freeze(conda: str, env_name: str) -> list[str]:
    raw = run_checked([conda, "run", "-n", env_name, "python", "-m", "pip", "freeze"])
    return [line.strip() for line in raw.splitlines() if line.strip()]


def parse_requirement_name(requirement_line: str) -> str | None:
    line = requirement_line.strip()
    if not line or line.startswith("#"):
        return None

    editable_prefix = "-e "
    if line.startswith(editable_prefix):
        payload = line[len(editable_prefix) :].strip()
        egg_match = re.search(r"#egg=([A-Za-z0-9_.-]+)", payload)
        if egg_match:
            return egg_match.group(1)
        if " @ " in payload:
            return payload.split(" @ ", 1)[0].strip()
        return Path(payload).name

    if " @ " in line:
        return line.split(" @ ", 1)[0].strip()

    name_match = re.match(r"([A-Za-z0-9_.-]+)==", line)
    if name_match:
        return name_match.group(1)

    return None


def is_portable_requirement_line(requirement_line: str) -> bool:
    line = requirement_line.strip()
    if line.startswith("-e "):
        payload = line[3:].strip()
        if payload.startswith(PORTABLE_URL_PREFIXES):
            return True
        return False

    if " @ " in line:
        _, url = line.split(" @ ", 1)
        return url.startswith(PORTABLE_DIRECT_URL_PREFIXES)

    if line.startswith(("/", ".")) or "file://" in line:
        return False

    return True


def build_freeze_lookup(freeze_lines: Iterable[str]) -> dict[str, list[str]]:
    lookup: dict[str, list[str]] = {}
    for line in freeze_lines:
        name = parse_requirement_name(line)
        if not name:
            continue
        lookup.setdefault(canonicalize_name(name), []).append(line)
    return lookup


def choose_requirement_line(pkg: CondaPackage, freeze_lookup: dict[str, list[str]]) -> tuple[str | None, str | None]:
    matches = freeze_lookup.get(canonicalize_name(pkg.name), [])

    portable_matches = [line for line in matches if is_portable_requirement_line(line)]
    if portable_matches:
        return portable_matches[0], None

    if matches:
        rejected = matches[0]
        comment = f"# Omitted non-portable requirement for {pkg.name}: {rejected}"
        return None, comment

    return f"{pkg.name}=={pkg.version}", None


def infer_pytorch_extra_indexes(requirement_lines: Iterable[str]) -> list[str]:
    indexes: set[str] = set()
    for line in requirement_lines:
        name = parse_requirement_name(line)
        if not name or canonicalize_name(name) not in TORCH_FAMILY:
            continue

        version_match = re.search(r"==([^\s]+)", line)
        if not version_match:
            continue

        version = version_match.group(1)
        local_suffix_match = re.search(r"\+(cu\d+|cpu)$", version)
        if not local_suffix_match:
            continue

        suffix = local_suffix_match.group(1)
        index_url = PYTORCH_EXTRA_INDEXES.get(suffix)
        if index_url:
            indexes.add(index_url)
    return sorted(indexes)


def render_output(requirement_lines: list[str], comments: list[str], extra_indexes: list[str]) -> str:
    body: list[str] = []
    for index_url in extra_indexes:
        body.append(f"--extra-index-url {index_url}")

    if extra_indexes and comments:
        body.append("")

    if comments:
        body.extend(comments)
        body.append("")

    body.extend(requirement_lines)
    body.append("")
    return "\n".join(body)


def export_requirements(env_name: str, output_path: Path) -> dict[str, object]:
    conda = ensure_conda_available()
    ensure_env_exists(conda, env_name)
    ensure_pip_available(conda, env_name)

    conda_packages = load_conda_packages(conda, env_name)
    pip_packages = [pkg for pkg in conda_packages if pkg.channel == "pypi"]
    if not pip_packages:
        raise ExportError(f"No pip-managed packages were found in Conda env '{env_name}'.")

    freeze_lookup = build_freeze_lookup(load_pip_freeze(conda, env_name))

    requirement_set: set[str] = set()
    comments: list[str] = []

    for pkg in sorted(pip_packages, key=lambda item: canonicalize_name(item.name)):
        line, comment = choose_requirement_line(pkg, freeze_lookup)
        if line:
            requirement_set.add(line)
        if comment:
            comments.append(comment)

    requirement_lines = sorted(requirement_set, key=lambda line: canonicalize_name(parse_requirement_name(line) or line))
    comments = sorted(set(comments), key=str.casefold)
    extra_indexes = infer_pytorch_extra_indexes(requirement_lines)

    output_text = render_output(requirement_lines, comments, extra_indexes)
    output_path.write_text(output_text, encoding="utf-8")

    return {
        "output_path": str(output_path),
        "env_name": env_name,
        "pip_package_count": len(pip_packages),
        "written_requirement_count": len(requirement_lines),
        "omitted_nonportable_count": len(comments),
        "extra_index_count": len(extra_indexes),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-name",
        default="diffusers-dev",
        help="Source Conda environment name. Default: %(default)s",
    )
    parser.add_argument(
        "--output",
        default="requirements-pip-clean.txt",
        help="Output requirements file. Default: %(default)s",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    output_path = Path(args.output).resolve()

    try:
        result = export_requirements(args.env_name, output_path)
    except ExportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
