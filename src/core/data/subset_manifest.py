"""Helpers for manifest-defined training subsets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from src.core.paths import repo_root


def resolve_subset_manifest_path(
    manifest: str | Path,
    *,
    dataset_root: str | Path | None = None,
    split_dir: str | Path | None = None,
) -> Path:
    """Resolve an absolute, repo-relative, or dataset ``subsets/`` manifest."""
    manifest_path = Path(manifest)
    candidates: list[Path] = []

    if manifest_path.is_absolute():
        candidates.append(manifest_path)
    else:
        candidates.append(repo_root() / manifest_path)
        if dataset_root is not None:
            candidates.append(Path(dataset_root) / "subsets" / manifest_path)
        if split_dir is not None:
            split_path = Path(split_dir)
            candidates.append(split_path.parent / "subsets" / manifest_path)
            if split_path.parent.name == "images":
                candidates.append(split_path.parent.parent / "subsets" / manifest_path)

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Subset manifest {str(manifest)!r} was not found. Searched: {searched}"
    )


def load_subset_manifest_file_names(
    manifest: str | Path,
    *,
    dataset_root: str | Path | None = None,
    split_dir: str | Path | None = None,
) -> list[str]:
    """Load ordered unique selected image basenames from a subset manifest."""
    manifest_path = resolve_subset_manifest_path(
        manifest,
        dataset_root=dataset_root,
        split_dir=split_dir,
    )
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict) or not isinstance(payload.get("samples"), list):
        raise ValueError(
            f"Subset manifest {manifest_path} must contain a top-level samples list."
        )

    file_names: list[str] = []
    seen: set[str] = set()
    for index, sample in enumerate(payload["samples"]):
        if not isinstance(sample, dict):
            raise ValueError(
                f"Subset manifest {manifest_path} sample #{index} must be an object."
            )
        raw_path = sample.get(
            "path",
            sample.get("image_path", sample.get("file_name", sample.get("filename"))),
        )
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError(
                f"Subset manifest {manifest_path} sample #{index} is missing a path."
            )
        file_name = Path(raw_path).name
        if file_name not in seen:
            file_names.append(file_name)
            seen.add(file_name)

    if not file_names:
        raise ValueError(f"Subset manifest {manifest_path} does not select any files.")
    return file_names


def filter_files_by_subset_manifest(
    files: Sequence[str],
    manifest: str | Path | None,
    *,
    dataset_root: str | Path | None = None,
    split_dir: str | Path | None = None,
    context: str = "dataset",
) -> list[str]:
    """Return files ordered by the manifest, validating all requested files exist."""
    if manifest is None:
        return list(files)

    selected = load_subset_manifest_file_names(
        manifest,
        dataset_root=dataset_root,
        split_dir=split_dir,
    )
    available = set(files)
    missing = [file_name for file_name in selected if file_name not in available]
    if missing:
        preview = ", ".join(missing[:8])
        suffix = "" if len(missing) <= 8 else f", ... ({len(missing)} total)"
        raise FileNotFoundError(
            f"Subset manifest for {context} references files not present in "
            f"{split_dir or dataset_root or '<unknown split>'}: {preview}{suffix}"
        )
    return selected


def _record_file_name(record: dict[str, Any], *, index: int, context: str) -> str:
    raw_path = record.get(
        "image_path",
        record.get("path", record.get("file_name", record.get("filename"))),
    )
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(
            f"{context} record #{index} is missing image_path/path/file_name/filename."
        )
    return Path(raw_path).name


def filter_manifest_records_by_subset_manifest(
    records: Sequence[dict[str, Any]],
    manifest: str | Path | None,
    *,
    dataset_root: str | Path | None = None,
    split_dir: str | Path | None = None,
    context: str = "manifest-backed dataset",
) -> list[dict[str, Any]]:
    """Return manifest records ordered by a subset manifest selection."""
    if manifest is None:
        return list(records)

    selected = load_subset_manifest_file_names(
        manifest,
        dataset_root=dataset_root,
        split_dir=split_dir,
    )
    by_file_name: dict[str, dict[str, Any]] = {}
    duplicates: set[str] = set()
    for index, record in enumerate(records):
        file_name = _record_file_name(record, index=index, context=context)
        if file_name in by_file_name:
            duplicates.add(file_name)
        by_file_name[file_name] = record
    if duplicates:
        preview = ", ".join(sorted(duplicates)[:8])
        raise ValueError(
            f"{context} has duplicate manifest basenames: {preview}"
        )

    missing = [file_name for file_name in selected if file_name not in by_file_name]
    if missing:
        preview = ", ".join(missing[:8])
        suffix = "" if len(missing) <= 8 else f", ... ({len(missing)} total)"
        raise FileNotFoundError(
            f"Subset manifest for {context} references files not present in "
            f"the base manifest: {preview}{suffix}"
        )
    return [by_file_name[file_name] for file_name in selected]


__all__ = [
    "filter_files_by_subset_manifest",
    "filter_manifest_records_by_subset_manifest",
    "load_subset_manifest_file_names",
    "resolve_subset_manifest_path",
]
