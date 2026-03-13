"""Condition split utilities for curriculum training.

Provides helpers to extract condition IDs from text prompts and
build explicit base/incremental/test splits.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple


_PERSON_RE = re.compile(r"(\d+)\s*(person|persons|people)", re.IGNORECASE)


@dataclass
class ConditionSplit:
    """Explicit condition split for curriculum training."""

    base: List[int] = field(default_factory=list)
    incremental: List[int] = field(default_factory=list)
    test: List[int] = field(default_factory=list)

    def validate(self) -> None:
        base_set = set(self.base)
        inc_set = set(self.incremental)
        test_set = set(self.test)
        if base_set & inc_set or base_set & test_set or inc_set & test_set:
            raise ValueError("Condition split sets must be disjoint")

    def as_dict(self) -> Dict[str, List[int]]:
        return {
            "base": list(self.base),
            "incremental": list(self.incremental),
            "test": list(self.test),
        }


def extract_person_count(text: str) -> Optional[int]:
    """Extract the first integer count from a text prompt.

    Returns None when no count-like pattern is found.
    """
    if text is None:
        return None
    match = _PERSON_RE.search(text)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def iter_text_entries(dataset) -> Iterable[Tuple[int, str]]:
    """Yield (index, text) pairs without loading image tensors.

    Uses the dataset's JSON map or .txt files when available.
    """
    files = getattr(dataset, "files", None)
    if files is None:
        # Fallback: access via __getitem__ (may load images)
        for i in range(len(dataset)):
            item = dataset[i]
            text = item.get("text") if isinstance(item, dict) else None
            yield i, text
        return

    text_map = getattr(dataset, "_text_map", {})
    root_dir = getattr(dataset, "root_dir", None)
    fallback = getattr(dataset, "fallback_text", None)

    for idx, fname in enumerate(files):
        stem = Path(fname).stem
        text = text_map.get(stem)
        if text is None and root_dir is not None:
            txt_path = os.path.join(root_dir, f"{stem}.txt")
            if os.path.isfile(txt_path):
                with open(txt_path, "r") as f:
                    text = f.read().strip()
        if text is None:
            text = fallback
        yield idx, text


def build_condition_index(
    dataset,
    *,
    extractor: Callable[[str], Optional[int]] = extract_person_count,
) -> Dict[int, List[int]]:
    """Map condition IDs to dataset indices."""
    cond_index: Dict[int, List[int]] = {}
    for idx, text in iter_text_entries(dataset):
        cond = extractor(text)
        if cond is None:
            continue
        cond_index.setdefault(cond, []).append(idx)
    return cond_index


def indices_for_conditions(
    cond_index: Dict[int, List[int]],
    conditions: List[int],
) -> List[int]:
    """Flatten indices for a list of conditions."""
    indices: List[int] = []
    for cond in conditions:
        indices.extend(cond_index.get(cond, []))
    return indices


def save_split(path: str, split: ConditionSplit) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(split.as_dict(), f, indent=2, sort_keys=True)
