"""Tests for the text FM sampling CLI helpers."""

from __future__ import annotations

import json

import pytest

from src.cli.sample_text_fm import load_metadata_prompts


def test_load_metadata_prompts_reads_text_entries(tmp_path):
    metadata_path = tmp_path / "metadata.jsonl"
    rows = [
        {"file_name": "a.npy", "text": "first prompt"},
        {"file_name": "b.npy", "text": "second prompt"},
        {"file_name": "c.npy", "text": "third prompt"},
    ]
    metadata_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    prompts = load_metadata_prompts(str(metadata_path), max_samples=2)

    assert prompts == ["first prompt", "second prompt"]


def test_load_metadata_prompts_requires_non_empty_text(tmp_path):
    metadata_path = tmp_path / "metadata.jsonl"
    metadata_path.write_text(
        json.dumps({"file_name": "a.npy", "text": ""}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Missing non-empty 'text' field"):
        load_metadata_prompts(str(metadata_path), max_samples=1)
