import torch

from src.core.data.adapters import (
    canonical_batch_from_mapping,
    canonical_sample_from_mapping,
    normalize_sample,
)
from src.core.data.layout_batching import collate_layout_batch
from src.core.data.schema import CanonicalBatch


def test_image_only_tensor_sample_canonicalizes_to_pixel_values() -> None:
    image = torch.zeros(1, 4, 4)

    normalized = normalize_sample(image)
    canonical = canonical_sample_from_mapping(image)

    assert normalized["pixel_values"] is image
    assert canonical.pixel_values is image
    assert canonical.metadata == {}


def test_text_and_tokenizer_fields_are_preserved() -> None:
    sample = {
        "pixel_values": torch.zeros(1, 4, 4),
        "text": "thermal image",
        "input_ids": torch.tensor([1, 2, 3]),
        "attention_mask": torch.tensor([1, 1, 1]),
    }

    canonical = canonical_sample_from_mapping(sample)

    assert canonical.text == "thermal image"
    assert torch.equal(canonical.input_ids, sample["input_ids"])
    assert torch.equal(canonical.attention_mask, sample["attention_mask"])


def test_layout_collate_output_conforms_to_canonical_batch() -> None:
    samples = [
        {
            "pixel_values": torch.zeros(1, 8, 8),
            "boxes_xyxy": torch.tensor([[1.0, 2.0, 4.0, 6.0]]),
            "labels": torch.tensor([2]),
            "image_id": "img-1",
            "file_name": "img-1.npy",
            "n_objects": 1,
            "label_names": ["car"],
        },
        {
            "pixel_values": torch.ones(1, 8, 8),
            "boxes_xyxy": torch.zeros(0, 4),
            "labels": torch.zeros(0, dtype=torch.long),
            "image_id": "img-2",
            "file_name": "img-2.npy",
            "n_objects": 0,
            "label_names": [],
        },
    ]

    batch = collate_layout_batch(samples)
    canonical = canonical_batch_from_mapping(batch)

    assert isinstance(canonical, CanonicalBatch)
    assert canonical.pixel_values.shape == (2, 1, 8, 8)
    assert canonical.boxes_xyxy.shape == (2, 1, 4)
    assert canonical.boxes_xyxy_norm.max() <= 1.0
    assert canonical.object_mask.tolist() == [[True], [False]]


def test_metadata_aliases_are_copied_without_removing_top_level_keys() -> None:
    sample = {
        "pixel_values": torch.zeros(1, 4, 4),
        "image_id": "abc",
        "file_name": "abc.npy",
        "prompt_text": "a prompt",
        "caption_text": "a caption",
        "metadata": {"source": "unit"},
    }

    normalized = normalize_sample(sample)

    assert normalized["image_id"] == "abc"
    assert normalized["file_name"] == "abc.npy"
    assert normalized["prompt_text"] == "a prompt"
    assert normalized["metadata"]["source"] == "unit"
    assert normalized["metadata"]["image_id"] == "abc"
    assert normalized["metadata"]["file_name"] == "abc.npy"
    assert normalized["metadata"]["prompt_text"] == "a prompt"
    assert normalized["metadata"]["caption_text"] == "a caption"
    assert normalized["text"] == "a prompt"
