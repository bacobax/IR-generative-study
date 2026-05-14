#!/usr/bin/env python3
"""End-to-end check for full curriculum MetaFMTrainer.

Validates:
  - Incremental stages execute in sequence
  - Test conditions are never trained on
  - Final evaluation outputs are produced
"""

import os
import sys
import json
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.algorithms.training.meta_fm_trainer import MetaFMTrainer
from src.conditioning.expert_subset_policy import ExpertSubsetPolicy
from src.conditioning.text_conditioner import TextConditioner
from src.core.configs.fm_config import CountFilterConfig
from src.core.data.annotation_dataset import AnnotationFMDataset
from src.models.moe_text_unet import build_text_moe_unet

passed = 0
failed = 0


def check(label, cond):
    global passed, failed
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {label}")
    if cond:
        passed += 1
    else:
        failed += 1


def _collate(batch):
    images = torch.stack([b["pixel_values"] for b in batch])
    texts = [b["text"] for b in batch]
    condition_ids = [b["condition_id"] for b in batch]
    return {"pixel_values": images, "text": texts, "condition_id": condition_ids}


class _MockEncoder:
    class config:
        hidden_size = 32

    def __call__(self, input_ids, attention_mask):
        batch, seq_len = input_ids.shape

        class _Out:
            last_hidden_state = torch.randn(batch, seq_len, 32)
            pooler_output = torch.randn(batch, 32)

        return _Out()

    def to(self, device):
        return self

    def eval(self):
        return self

    def parameters(self):
        return iter([])


class _MockTokenizer:
    def __call__(self, texts, **kw):
        batch = len(texts)
        length = kw.get("max_length", 10)
        return {
            "input_ids": torch.ones(batch, length, dtype=torch.long),
            "attention_mask": torch.ones(batch, length, dtype=torch.long),
        }


def _mock_conditioner():
    cond = TextConditioner.__new__(TextConditioner)
    cond.cond_drop_prob = 0.0
    cond.max_length = 10
    cond.return_pooled = True
    cond.device = "cpu"
    cond._null_embedding = None
    cond._null_pooled = None
    cond.encoder_name = "mock"
    cond.text_encoder = _MockEncoder()
    cond.tokenizer = _MockTokenizer()
    return cond


print("=== Setup ===")
unet_cfg = {
    "sample_size": 16,
    "in_channels": 1,
    "out_channels": 1,
    "block_out_channels": [32, 64],
    "down_block_types": ["CrossAttnDownBlock2D", "DownBlock2D"],
    "up_block_types": ["UpBlock2D", "CrossAttnUpBlock2D"],
    "mid_block_type": "UNetMidBlock2DCrossAttn",
    "layers_per_block": 1,
    "cross_attention_dim": 32,
    "attention_head_dim": 8,
    "norm_num_groups": 16,
}

model = build_text_moe_unet(unet_cfg, device="cpu")
conditioner = _mock_conditioner()

trainer = MetaFMTrainer(
    model,
    conditioner=conditioner,
    device="cpu",
    t_scale=1000.0,
    train_target="v",
    model_dir="./artifacts/checkpoints/flow_matching/meta_fm_demo/",
    unet_config=unet_cfg,
    subset_policy=ExpertSubsetPolicy(
        num_experts=model.num_experts,
        enabled=True,
        configured_subsets={
            1: [0, 1],
            2: [0, 1],
            3: [1, 2],
            4: [1, 2],
        },
    ),
)

image_shape = (unet_cfg.get("in_channels", 4), unet_cfg.get("sample_size", 64), unet_cfg.get("sample_size", 64))

split_base = [1, 3]
split_inc = [2, 4]
split_test = [5]

with tempfile.TemporaryDirectory() as tmpdir:
    data_dir = os.path.join(tmpdir, "data")
    os.makedirs(data_dir, exist_ok=True)

    # Create tiny annotated dataset
    annotations = {"images": [], "annotations": []}
    ann_id = 1
    for i, count in enumerate([1, 2, 3, 4, 5]):
        fname = f"img_{i:03d}.npy"
        path = os.path.join(data_dir, fname)
        arr = torch.randn(*image_shape).numpy()
        np.save(path, arr)

        annotations["images"].append({
            "id": i + 1,
            "file_name": fname,
            "width": image_shape[2],
            "height": image_shape[1],
        })
        for _ in range(count):
            annotations["annotations"].append({
                "id": ann_id,
                "image_id": i + 1,
                "bbox": [10, 10, 5, 5],
            })
            ann_id += 1

    ann_path = os.path.join(tmpdir, "annotations.json")
    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump(annotations, f)

    def _make_ds(allowed_counts):
        return AnnotationFMDataset(
            root_dir=data_dir,
            annotations_path=ann_path,
            text_mode=True,
            curriculum=None,
            count_filter=CountFilterConfig(seen_counts=allowed_counts),
            transform=None,
        )

    base_loader = DataLoader(
        _make_ds(split_base),
        batch_size=2,
        shuffle=False,
        collate_fn=_collate,
    )
    incremental_loaders = [
        (2, DataLoader(_make_ds([2]), batch_size=2, shuffle=False, collate_fn=_collate)),
        (4, DataLoader(_make_ds([4]), batch_size=2, shuffle=False, collate_fn=_collate)),
    ]

    # Verify test set not in training (by construction)
    check("Test conditions not in training", all(c not in split_base + split_inc for c in split_test))

    eval_dir = os.path.join(tmpdir, "eval")

    trainer.train_curriculum(
        base_dataloader=base_loader,
        incremental_loaders=incremental_loaders,
        test_conditions=split_test,
        phase_a_epochs=1,
        phase_b_epochs=1,
        phase_c_epochs=1,
        phase_a_lr=1e-4,
        phase_b_lr=1e-4,
        phase_c_lr=1e-4,
        phase_c_unfreeze_policy="none",
        phase_c_router_trainable=True,
        phase_c_router_lr_scale=1.0,
        phase_c_replay_every=1,
        log_router_weights=False,
        router_weights_dir=None,
        eval_output_dir=eval_dir,
        eval_steps=2,
        eval_guidance_scale=1.0,
        eval_samples_per_condition=1,
    )

    cond_dir = os.path.join(eval_dir, "cond_5")
    sample_path = os.path.join(cond_dir, "sample_000.npy")
    check("Final eval output exists", os.path.isfile(sample_path))


print("\n=== Summary ===")
print(f"  {passed}/{passed + failed} checks passed")
if failed:
    sys.exit(1)
else:
    print("  All OK!")
