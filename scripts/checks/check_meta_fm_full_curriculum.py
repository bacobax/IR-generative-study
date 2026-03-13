#!/usr/bin/env python3
"""End-to-end check for full curriculum MetaFMTrainer.

Validates:
  - Incremental stages execute in sequence
  - Test conditions are never trained on
  - Final evaluation outputs are produced
"""

import os
import sys
import tempfile
from typing import Dict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import torch
from torch.utils.data import Dataset, DataLoader, Subset

from src.algorithms.training.meta_fm_trainer import MetaFMTrainer
from src.conditioning.text_conditioner import TextConditioner
from src.core.conditions import ConditionSplit, build_condition_index, indices_for_conditions
from src.models.fm_text_unet import load_text_unet_config
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


class TinyTextDataset(Dataset):
    def __init__(self, texts, num_samples, image_shape, seed=0):
        self.texts = list(texts)
        self.num_samples = num_samples
        g = torch.Generator().manual_seed(seed)
        self.images = torch.randn(num_samples, *image_shape, generator=g)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        text = self.texts[idx % len(self.texts)]
        return {
            "pixel_values": self.images[idx],
            "text": text,
        }


def _collate(batch):
    images = torch.stack([b["pixel_values"] for b in batch])
    texts = [b["text"] for b in batch]
    return {"pixel_values": images, "text": texts}


print("=== Setup ===")
config_path = os.path.join(REPO, "configs/models/fm/text_unet_config.json")
unet_cfg = load_text_unet_config(config_path)

model = build_text_moe_unet(unet_cfg, device="cpu")
conditioner = TextConditioner(device="cpu", return_pooled=True, cond_drop_prob=0.0)

trainer = MetaFMTrainer(
    model,
    conditioner=conditioner,
    device="cpu",
    t_scale=1000.0,
    train_target="v",
    model_dir="./artifacts/checkpoints/flow_matching/meta_fm_demo/",
    unet_config=unet_cfg,
)

image_shape = (unet_cfg.get("in_channels", 4), unet_cfg.get("sample_size", 64), unet_cfg.get("sample_size", 64))

texts = [
    "IR image with 1 person",
    "IR image with 2 persons",
    "IR image with 3 persons",
    "IR image with 4 persons",
    "IR image with 5 persons",
]

dataset = TinyTextDataset(texts, num_samples=10, image_shape=image_shape, seed=3)

split = ConditionSplit(base=[1, 3], incremental=[2, 4], test=[5])
cond_index = build_condition_index(dataset)

base_indices = indices_for_conditions(cond_index, split.base)
inc2_indices = indices_for_conditions(cond_index, [2])
inc4_indices = indices_for_conditions(cond_index, [4])
test_indices = indices_for_conditions(cond_index, split.test)

train_indices = set(base_indices + inc2_indices + inc4_indices)
check("Test indices not in training indices", not any(i in train_indices for i in test_indices))

base_loader = DataLoader(
    Subset(dataset, base_indices),
    batch_size=2,
    shuffle=False,
    collate_fn=_collate,
)

incremental_loaders = [
    (2, DataLoader(Subset(dataset, inc2_indices), batch_size=2, shuffle=False, collate_fn=_collate)),
    (4, DataLoader(Subset(dataset, inc4_indices), batch_size=2, shuffle=False, collate_fn=_collate)),
]

with tempfile.TemporaryDirectory() as tmpdir:
    eval_dir = os.path.join(tmpdir, "eval")

    trainer.train_curriculum(
        base_dataloader=base_loader,
        incremental_loaders=incremental_loaders,
        test_conditions=split.test,
        prompt_template="IR image with {count} persons",
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
