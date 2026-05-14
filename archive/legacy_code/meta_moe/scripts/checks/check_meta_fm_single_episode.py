#!/usr/bin/env python3
"""Check script for single-episode MetaFMTrainer.

Verifies:
  - Phase A runs
  - Phase B updates only router
  - Phase C updates adapters (and optional router)
  - Loss on new condition changes after Phase B
"""

import os
import sys
from typing import Dict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import torch
from torch.utils.data import Dataset, DataLoader

from src.algorithms.training.meta_fm_trainer import MetaFMTrainer
from src.conditioning.text_conditioner import TextConditioner
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


def snapshot_params(module) -> Dict[str, torch.Tensor]:
    return {name: p.detach().clone() for name, p in module.named_parameters()}


def params_changed(before: Dict[str, torch.Tensor], module) -> bool:
    for name, p in module.named_parameters():
        if name not in before:
            continue
        if not torch.allclose(before[name], p.detach(), atol=1e-8):
            return True
    return False


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
    model_dir="./artifacts/checkpoints/flow_matching/meta_fm/",
    unet_config=unet_cfg,
)

image_shape = (unet_cfg.get("in_channels", 4), unet_cfg.get("sample_size", 64), unet_cfg.get("sample_size", 64))

base_dataset = TinyTextDataset(
    texts=["IR image with 1 person", "IR image with 2 persons"],
    num_samples=4,
    image_shape=image_shape,
    seed=1,
)
new_dataset = TinyTextDataset(
    texts=["IR image with 3 persons"],
    num_samples=4,
    image_shape=image_shape,
    seed=2,
)

base_loader = DataLoader(base_dataset, batch_size=2, shuffle=False)
new_loader = DataLoader(new_dataset, batch_size=2, shuffle=False)


print("\n=== Phase A: base training ===")
trainer._set_trainable(trainer.unet, True)
trainer._train_phase(
    base_loader,
    epochs=1,
    lr=1e-4,
    phase_name="Phase A (base)",
)
check("Phase A ran", True)


print("\n=== Phase B: router-only ===")
router_before = snapshot_params(trainer._moe_unet().router)
adapter_before = snapshot_params(trainer._moe_unet().mid_adapter)
unet_before = snapshot_params(trainer._moe_unet().unet)

# Loss on new condition before Phase B
batch = next(iter(new_loader))
loss_before = trainer._loss_from_batch(batch).item()

trainer._freeze_all()
trainer._set_router_trainable(True)
trainer._train_phase(
    new_loader,
    epochs=1,
    lr=1e-4,
    phase_name="Phase B (router-only)",
)

loss_after = trainer._loss_from_batch(batch).item()

check("Router params changed in Phase B", params_changed(router_before, trainer._moe_unet().router))
check("Adapter params unchanged in Phase B", not params_changed(adapter_before, trainer._moe_unet().mid_adapter))
check("UNet params unchanged in Phase B", not params_changed(unet_before, trainer._moe_unet().unet))
check("Loss on new condition changed", abs(loss_after - loss_before) > 1e-8)


print("\n=== Phase C: adapters + replay ===")
router_before_c = snapshot_params(trainer._moe_unet().router)
adapter_before_c = snapshot_params(trainer._moe_unet().mid_adapter)
unet_before_c = snapshot_params(trainer._moe_unet().unet)

trainer._freeze_all()
trainer._set_adapter_trainable(True)
trainer._set_unet_parts_trainable("none")
trainer._set_router_trainable(True)
trainer._train_phase(
    new_loader,
    epochs=1,
    lr=1e-4,
    phase_name="Phase C (refine+replay)",
    replay_dataloader=base_loader,
    replay_every=1,
    router_lr_scale=1.0,
)

check("Adapter params changed in Phase C", params_changed(adapter_before_c, trainer._moe_unet().mid_adapter))
check("Router params changed in Phase C", params_changed(router_before_c, trainer._moe_unet().router))
check("UNet params unchanged in Phase C", not params_changed(unet_before_c, trainer._moe_unet().unet))


print("\n=== Summary ===")
print(f"  {passed}/{passed + failed} checks passed")
if failed:
    sys.exit(1)
else:
    print("  All OK!")
