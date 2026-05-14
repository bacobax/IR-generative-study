#!/usr/bin/env python3
"""Smoke-check: routed MOE Text UNet integration.

Checks:
  A. Router weights vary across prompts
  B. Same prompt yields same weights in eval mode
  C. MOE UNet forward works with pooled embeddings
  D. One mini training step works
  E. Gradients reach router and adapter parameters
"""

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import torch

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


def nonzero_grad(params):
    return any(
        p.grad is not None and torch.any(p.grad != 0)
        for p in params
    )


print("=== A. Router weight variation ===")
conditioner = TextConditioner(device="cpu", return_pooled=True, cond_drop_prob=0.0)
conditioner.text_encoder.eval()

prompts = [
    "IR image with 1 person",
    "IR image with 3 persons",
    "IR image with 5 persons",
]

with torch.no_grad():
    _, pooled = conditioner.encode_text_with_pooler(prompts, device="cpu")

config_path = os.path.join(REPO, "configs/models/fm/text_unet_config.json")
unet_cfg = load_text_unet_config(config_path)
model = build_text_moe_unet(unet_cfg, device="cpu")
model.eval()

with torch.no_grad():
    w = model.compute_router_weights(pooled)

check("router weights batch size matches prompts", w.shape[0] == len(prompts))
check("router weights K matches num_experts", w.shape[1] == model.num_experts)
check("different prompts -> different weights", not torch.allclose(w[0], w[1], atol=1e-4))
check("different prompts -> different weights", not torch.allclose(w[1], w[2], atol=1e-4))


print("\n=== B. Deterministic eval weights ===")
with torch.no_grad():
    w1 = model.compute_router_weights(pooled[:1])
    w2 = model.compute_router_weights(pooled[:1])
check("same prompt -> same weights in eval mode", torch.allclose(w1, w2))


print("\n=== C. Forward pass with pooled embeddings ===")
batch_size = 2
in_channels = unet_cfg.get("in_channels", 4)
sample_size = unet_cfg.get("sample_size", 64)
seq_len = 77
cross_dim = unet_cfg.get("cross_attention_dim", 768)

x = torch.randn(batch_size, in_channels, sample_size, sample_size)
t = torch.tensor([500.0, 250.0])
encoder_hidden_states = torch.randn(batch_size, seq_len, cross_dim)
pooled_for_forward = pooled[:batch_size]

with torch.no_grad():
    out = model(
        x,
        t,
        encoder_hidden_states=encoder_hidden_states,
        pooled_text_embeds=pooled_for_forward,
    )

sample = out.sample if hasattr(out, "sample") else out
check("output shape matches input shape", sample.shape == x.shape)
check("output has no NaNs", not torch.isnan(sample).any().item())


print("\n=== D. One mini training step ===")
model.train()
optim = torch.optim.Adam(model.parameters(), lr=1e-4)

x = torch.randn(batch_size, in_channels, sample_size, sample_size)
t = torch.tensor([600.0, 100.0])

batch = {"text": ["IR image with 2 persons", "IR image with 4 persons"]}
cond_kwargs = conditioner.prepare_for_training(batch, device="cpu")

out = model(
    x,
    t,
    **cond_kwargs,
)

sample = out.sample if hasattr(out, "sample") else out
loss = sample.mean()
optim.zero_grad(set_to_none=True)
loss.backward()
optim.step()

check("mini training step ran", True)


print("\n=== E. Gradient flow ===")
router_grads = nonzero_grad(model.router.parameters())
adapter_grads = nonzero_grad(model.mid_adapter.parameters())
check("router parameters receive gradients", router_grads)
check("adapter parameters receive gradients", adapter_grads)


print("\n=== Summary ===")
print(f"  {passed}/{passed + failed} checks passed")
if failed:
    sys.exit(1)
else:
    print("  All OK!")
