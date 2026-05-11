"""Tests for DomainStudio-style Stage-1 Stable Diffusion losses."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

import src.algorithms.stable_diffusion.training as sd_training
from src.algorithms.stable_diffusion.config import parse_args
from src.algorithms.stable_diffusion.data import (
    create_prior_dataloader,
    cycle_dataloader,
    find_prior_image_paths,
)
from src.algorithms.stable_diffusion.domainstudio import (
    compute_domainstudio_losses,
    haar_high_frequency,
    pairwise_kl_loss,
    predict_original_latents_from_epsilon,
)
from src.algorithms.stable_diffusion.models import ModelComponents
from src.algorithms.stable_diffusion.training import Trainer


class _TokenizerOutput:
    def __init__(self, input_ids: torch.Tensor):
        self.input_ids = input_ids


class _MockTokenizer:
    model_max_length = 8

    def __call__(self, text, **kwargs):
        texts = [text] if isinstance(text, str) else list(text)
        return _TokenizerOutput(torch.ones(len(texts), self.model_max_length, dtype=torch.long))


class _FakeScheduler:
    def __init__(self, prediction_type: str = "epsilon"):
        self.config = SimpleNamespace(
            prediction_type=prediction_type,
            num_train_timesteps=10,
        )
        self.alphas_cumprod = torch.linspace(0.95, 0.05, self.config.num_train_timesteps)

    def add_noise(self, latents, noise, timesteps):
        alpha = self.alphas_cumprod.to(latents.device, latents.dtype)[timesteps].view(-1, 1, 1, 1)
        return alpha.sqrt() * latents + (1.0 - alpha).sqrt() * noise


class _LatentDist:
    def __init__(self, latents: torch.Tensor):
        self.latents = latents

    def sample(self):
        return self.latents


class _FakeVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.decoder = torch.nn.Conv2d(4, 3, kernel_size=3, padding=1)
        self.config = SimpleNamespace(scaling_factor=1.0)

    def encode(self, pixel_values):
        return SimpleNamespace(latent_dist=_LatentDist(self.encoder(pixel_values)))

    def decode(self, latents):
        return SimpleNamespace(sample=torch.tanh(self.decoder(latents)))


class _FakeTextEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 8)

    def forward(self, input_ids, return_dict=False):
        return (self.embedding(input_ids.clamp(min=0, max=15)),)


class _FakeUNet(torch.nn.Module):
    def __init__(self, *, train_kind: str = "lora"):
        super().__init__()
        self.backbone = torch.nn.Conv2d(4, 4, kernel_size=1)
        self.lora_adapter = torch.nn.Conv2d(4, 4, kernel_size=1, bias=False)
        self.cond = torch.nn.Linear(8, 4)
        for param in self.parameters():
            param.requires_grad_(False)
        if train_kind == "lora":
            for param in self.lora_adapter.parameters():
                param.requires_grad_(True)
        elif train_kind == "unet":
            for param in self.backbone.parameters():
                param.requires_grad_(True)

    def forward(self, sample, timesteps, encoder_hidden_states, cross_attention_kwargs=None, return_dict=False):
        pooled = encoder_hidden_states.mean(dim=1)
        cond = self.cond(pooled).view(sample.shape[0], 4, 1, 1)
        out = self.backbone(sample) + self.lora_adapter(sample) + cond
        return (out,)


class _NoZeroOptimizer:
    def __init__(self, params):
        self.params = list(params)
        self.step_calls = 0

    def step(self):
        self.step_calls += 1

    def zero_grad(self):
        pass


class _FakeLRScheduler:
    def __init__(self):
        self.step_calls = 0

    def step(self):
        self.step_calls += 1

    def get_last_lr(self):
        return [0.001]


class _FakeAccelerator:
    def __init__(self):
        self.device = torch.device("cpu")
        self.sync_gradients = True
        self.num_processes = 1
        self.is_main_process = True
        self.is_local_main_process = True
        self.logged = []
        self.trackers = []

    def accumulate(self, *models):
        return nullcontext()

    def backward(self, loss):
        loss.backward()

    def clip_grad_norm_(self, params, max_norm):
        return None

    def prepare(self, *args):
        return args if len(args) != 1 else args[0]

    def init_trackers(self, *args, **kwargs):
        return None

    def register_save_state_pre_hook(self, hook):
        return None

    def register_load_state_pre_hook(self, hook):
        return None

    def main_process_first(self):
        return nullcontext()

    def log(self, payload, step=None):
        self.logged.append((payload, step))


def _domainstudio_cfg(tmp_path: Path, *, baseline_mode: str = "sd_ir_lora"):
    prior_dir = tmp_path / "priors"
    prior_dir.mkdir()
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(prior_dir / "prior.png")
    return parse_args(
        [
            "--dataset_id",
            "v18",
            "--baseline_mode",
            baseline_mode,
            "--domainstudio_enabled",
            "true",
            "--domainstudio_source_prompt",
            "a photo",
            "--domainstudio_target_prompt",
            "thermal image",
            "--domainstudio_prior_data_dir",
            str(prior_dir),
            "--resolution",
            "8",
            "--train_batch_size",
            "2",
            "--max_train_steps",
            "1",
            "--output_dir",
            str(tmp_path / "run"),
        ]
    )


def _components(train_kind: str = "lora") -> ModelComponents:
    vae = _FakeVAE()
    text_encoder = _FakeTextEncoder()
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    return ModelComponents(
        unet=_FakeUNet(train_kind=train_kind),
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=_MockTokenizer(),
        noise_scheduler=_FakeScheduler(),
        weight_dtype=torch.float32,
    )


def test_domainstudio_config_parses_yaml_keys_and_target_prompt(tmp_path: Path):
    prior_dir = tmp_path / "prior"
    prior_dir.mkdir()
    config_path = tmp_path / "sd_domainstudio.yaml"
    config_path.write_text(
        "\n".join(
            [
                "dataset_id: v18",
                f"output_dir: {tmp_path / 'run'}",
                "domainstudio_enabled: true",
                "domainstudio_source_prompt: a photo",
                "domainstudio_target_prompt: thermal image",
                f"domainstudio_prior_data_dir: {prior_dir}",
                "domainstudio_lambda_prior: 2.0",
            ]
        ),
        encoding="utf-8",
    )

    cfg = parse_args(["--config", str(config_path)])

    assert cfg.domainstudio_enabled is True
    assert cfg.domainstudio_lambda_prior == 2.0
    assert cfg.resolved_training_prompt_text() == "thermal image"


def test_domainstudio_rejects_layout_and_v_prediction(tmp_path: Path):
    with pytest.raises(ValueError, match="layout_conditioning_enabled"):
        parse_args(
            [
                "--dataset_id",
                "v18",
                "--domainstudio_enabled",
                "true",
                "--domainstudio_source_prompt",
                "a photo",
                "--layout_conditioning_enabled",
                "true",
                "--output_dir",
                str(tmp_path / "layout"),
            ]
        )

    with pytest.raises(NotImplementedError, match="epsilon"):
        parse_args(
            [
                "--dataset_id",
                "v18",
                "--domainstudio_enabled",
                "true",
                "--domainstudio_source_prompt",
                "a photo",
                "--prediction_type",
                "v_prediction",
                "--output_dir",
                str(tmp_path / "vpred"),
            ]
        )


def test_predict_original_latents_from_epsilon_returns_finite_values():
    scheduler = _FakeScheduler()
    noisy = torch.randn(2, 4, 4, 4)
    eps = torch.randn_like(noisy)
    timesteps = torch.tensor([0, 3])

    z0 = predict_original_latents_from_epsilon(noisy, timesteps, eps, scheduler)

    assert z0.shape == noisy.shape
    assert torch.isfinite(z0).all()


def test_haar_high_frequency_is_differentiable():
    x = torch.randn(2, 3, 7, 9, requires_grad=True)
    hf = haar_high_frequency(x)
    loss = hf.square().mean()
    loss.backward()

    assert hf.shape == (2, 3, 3, 4)
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_pairwise_kl_loss_is_finite_and_singleton_safe():
    target = torch.randn(3, 3, 4, 4, requires_grad=True)
    source = torch.randn(3, 3, 4, 4, requires_grad=True)

    loss = pairwise_kl_loss(target, source, temperature=1.0)
    loss.backward()

    assert torch.isfinite(loss)
    assert target.grad is not None
    assert source.grad is not None

    singleton = pairwise_kl_loss(target[:1], source[:1])
    assert singleton.item() == pytest.approx(0.0)


def test_compute_domainstudio_losses_backward_with_all_terms():
    student = torch.randn(2, 4, 4, 4, requires_grad=True)
    teacher = torch.randn(2, 4, 4, 4)
    target = torch.randn(2, 3, 8, 8, requires_grad=True)
    prior = torch.randn(2, 3, 8, 8, requires_grad=True)
    real = torch.randn(2, 3, 8, 8)

    losses = compute_domainstudio_losses(
        student_pred_prior=student,
        teacher_pred_prior=teacher,
        img_target_hat=target,
        img_prior_hat=prior,
        pixel_values=real,
    )
    total = sum(losses.values())
    total.backward()

    assert set(losses) == {"prior", "img_pairwise", "hf_pairwise", "hf_mse"}
    assert all(torch.isfinite(loss) for loss in losses.values())
    assert student.grad is not None
    assert target.grad is not None
    assert prior.grad is not None


def test_prior_dataloader_loads_npy_and_images_and_cycles(tmp_path: Path):
    prior_dir = tmp_path / "priors"
    prior_dir.mkdir()
    np.save(prior_dir / "sample.npy", np.zeros((8, 8, 3), dtype=np.uint8))
    Image.fromarray(np.ones((8, 8, 3), dtype=np.uint8) * 255).save(prior_dir / "sample.png")

    paths = find_prior_image_paths(str(prior_dir))
    dataloader = create_prior_dataloader(
        prior_image_paths=paths,
        tokenizer=_MockTokenizer(),
        source_prompt="a photo",
        resolution=8,
        center_crop=False,
        random_flip=False,
        interpolation_mode="nearest",
        batch_size=1,
    )
    iterator = cycle_dataloader(dataloader)
    batches = [next(iterator) for _ in range(3)]

    assert len(paths) == 2
    assert all(batch["pixel_values"].shape == (1, 3, 8, 8) for batch in batches)
    assert all(batch["input_ids"].shape == (1, 8) for batch in batches)


def test_trainer_setup_loads_frozen_teacher_when_enabled(tmp_path: Path, monkeypatch):
    cfg = _domainstudio_cfg(tmp_path)
    models = _components()
    train_dataloader = DataLoader(
        [{"pixel_values": torch.zeros(3, 8, 8), "input_ids": torch.ones(8, dtype=torch.long)}],
        batch_size=1,
    )
    teacher = _FakeUNet(train_kind="none")
    calls = {"teacher": 0}

    def _fake_from_pretrained(*args, **kwargs):
        calls["teacher"] += 1
        return teacher

    monkeypatch.setattr(sd_training.UNet2DConditionModel, "from_pretrained", _fake_from_pretrained)
    trainer = Trainer(
        config=cfg,
        models=models,
        train_dataloader=train_dataloader,
        normalization_mode="uint8_linear",
        adaptation_info={},
        accelerator=_FakeAccelerator(),
    )

    trainer.setup()

    assert calls["teacher"] == 1
    assert trainer.domainstudio_teacher_unet is teacher
    assert all(not param.requires_grad for param in teacher.parameters())
    assert trainer.domainstudio_prior_iter is not None


def test_train_step_disabled_path_does_not_require_domainstudio(tmp_path: Path):
    cfg = parse_args(
        [
            "--dataset_id",
            "v18",
            "--baseline_mode",
            "sd_ir_lora",
            "--resolution",
            "8",
            "--train_batch_size",
            "2",
            "--output_dir",
            str(tmp_path / "run"),
        ]
    )
    models = _components(train_kind="lora")
    trainer = Trainer(
        config=cfg,
        models=models,
        train_dataloader=[],
        normalization_mode="uint8_linear",
        adaptation_info={},
        accelerator=_FakeAccelerator(),
    )
    trainer.optimizer = _NoZeroOptimizer([param for param in models.unet.parameters() if param.requires_grad])
    trainer.lr_scheduler = _FakeLRScheduler()

    loss = trainer._train_step(
        {
            "pixel_values": torch.randn(2, 3, 8, 8),
            "input_ids": torch.ones(2, 8, dtype=torch.long),
        }
    )

    assert torch.isfinite(loss)
    assert trainer._last_loss_logs == {}


@pytest.mark.parametrize(
    ("baseline_mode", "train_kind", "expected_prefix"),
    [
        ("sd_ir_lora", "lora", "lora_adapter"),
        ("sd_ir_unet", "unet", "backbone"),
    ],
)
def test_domainstudio_train_step_gradients_follow_configured_trainable_params(
    tmp_path: Path,
    baseline_mode: str,
    train_kind: str,
    expected_prefix: str,
):
    cfg = _domainstudio_cfg(tmp_path, baseline_mode=baseline_mode)
    models = _components(train_kind=train_kind)
    teacher = _FakeUNet(train_kind="none")
    teacher.requires_grad_(False)
    trainer = Trainer(
        config=cfg,
        models=models,
        train_dataloader=[],
        normalization_mode="uint8_linear",
        adaptation_info={},
        accelerator=_FakeAccelerator(),
    )
    trainable_params = [param for param in models.unet.parameters() if param.requires_grad]
    trainer.optimizer = _NoZeroOptimizer(trainable_params)
    trainer.lr_scheduler = _FakeLRScheduler()
    trainer.domainstudio_teacher_unet = teacher
    trainer.domainstudio_prior_iter = iter(
        [
            {
                "pixel_values": torch.randn(2, 3, 8, 8),
                "input_ids": torch.ones(2, 8, dtype=torch.long),
            }
        ]
    )

    loss = trainer._train_step(
        {
            "pixel_values": torch.randn(2, 3, 8, 8),
            "input_ids": torch.ones(2, 8, dtype=torch.long),
        }
    )

    grads = {
        name: param.grad
        for name, param in models.unet.named_parameters()
        if param.grad is not None and param.grad.abs().sum() > 0
    }
    assert torch.isfinite(loss)
    assert grads
    assert all(name.startswith(expected_prefix) for name in grads)
    assert all(param.grad is None for param in teacher.parameters())
    assert "train/loss_total" in trainer._last_loss_logs
