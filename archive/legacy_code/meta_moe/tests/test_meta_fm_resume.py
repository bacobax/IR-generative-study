from __future__ import annotations

import tempfile

import torch


class _MockEncoder:
    class config:
        hidden_size = 32

    def __call__(self, input_ids, attention_mask):
        class _Out:
            last_hidden_state = torch.randn(
                input_ids.shape[0], input_ids.shape[1], 32,
            )

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
    from src.conditioning.text_conditioner import TextConditioner

    cond = TextConditioner.__new__(TextConditioner)
    cond.cond_drop_prob = 0.0
    cond.max_length = 10
    cond.device = "cpu"
    cond._null_embedding = None
    cond.encoder_name = "mock"
    cond.text_encoder = _MockEncoder()
    cond.tokenizer = _MockTokenizer()
    return cond


def _small_moe_unet():
    from src.models.moe_text_unet import build_text_moe_unet

    return build_text_moe_unet(
        {
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
        },
        device="cpu",
    )


def _make_trainer(model_dir: str, subset_policy=None):
    from src.algorithms.training.meta_fm_trainer import MetaFMTrainer

    return MetaFMTrainer(
        _small_moe_unet(),
        conditioner=_mock_conditioner(),
        device="cpu",
        t_scale=1.0,
        train_target="v",
        model_dir=model_dir,
        subset_policy=subset_policy,
    )


def test_meta_config_resume_and_phase_controls_load():
    from src.cli.train_meta_fm import build_parser
    from src.core.configs.config_loader import merge_config_and_cli
    from src.core.configs.meta_fm_config import MetaFMTrainConfig

    parser = build_parser()
    args = parser.parse_args(
        ["--config", "configs/fm/train/presets/meta_curriculum_cfg_resume_latest.yaml"],
    )
    cfg = merge_config_and_cli(MetaFMTrainConfig, args.config, parser, args)

    assert cfg.output.resume == "latest"
    assert cfg.checkpoint.enabled is True
    assert cfg.checkpoint.save_every_epochs == 1
    assert cfg.phase_b.router_trainable is True
    assert cfg.phase_b.moe_trainable is False
    assert cfg.phase_c.unet_trainable is True
    assert cfg.phase_c.unfreeze_unet_policy == "mid"
    assert cfg.subset_policy.enabled is False
    assert cfg.subset_policy.unseen_policy == "router_topk"
    assert cfg.subset_policy.empty_fallback == "top1"


def test_meta_phase_trainability_is_yaml_driven():
    from src.core.configs.meta_fm_config import MetaPhaseConfig

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = _make_trainer(tmpdir)
        summary = trainer._apply_phase_trainability(
            MetaPhaseConfig(
                mlp_trainable=False,
                router_trainable=True,
                moe_trainable=False,
                unet_trainable=True,
                unfreeze_unet_policy="mid",
                lambda_corr=0.05,
            ),
        )

        assert summary["mlp"] is False
        assert summary["router"] is True
        assert summary["moe"] is False
        assert summary["unet"] is True
        assert torch.isclose(trainer._moe_unet().lambda_corr, torch.tensor(0.05))


def test_meta_checkpoint_roundtrip_and_resume_cursor():
    from src.conditioning.expert_subset_policy import ExpertSubsetPolicy

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = _make_trainer(
            tmpdir,
            subset_policy=ExpertSubsetPolicy(
                num_experts=4,
                enabled=True,
                configured_subsets={1: [0, 1], 7: [1, 2]},
                unseen_policy="router_threshold",
                threshold=0.25,
            ),
        )
        trainer._global_step = 17
        trainer._apply_phase_trainability(
            type(
                "PhaseCfg",
                (),
                {
                    "mlp_trainable": True,
                    "router_trainable": True,
                    "moe_trainable": False,
                    "unet_trainable": False,
                    "unfreeze_unet_policy": "none",
                },
            )(),
        )
        optimizer = trainer._build_optimizer(1e-4)
        ckpt_dir = trainer._checkpoint_dir()
        ckpt_path = trainer._save_training_checkpoint(
            checkpoint_dir=ckpt_dir,
            latest_filename="meta_fm_latest.pt",
            save_latest=True,
            phase="phase_b",
            epoch_in_phase=2,
            incremental_index=1,
            condition=7,
            optimizer=optimizer,
            scheduler_state=None,
            lr=1e-4,
            router_lr_scale=None,
            replay_every=0,
        )

        resumed = _make_trainer(tmpdir)
        payload = resumed._load_training_checkpoint(
            "latest",
            checkpoint_dir=ckpt_dir,
            latest_filename="meta_fm_latest.pt",
        )
        assert payload is not None
        assert payload["_resolved_path"].endswith("meta_fm_latest.pt")
        assert payload["phase"] == "phase_b"
        assert payload["epoch_in_phase"] == 2
        assert payload["incremental_index"] == 1
        assert payload["condition"] == 7
        assert resumed._global_step == 17
        assert ckpt_path.endswith("meta_fm_phase_b_cond_7_epoch_0002.pt")
        assert resumed.subset_policy.enabled is True
        assert resumed.subset_policy.configured_subsets[1] == (0, 1)
        assert resumed.subset_policy.configured_subsets[7] == (1, 2)
        assert resumed.subset_policy.unseen_policy == "router_threshold"
        assert resumed.subset_policy.threshold == 0.25

        cursor = resumed._advance_resume_cursor(
            phase="phase_b",
            epoch_in_phase=2,
            incremental_index=1,
            incremental_loaders=[(3, object()), (7, object()), (9, object())],
            phase_epochs={"phase_a": 4, "phase_b": 2, "phase_c": 3},
        )
        assert cursor["phase"] == "phase_c"
        assert cursor["epoch_in_phase"] == 0
        assert cursor["incremental_index"] == 1
        assert cursor["condition"] == 7
