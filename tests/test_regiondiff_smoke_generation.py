"""Tests for RegionDiff smoke synthetic export helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.algorithms.inference import regiondiff_smoke_generation as production_generation
from src.algorithms.inference.regiondiff_smoke_generation import (
    compute_metric_summary_from_features,
    export_generated_candidate_dataset,
    generate_production_synthetic_datasets,
    render_layout_overlay_previews,
    render_sanity_check_images,
    write_filtered_annotations_from_audit,
)
from src.algorithms.training.yolo_experiment_b import (
    YOLOBox,
    YOLOTrainSample,
    prepare_experiment_b_dataset,
    validate_experiment_b_config,
)
from src.core.configs.yolo_experiment_config import YOLOExperimentConfig


def _write_png(path: Path, value: int = 32) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((8, 8), value, dtype=np.uint8), mode="L").convert("RGB").save(path)


def _make_yolo_dataset(tmp_path: Path) -> Path:
    import yaml

    root = tmp_path / "yolo"
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    for idx in range(2):
        _write_png(root / "images" / "train" / f"real{idx}.png", value=40 + idx)
        (root / "labels" / "train" / f"real{idx}.txt").write_text(
            "0 0.5 0.5 0.25 0.25\n",
            encoding="utf-8",
        )
    _write_png(root / "images" / "val" / "val.png")
    _write_png(root / "images" / "test" / "test.png")
    yaml_path = root / "full_train.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": str(root / "images" / "train"),
                "val": str(root / "images" / "val"),
                "test": str(root / "images" / "test"),
                "names": {0: "person", 1: "car"},
                "nc": 2,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return yaml_path


def test_export_generated_candidate_dataset_writes_experiment_b_shape(tmp_path: Path) -> None:
    image_path = tmp_path / "real.png"
    label_path = tmp_path / "real.txt"
    image_path.write_bytes(b"fake")
    label_path.write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
    sample = YOLOTrainSample(
        index=3,
        image_path=image_path,
        label_path=label_path,
        boxes=[YOLOBox(0, 0.5, 0.5, 0.25, 0.25)],
    )
    output_dir = tmp_path / "generated"

    export_generated_candidate_dataset(
        output_dir=output_dir,
        source_samples=[sample],
        generated_arrays=[np.zeros((8, 8), dtype=np.float32)],
        dataset_payload={"names": {0: "person"}, "_yaml_path": "dataset.yaml"},
        generator_kind="regiondiff_test",
    )

    assert (output_dir / "images" / "sample_000001.npy").is_file()
    assert (output_dir / "previews" / "sample_000001.png").is_file()
    assert (output_dir / "annotations.json").is_file()
    assert (output_dir / "annotations_unfiltered.json").is_file()
    assert (output_dir / "metadata" / "provenance.jsonl").is_file()
    summary = json.loads((output_dir / "metadata" / "summary.json").read_text(encoding="utf-8"))
    assert summary["generator_kind"] == "regiondiff_test"
    assert summary["n_generated_samples"] == 1

    overlay_paths = render_layout_overlay_previews(dataset_dir=output_dir, max_images=1)
    assert len(overlay_paths) == 1
    assert (output_dir / "layout_overlays" / "sample_000001.png").is_file()
    overlay_summary = json.loads((output_dir / "layout_overlays" / "summary.json").read_text(encoding="utf-8"))
    assert overlay_summary["n_images"] == 1


def test_production_generation_routes_each_configured_generator_to_own_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset_yaml = _make_yolo_dataset(tmp_path)
    output_root = tmp_path / "generated"

    def _fake_backend(**kwargs):
        samples = kwargs["source_samples"]
        seed = int(kwargs["seed"])
        return [np.full((8, 8), seed + idx, dtype=np.float32) for idx, _sample in enumerate(samples)]

    monkeypatch.setitem(production_generation.GENERATOR_BACKENDS, "fake_a", _fake_backend)
    monkeypatch.setitem(production_generation.GENERATOR_BACKENDS, "fake_b", _fake_backend)
    config = {
        "yolo_dataset_yaml": str(dataset_yaml),
        "output_root": str(output_root),
        "seed": 11,
        "filter": {"enabled": False},
        "metrics": {"enabled": False},
        "generators": [
            {"name": "first", "backend": "fake_a"},
            {"name": "second", "backend": "fake_b", "seed_offset": 5},
        ],
    }

    summary = generate_production_synthetic_datasets(config=config, skip_filter=True, skip_metrics=True)

    assert summary["n_source_images"] == 2
    assert (output_root / "first" / "images" / "sample_000001.npy").is_file()
    assert (output_root / "second" / "images" / "sample_000002.npy").is_file()
    assert (output_root / "first" / "layout_overlays" / "sample_000001.png").is_file()
    assert (output_root / "first" / "filtered_layout_overlays" / "sample_000001.png").is_file()
    assert (output_root / "first" / "annotations_unfiltered.json").is_file()
    assert (output_root / "summary.json").is_file()


def test_production_generation_uses_streaming_backend_when_available(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset_yaml = _make_yolo_dataset(tmp_path)
    output_root = tmp_path / "generated"
    observed_counts: list[int] = []

    def _fake_streaming_backend(**kwargs):
        output_dir = kwargs["output_dir"]
        samples = kwargs["source_samples"]
        production_generation.initialize_generated_candidate_dataset(
            output_dir=output_dir,
            source_samples=samples,
            dataset_payload=kwargs["dataset_payload"],
            generator_kind="stream_fake",
            generator_config=kwargs["generator_cfg"],
            image_size=8,
        )
        for image_id, sample in enumerate(samples, start=1):
            production_generation._save_generated_array(
                output_dir,
                image_id=image_id,
                array=np.full((8, 8), int(sample.index) + 10, dtype=np.float32),
            )
            observed_counts.append(len(list((Path(output_dir) / "images").glob("*.npy"))))
        return len(samples)

    monkeypatch.setitem(production_generation.GENERATOR_BACKENDS, "stream_fake", lambda **_kwargs: pytest.fail("array backend should not run"))
    monkeypatch.setitem(production_generation.STREAMING_GENERATOR_BACKENDS, "stream_fake", _fake_streaming_backend)
    config = {
        "yolo_dataset_yaml": str(dataset_yaml),
        "output_root": str(output_root),
        "filter": {"enabled": False},
        "metrics": {"enabled": False},
        "generators": [{"name": "streamed", "backend": "stream_fake"}],
    }

    generate_production_synthetic_datasets(config=config, skip_filter=True, skip_metrics=True)

    assert observed_counts == [1, 2]
    assert (output_root / "streamed" / "images" / "sample_000001.npy").is_file()
    assert (output_root / "streamed" / "previews" / "sample_000002.png").is_file()


def test_streaming_generation_resume_skips_existing_images(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset_yaml = _make_yolo_dataset(tmp_path)
    output_root = tmp_path / "generated"
    output_dir = output_root / "streamed"
    (output_dir / "images").mkdir(parents=True)
    (output_dir / "previews").mkdir(parents=True)
    np.save(output_dir / "images" / "sample_000001.npy", np.full((8, 8), 99, dtype=np.float32))

    saved_ids: list[int] = []

    def _fake_streaming_backend(**kwargs):
        output_dir_arg = kwargs["output_dir"]
        samples = kwargs["source_samples"]
        image_ids = kwargs.get("image_ids") or list(range(1, len(samples) + 1))
        production_generation.initialize_generated_candidate_dataset(
            output_dir=output_dir_arg,
            source_samples=samples,
            dataset_payload=kwargs["dataset_payload"],
            generator_kind="stream_fake",
            generator_config=kwargs["generator_cfg"],
            image_size=8,
        )
        count = 0
        for image_id, sample in zip(image_ids, samples):
            if kwargs.get("resume") and production_generation._generated_sample_exists(output_dir_arg, image_id=image_id):
                count += 1
                continue
            production_generation._save_generated_array(
                output_dir_arg,
                image_id=image_id,
                array=np.full((8, 8), int(sample.index) + 10, dtype=np.float32),
            )
            saved_ids.append(int(image_id))
            count += 1
        return count

    monkeypatch.setitem(production_generation.GENERATOR_BACKENDS, "stream_fake", lambda **_kwargs: pytest.fail("array backend should not run"))
    monkeypatch.setitem(production_generation.STREAMING_GENERATOR_BACKENDS, "stream_fake", _fake_streaming_backend)
    config = {
        "yolo_dataset_yaml": str(dataset_yaml),
        "output_root": str(output_root),
        "overwrite": False,
        "resume": True,
        "filter": {"enabled": False},
        "metrics": {"enabled": False},
        "generators": [{"name": "streamed", "backend": "stream_fake"}],
    }

    generate_production_synthetic_datasets(config=config, skip_filter=True, skip_metrics=True)

    assert saved_ids == [2]
    assert np.load(output_dir / "images" / "sample_000001.npy").mean() == pytest.approx(99.0)
    assert np.load(output_dir / "images" / "sample_000002.npy").mean() == pytest.approx(11.0)


def test_generation_retries_images_above_invalid_instance_ratio_threshold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_yaml = _make_yolo_dataset(tmp_path)
    output_root = tmp_path / "generated"
    backend_calls: list[list[int]] = []

    def _fake_backend(**kwargs):
        samples = kwargs["source_samples"]
        backend_calls.append([int(sample.index) for sample in samples])
        if len(backend_calls) == 1:
            return [
                np.full((8, 8), 1 if sample.index == 0 else 200, dtype=np.float32)
                for sample in samples
            ]
        return [np.full((8, 8), 200, dtype=np.float32) for _sample in samples]

    def _fake_load_filter(config, *, device):  # noqa: ANN001
        return object(), {"classifier_mode": "multiclass"}, {"0": 0.5}, 8, 0.0, tmp_path / "filter"

    def _fake_audit_dataset_with_loaded_filter(
        *,
        dataset_dir,
        config,
        device,
        model,
        summary,
        threshold,
        input_size,
        context_ratio,
        resolved_run_dir,
        write_filtered_annotations,
        export_results=True,
    ):
        del config, device, model, summary, threshold, input_size, context_ratio, resolved_run_dir, export_results
        payload = json.loads((Path(dataset_dir) / "annotations_unfiltered.json").read_text(encoding="utf-8"))
        images_by_id = {int(row["id"]): row for row in payload["images"]}
        instance_rows = []
        image_rows = []
        for image in payload["images"]:
            image_id = int(image["id"])
            value = float(np.load(Path(dataset_dir) / "images" / str(image["file_name"])).mean())
            anns = [ann for ann in payload["annotations"] if int(ann["image_id"]) == image_id]
            n_positive = 0
            for ann in anns:
                is_positive = value >= 100
                n_positive += int(is_positive)
                x, y, w, h = [float(v) for v in ann["bbox"]]
                instance_rows.append(
                    {
                        "annotation_id": int(ann["id"]),
                        "generated_image_id": image_id,
                        "generated_file_name": images_by_id[image_id]["file_name"],
                        "category_id": int(ann["category_id"]),
                        "category_name": "person",
                        "bbox_xywh": [x, y, w, h],
                        "bbox_xyxy": [x, y, x + w, y + h],
                        "crop_box_xyxy": [int(x), int(y), int(x + w), int(y + h)],
                        "size_bin": "small",
                        "normalized_area_ratio": float((w * h) / 64.0),
                        "expected_category_id": int(ann["category_id"]),
                        "expected_category_name": "person",
                        "predicted_category_id": int(ann["category_id"]) if is_positive else None,
                        "predicted_category_name": "person" if is_positive else "background",
                        "expected_class_probability": 0.9 if is_positive else 0.1,
                        "predicted_probability": 0.9 if is_positive else 0.9,
                        "is_background_prediction": not is_positive,
                        "is_class_match": is_positive,
                        "passes_expected_class_threshold": is_positive,
                        "is_positive": is_positive,
                    }
                )
            image_rows.append(
                {
                    "generated_image_id": image_id,
                    "generated_file_name": image["file_name"],
                    "n_instances": len(anns),
                    "n_positive_instances": n_positive,
                    "n_negative_instances": len(anns) - n_positive,
                    "valid_fraction": n_positive / max(1, len(anns)),
                }
            )
        audit_dir = Path(dataset_dir) / "filter_audit"
        audit_dir.mkdir(parents=True, exist_ok=True)
        with (audit_dir / "per_instance_manifest.jsonl").open("w", encoding="utf-8") as handle:
            for row in instance_rows:
                handle.write(json.dumps(row) + "\n")
        with (audit_dir / "per_image_manifest.jsonl").open("w", encoding="utf-8") as handle:
            for row in image_rows:
                handle.write(json.dumps(row) + "\n")
        filtered_summary = production_generation.write_filtered_annotations_from_audit(
            dataset_dir=dataset_dir,
            instance_rows=instance_rows,
        )
        return {"filtered_annotation_summary": filtered_summary}, instance_rows, image_rows

    monkeypatch.setitem(production_generation.GENERATOR_BACKENDS, "retry_fake", _fake_backend)
    monkeypatch.setattr(production_generation, "_load_filter", _fake_load_filter)
    monkeypatch.setattr(production_generation, "_audit_dataset_with_loaded_filter", _fake_audit_dataset_with_loaded_filter)
    monkeypatch.setattr(production_generation, "_export_loaded_filter_audit_results", lambda **_kwargs: None)
    config = {
        "yolo_dataset_yaml": str(dataset_yaml),
        "output_root": str(output_root),
        "seed": 11,
        "filter": {"enabled": True},
        "retry": {"enabled": True, "invalid_instance_ratio_threshold": 0.5, "max_tries": 3},
        "metrics": {"enabled": False},
        "generators": [{"name": "retry", "backend": "retry_fake"}],
    }

    summary = generate_production_synthetic_datasets(config=config, skip_metrics=True)
    generated_dir = output_root / "retry"
    filtered = json.loads((generated_dir / "annotations.json").read_text(encoding="utf-8"))
    retry_summary = json.loads((generated_dir / "metadata" / "retry_summary.json").read_text(encoding="utf-8"))

    assert backend_calls == [[0, 1], [0]]
    assert np.load(generated_dir / "images" / "sample_000001.npy").mean() == pytest.approx(200.0)
    assert len(filtered["images"]) == 2
    assert len(filtered["annotations"]) == 2
    assert retry_summary["n_retried_images"] == 1
    assert summary["generators"][0]["retry"]["n_retried_images"] == 1


def test_stay_backend_reconstructs_from_checkpoint_only_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import yaml

    checkpoint_path = tmp_path / "UNET" / "unet_last_ckpt.pt"
    checkpoint_path.parent.mkdir(parents=True)
    torch.save(
        {
            "unet_state": {
                "sentinel": torch.ones(1),
                "object_encoder.class_embedding.weight": torch.ones(80, 12),
            }
        },
        checkpoint_path,
    )
    preset_path = tmp_path / "stay_preset.yaml"
    preset_path.write_text(
        yaml.safe_dump(
            {
                "data": {"image_size": 37},
                "model": {"unet_config": str(tmp_path / "unet.json")},
                "layout_conditioning": {
                    "class_embed_dim": 12,
                    "bbox_embed_dim": 13,
                    "object_embed_dim": 14,
                    "use_style_latent": False,
                },
                "training": {"t_scale": 321.0, "train_target": "x0"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    class DummyUnet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.loaded_state = None

        def load_state_dict(self, state, strict=True):  # noqa: ANN001
            self.loaded_state = state

        def eval(self):
            return self

    captured: dict[str, object] = {}
    dummy_unet = DummyUnet()

    monkeypatch.setattr(production_generation, "load_unet_config", lambda _path: {"in_channels": 4})
    monkeypatch.setattr(production_generation, "_build_vae_from_preset", lambda _preset, device: "vae")

    def _fake_build_stay(unet_cfg, **kwargs):  # noqa: ANN001
        captured["stay_kwargs"] = kwargs
        return dummy_unet

    def _fake_from_stable(cls, unet, vae, **kwargs):  # noqa: ANN001
        captured["sampler_kwargs"] = kwargs
        captured["sampler_unet"] = unet
        captured["sampler_vae"] = vae
        return "stay_sampler"

    monkeypatch.setattr(production_generation, "build_stay_layout_conditioned_unet", _fake_build_stay)
    monkeypatch.setattr(production_generation.LayoutFlowMatchingSampler, "from_stable", classmethod(_fake_from_stable))

    sampler, image_size = production_generation._load_stay_sampler(
        generator_cfg={"checkpoint_path": str(checkpoint_path), "preset_path": str(preset_path)},
        dataset_payload={"names": {0: "person", 2: "car"}},
        device="cpu",
    )

    assert sampler == "stay_sampler"
    assert image_size == 37
    assert captured["sampler_vae"] == "vae"
    assert captured["sampler_kwargs"]["t_scale"] == 321.0
    assert captured["sampler_kwargs"]["train_target"] == "x0"
    assert captured["stay_kwargs"]["category_id_to_name"] == {0: "person", 2: "car"}
    assert captured["stay_kwargs"]["class_embed_dim"] == 12
    assert captured["stay_kwargs"]["num_classes"] == 80
    assert torch.equal(dummy_unet.loaded_state["sentinel"], torch.ones(1))


def test_stay_backend_uses_training_latent_size_with_pretrained_vae(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import yaml

    checkpoint_path = tmp_path / "UNET" / "unet_fm_epoch_120.pt"
    checkpoint_path.parent.mkdir(parents=True)
    torch.save(
        {
            "unet_state": {
                "sentinel": torch.ones(1),
                "object_encoder.class_embedding.weight": torch.ones(80, 12),
            }
        },
        checkpoint_path,
    )
    preset_path = tmp_path / "stay_preset.yaml"
    preset_path.write_text(
        yaml.safe_dump(
            {
                "data": {"image_size": 512},
                "model": {
                    "unet_config": str(tmp_path / "unet.json"),
                    "vae_config": None,
                    "vae_pretrained_model_name_or_path": "runwayml/stable-diffusion-v1-5",
                    "vae_pretrained_subfolder": "vae",
                },
                "layout_conditioning": {
                    "class_embed_dim": 12,
                    "bbox_embed_dim": 13,
                    "object_embed_dim": 14,
                    "use_style_latent": False,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    class DummyUnet(torch.nn.Module):
        def load_state_dict(self, state, strict=True):  # noqa: ANN001
            self.loaded_state = state

        def eval(self):
            return self

    captured: dict[str, object] = {}

    monkeypatch.setattr(
        production_generation,
        "load_unet_config",
        lambda _path: {"sample_size": 128, "in_channels": 4, "out_channels": 4},
    )
    monkeypatch.setattr(
        production_generation,
        "load_diffusers_vae_config",
        lambda *_args, **_kwargs: {
            "_backend": "diffusers_autoencoder_kl",
            "latent_channels": 4,
            "down_block_types": [
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
            ],
        },
    )
    monkeypatch.setattr(production_generation, "_build_vae_from_preset", lambda _preset, device: "vae")

    def _fake_build_stay(unet_cfg, **kwargs):  # noqa: ANN001
        captured["unet_cfg"] = unet_cfg
        return DummyUnet()

    monkeypatch.setattr(production_generation, "build_stay_layout_conditioned_unet", _fake_build_stay)
    monkeypatch.setattr(
        production_generation.LayoutFlowMatchingSampler,
        "from_stable",
        classmethod(lambda cls, unet, vae, **kwargs: "stay_sampler"),
    )

    sampler, image_size = production_generation._load_stay_sampler(
        generator_cfg={"checkpoint_path": str(checkpoint_path), "preset_path": str(preset_path)},
        dataset_payload={"names": {0: "person"}},
        device="cpu",
    )

    assert sampler == "stay_sampler"
    assert image_size == 512
    assert captured["unet_cfg"]["sample_size"] == 64


def test_regiondiff_backend_reconstructs_from_checkpoint_only_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import yaml

    checkpoint_path = tmp_path / "UNET" / "unet_fm_epoch_80.pt"
    checkpoint_path.parent.mkdir(parents=True)
    torch.save({"unet_state": {"sentinel": torch.ones(1) * 2}}, checkpoint_path)
    preset_path = tmp_path / "regiondiff_preset.yaml"
    preset_path.write_text(
        yaml.safe_dump(
            {
                "data": {"image_size": 41},
                "model": {"unet_config": str(tmp_path / "unet.json")},
                "layout_conditioning": {
                    "layout_token_dim": 17,
                    "bbox_fourier_dim": 9,
                    "attachment_kind": "residual",
                },
                "training": {"t_scale": 222.0, "train_target": "v"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    class DummyWrapper(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.loaded_state = None

        def to(self, device):  # noqa: ANN001
            self.device = device
            return self

        def load_state_dict(self, state, strict=True):  # noqa: ANN001
            self.loaded_state = state

        def eval(self):
            return self

    captured: dict[str, object] = {}
    dummy_wrapper = DummyWrapper()

    monkeypatch.setattr(
        production_generation,
        "load_unet_config",
        lambda _path: {"in_channels": 4},
    )
    monkeypatch.setattr(
        production_generation,
        "build_fm_unet_from_config",
        lambda _cfg, device: "base_unet",
    )
    monkeypatch.setattr(
        production_generation,
        "_build_vae_from_preset",
        lambda _preset, device: "vae",
    )

    def _fake_build_regiondiff(**kwargs):  # noqa: ANN003
        captured["regiondiff_kwargs"] = kwargs
        return dummy_wrapper

    def _fake_from_stable(cls, unet, vae, **kwargs):  # noqa: ANN001
        captured["sampler_kwargs"] = kwargs
        captured["sampler_unet"] = unet
        captured["sampler_vae"] = vae
        return "regiondiff_sampler"

    monkeypatch.setattr(production_generation, "build_regiondiff_wrapper", _fake_build_regiondiff)
    monkeypatch.setattr(production_generation.FlowMatchingSampler, "from_stable", classmethod(_fake_from_stable))

    sampler, image_size, label_id_map = production_generation._load_regiondiff_sampler(
        generator_cfg={"checkpoint_path": str(checkpoint_path), "preset_path": str(preset_path)},
        dataset_payload={"names": {0: "person", 1: "car"}},
        device="cpu",
    )

    assert sampler == "regiondiff_sampler"
    assert image_size == 41
    assert label_id_map == {}
    assert captured["sampler_vae"] == "vae"
    assert captured["sampler_kwargs"]["t_scale"] == 222.0
    assert captured["regiondiff_kwargs"]["category_id_to_name"] == {0: "person", 1: "car"}
    assert captured["regiondiff_kwargs"]["backbone_kind"] == "fm_unet2d"
    assert captured["regiondiff_kwargs"]["attachment_kind"] == "residual"
    assert torch.equal(dummy_wrapper.loaded_state["sentinel"], torch.ones(1) * 2)


def test_regiondiff_backend_uses_training_latent_size_with_pretrained_vae(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import yaml

    checkpoint_path = tmp_path / "UNET" / "unet_fm_epoch_80.pt"
    checkpoint_path.parent.mkdir(parents=True)
    torch.save({"unet_state": {"sentinel": torch.ones(1) * 2}}, checkpoint_path)
    preset_path = tmp_path / "regiondiff_preset.yaml"
    preset_path.write_text(
        yaml.safe_dump(
            {
                "data": {"image_size": 512},
                "model": {
                    "unet_config": str(tmp_path / "unet.json"),
                    "vae_config": None,
                    "vae_pretrained_model_name_or_path": "runwayml/stable-diffusion-v1-5",
                    "vae_pretrained_subfolder": "vae",
                },
                "layout_conditioning": {"layout_token_dim": 17},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    class DummyWrapper(torch.nn.Module):
        def to(self, device):  # noqa: ANN001
            return self

        def load_state_dict(self, state, strict=True):  # noqa: ANN001
            self.loaded_state = state

        def eval(self):
            return self

    captured: dict[str, object] = {}

    monkeypatch.setattr(
        production_generation,
        "load_unet_config",
        lambda _path: {"sample_size": 128, "in_channels": 4, "out_channels": 4},
    )
    monkeypatch.setattr(
        production_generation,
        "load_diffusers_vae_config",
        lambda *_args, **_kwargs: {
            "_backend": "diffusers_autoencoder_kl",
            "latent_channels": 4,
            "down_block_types": [
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
            ],
        },
    )

    def _fake_build_fm(unet_cfg, device):  # noqa: ANN001
        captured["unet_cfg"] = unet_cfg
        return "base_unet"

    monkeypatch.setattr(production_generation, "build_fm_unet_from_config", _fake_build_fm)
    monkeypatch.setattr(production_generation, "_build_vae_from_preset", lambda _preset, device: "vae")
    monkeypatch.setattr(production_generation, "build_regiondiff_wrapper", lambda **_kwargs: DummyWrapper())
    monkeypatch.setattr(
        production_generation.FlowMatchingSampler,
        "from_stable",
        classmethod(lambda cls, unet, vae, **kwargs: "regiondiff_sampler"),
    )

    sampler, image_size, label_id_map = production_generation._load_regiondiff_sampler(
        generator_cfg={"checkpoint_path": str(checkpoint_path), "preset_path": str(preset_path)},
        dataset_payload={"names": {0: "person", 1: "car"}},
        device="cpu",
    )

    assert sampler == "regiondiff_sampler"
    assert image_size == 512
    assert label_id_map == {}
    assert captured["unet_cfg"]["sample_size"] == 64


def test_regiondiff_sd_backend_reconstructs_from_checkpoint_only_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import yaml

    checkpoint_path = tmp_path / "UNET" / "unet_sd_uncond_epoch_80.pt"
    checkpoint_path.parent.mkdir(parents=True)
    torch.save({"unet_state": {"sentinel": torch.ones(1) * 4}}, checkpoint_path)
    preset_path = tmp_path / "regiondiff_sd_preset.yaml"
    preset_path.write_text(
        yaml.safe_dump(
            {
                "data": {"image_size": 43},
                "model": {"unet_config": str(tmp_path / "unet.json")},
                "layout_conditioning": {
                    "enabled": True,
                    "variant": "regiondiff_v1",
                    "layout_token_dim": 19,
                    "attachment_kind": "attention",
                },
                "diffusion": {
                    "num_train_timesteps": 111,
                    "beta_schedule": "linear",
                    "beta_start": 0.001,
                    "beta_end": 0.02,
                    "prediction_type": "epsilon",
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    class DummyWrapper(torch.nn.Module):
        def to(self, device):  # noqa: ANN001
            self.device = device
            return self

        def load_state_dict(self, state, strict=True):  # noqa: ANN001
            self.loaded_state = state

        def eval(self):
            return self

    class DummyScheduler:
        def __init__(self, **kwargs):  # noqa: ANN003
            self.kwargs = kwargs

    captured: dict[str, object] = {}
    dummy_wrapper = DummyWrapper()

    monkeypatch.setattr(production_generation, "load_unet_config", lambda _path: {"in_channels": 4})
    monkeypatch.setattr(production_generation, "build_fm_unet_from_config", lambda _cfg, device: "base_unet")
    monkeypatch.setattr(production_generation, "_build_vae_from_preset", lambda _preset, device: "vae")
    monkeypatch.setattr(production_generation, "import_diffusers_attr", lambda *_args: DummyScheduler)

    def _fake_build_regiondiff(**kwargs):  # noqa: ANN003
        captured["regiondiff_kwargs"] = kwargs
        return dummy_wrapper

    def _fake_from_stable(cls, unet, vae, noise_scheduler, **kwargs):  # noqa: ANN001
        captured["sampler_unet"] = unet
        captured["sampler_vae"] = vae
        captured["scheduler"] = noise_scheduler
        captured["sampler_kwargs"] = kwargs
        return "regiondiff_sd_sampler"

    monkeypatch.setattr(production_generation, "build_regiondiff_wrapper", _fake_build_regiondiff)
    monkeypatch.setattr(
        production_generation.UnconditionalStableDiffusionSampler,
        "from_stable",
        classmethod(_fake_from_stable),
    )

    sampler, image_size, label_id_map = production_generation._load_regiondiff_sd_sampler(
        generator_cfg={"checkpoint_path": str(checkpoint_path), "preset_path": str(preset_path)},
        dataset_payload={"names": {0: "person", 1: "car"}},
        device="cpu",
    )

    assert sampler == "regiondiff_sd_sampler"
    assert image_size == 43
    assert label_id_map == {}
    assert captured["sampler_vae"] == "vae"
    assert captured["sampler_kwargs"]["device"] == "cpu"
    assert captured["regiondiff_kwargs"]["backbone_kind"] == "sd_uncond_unet2d"
    assert captured["regiondiff_kwargs"]["category_id_to_name"] == {0: "person", 1: "car"}
    assert captured["regiondiff_kwargs"]["num_classes"] == 2
    assert captured["scheduler"].kwargs["num_train_timesteps"] == 111
    assert torch.equal(dummy_wrapper.loaded_state["sentinel"], torch.ones(1) * 4)


def test_regiondiff_backend_maps_compact_labels_to_checkpoint_classes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import yaml

    checkpoint_path = tmp_path / "UNET" / "unet_fm_epoch_80.pt"
    checkpoint_path.parent.mkdir(parents=True)
    torch.save(
        {
            "layout_tokenizer.class_text_features": torch.zeros(4, 17),
            "sentinel": torch.ones(1) * 3,
        },
        checkpoint_path,
    )
    preset_path = tmp_path / "regiondiff_preset.yaml"
    preset_path.write_text(
        yaml.safe_dump(
            {
                "data": {"image_size": 41},
                "model": {"unet_config": str(tmp_path / "unet.json")},
                "layout_conditioning": {"layout_token_dim": 17},
                "training": {"t_scale": 222.0, "train_target": "v"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    class DummyWrapper(torch.nn.Module):
        def to(self, device):  # noqa: ANN001
            return self

        def load_state_dict(self, state, strict=True):  # noqa: ANN001
            self.loaded_state = state

        def eval(self):
            return self

    captured: dict[str, object] = {}

    monkeypatch.setattr(production_generation, "load_unet_config", lambda _path: {"in_channels": 4})
    monkeypatch.setattr(production_generation, "build_fm_unet_from_config", lambda _cfg, device: "base_unet")
    monkeypatch.setattr(production_generation, "_build_vae_from_preset", lambda _preset, device: "vae")

    def _fake_build_regiondiff(**kwargs):  # noqa: ANN003
        captured["regiondiff_kwargs"] = kwargs
        return DummyWrapper()

    monkeypatch.setattr(production_generation, "build_regiondiff_wrapper", _fake_build_regiondiff)
    monkeypatch.setattr(
        production_generation.FlowMatchingSampler,
        "from_stable",
        classmethod(lambda cls, unet, vae, **kwargs: "regiondiff_sampler"),
    )

    _sampler, _image_size, label_id_map = production_generation._load_regiondiff_sampler(
        generator_cfg={
            "checkpoint_path": str(checkpoint_path),
            "preset_path": str(preset_path),
            "checkpoint_category_id_to_name": {0: "person", 1: "bike", 2: "car", 3: "bus"},
        },
        dataset_payload={"names": {0: "person", 1: "car"}},
        device="cpu",
    )

    assert label_id_map == {0: 0, 1: 2}
    assert captured["regiondiff_kwargs"]["category_id_to_name"] == {
        0: "person",
        1: "bike",
        2: "car",
        3: "bus",
    }

    batch = {
        "labels": torch.tensor([[0, 1, 0]]),
        "object_mask": torch.tensor([[True, True, False]]),
    }
    remapped = production_generation._remap_layout_batch_labels(batch, label_id_map)

    assert remapped["labels"].tolist() == [[0, 2, 0]]


def test_filtered_annotations_remove_only_invalid_instances_and_keep_images(tmp_path: Path) -> None:
    image_path = tmp_path / "real.png"
    label_path = tmp_path / "real.txt"
    image_path.write_bytes(b"fake")
    label_path.write_text("0 0.5 0.5 0.25 0.25\n1 0.2 0.2 0.1 0.1\n", encoding="utf-8")
    samples = [
        YOLOTrainSample(
            index=0,
            image_path=image_path,
            label_path=label_path,
            boxes=[YOLOBox(0, 0.5, 0.5, 0.25, 0.25), YOLOBox(1, 0.2, 0.2, 0.1, 0.1)],
        ),
        YOLOTrainSample(
            index=1,
            image_path=image_path,
            label_path=label_path,
            boxes=[YOLOBox(0, 0.5, 0.5, 0.25, 0.25)],
        ),
    ]
    dataset_dir = tmp_path / "generated"
    export_generated_candidate_dataset(
        output_dir=dataset_dir,
        source_samples=samples,
        generated_arrays=[np.ones((16, 16), dtype=np.float32), np.ones((16, 16), dtype=np.float32) * 2],
        dataset_payload={"names": {0: "person", 1: "car"}, "_yaml_path": "dataset.yaml"},
        generator_kind="test",
    )
    instance_rows = [
        {
            "annotation_id": 1,
            "generated_image_id": 1,
            "generated_file_name": "sample_000001.npy",
            "bbox_xyxy": [6, 6, 10, 10],
            "expected_category_name": "person",
            "predicted_category_name": "person",
            "expected_class_probability": 0.9,
            "is_positive": True,
        },
        {
            "annotation_id": 2,
            "generated_image_id": 1,
            "generated_file_name": "sample_000001.npy",
            "bbox_xyxy": [2, 2, 4, 4],
            "expected_category_name": "car",
            "predicted_category_name": "background",
            "expected_class_probability": 0.1,
            "is_positive": False,
        },
        {
            "annotation_id": 3,
            "generated_image_id": 2,
            "generated_file_name": "sample_000002.npy",
            "bbox_xyxy": [6, 6, 10, 10],
            "expected_category_name": "person",
            "predicted_category_name": "background",
            "expected_class_probability": 0.2,
            "is_positive": False,
        },
    ]

    summary = write_filtered_annotations_from_audit(dataset_dir=dataset_dir, instance_rows=instance_rows)
    filtered = json.loads((dataset_dir / "annotations.json").read_text(encoding="utf-8"))

    assert summary["n_images"] == 2
    assert summary["n_annotations"] == 1
    assert len(filtered["images"]) == 2
    assert [ann["id"] for ann in filtered["annotations"]] == [1]

    filtered_overlay_paths = render_layout_overlay_previews(
        dataset_dir=dataset_dir,
        max_images=2,
        annotations_filename="annotations.json",
        output_dir_name="filtered_layout_overlays",
    )
    assert len(filtered_overlay_paths) == 2
    assert (dataset_dir / "filtered_layout_overlays" / "sample_000001.png").is_file()
    assert (dataset_dir / "filtered_layout_overlays" / "sample_000002.png").is_file()
    filtered_overlay_summary = json.loads(
        (dataset_dir / "filtered_layout_overlays" / "summary.json").read_text(encoding="utf-8")
    )
    assert filtered_overlay_summary["annotations_path"] == "annotations.json"
    assert filtered_overlay_summary["n_annotations"] == 1

    audit_dir = dataset_dir / "filter_audit"
    audit_dir.mkdir()
    with (audit_dir / "per_instance_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in instance_rows:
            handle.write(json.dumps(row) + "\n")
    paths = render_sanity_check_images(dataset_dir=dataset_dir, max_images=2)
    assert paths
    assert all(Path(path).is_file() for path in paths)


def test_metric_summary_from_features_reports_fid_kid_and_mmd() -> None:
    real = np.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    generated = real.copy()

    summary = compute_metric_summary_from_features(
        real,
        generated,
        kid_subsets=2,
        kid_subset_size=2,
        kid_seed=3,
        mmd_bandwidths=[0.5, 1.0],
    )

    assert summary["skipped"] is False
    assert summary["fid"] == pytest.approx(0.0, abs=1e-6)
    assert summary["mmd"] == pytest.approx(0.0, abs=1e-12)


def test_yolo_experiment_b_accepts_precomputed_aug_without_filter(tmp_path: Path) -> None:
    from PIL import Image
    import yaml

    yolo_root = tmp_path / "yolo"
    for split in ("train", "val", "test"):
        (yolo_root / "images" / split).mkdir(parents=True)
        (yolo_root / "labels" / split).mkdir(parents=True)
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(yolo_root / "images" / "train" / "real.png")
    (yolo_root / "labels" / "train" / "real.txt").write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
    dataset_yaml = yolo_root / "full_train.yaml"
    dataset_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(yolo_root),
                "train": str((yolo_root / "images" / "train").resolve()),
                "val": str((yolo_root / "images" / "val").resolve()),
                "test": str((yolo_root / "images" / "test").resolve()),
                "names": {0: "person"},
                "nc": 1,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    export_generated_candidate_dataset(
        output_dir=generated_dir,
        source_samples=[
            YOLOTrainSample(
                index=0,
                image_path=yolo_root / "images" / "train" / "real.png",
                label_path=yolo_root / "labels" / "train" / "real.txt",
                boxes=[YOLOBox(0, 0.5, 0.5, 0.25, 0.25)],
            )
        ],
        generated_arrays=[np.ones((8, 8), dtype=np.float32)],
        dataset_payload={"names": {0: "person"}, "_yaml_path": str(dataset_yaml)},
        generator_kind="regiondiff_test",
    )

    cfg = YOLOExperimentConfig()
    cfg.data.dataset_yaml = str(dataset_yaml)
    cfg.data.full_train_dataset_yaml = str(dataset_yaml)
    cfg.experiment_b.mode = "precomputed_aug"
    cfg.experiment_b.precomputed_dataset_dir = str(generated_dir)
    cfg.experiment_b.augmented_yolo_root = str(tmp_path / "augmented")
    cfg.experiment_b.filter.enabled = False
    cfg.output.experiment_name = "smoked_yolo_test"

    validate_experiment_b_config(cfg)
    summary = prepare_experiment_b_dataset(cfg, device="cpu")
    augmented_yaml = Path(summary["augmented_dataset_yaml"])
    assert augmented_yaml.is_file()
    assert summary["n_generated_images"] == 1
    assert summary["n_kept_synthetic_images"] == 1
