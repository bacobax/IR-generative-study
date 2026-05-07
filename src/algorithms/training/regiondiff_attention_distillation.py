"""RegionDiff attention-map distillation helpers for flow matching.

The distillation target in this module is intentionally narrow: only the
attention probabilities produced by ``RegionSelfAttentionAdapter`` are
captured and compared. The teacher is a frozen SD1.5 + RegionDiff artifact;
its prediction target and latent features are not used for the FM loss.
"""

from __future__ import annotations

import math
import os
import re
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import torch
import torch.nn.functional as F

from src.models.regiondiffusion import RegionSelfAttentionAdapter


@dataclass
class AttentionMapRecord:
    """Captured RegionDiff adapter attention for one module call."""

    attention: torch.Tensor
    layer_name: str
    alias: str
    resolution: Optional[int]


def _infer_square_resolution(seq_len: int) -> Optional[int]:
    resolution = int(round(float(seq_len) ** 0.5))
    if resolution * resolution != int(seq_len):
        return None
    return resolution


def _normalize_text(value: object) -> str:
    return str(value).strip().lower().replace("_", " ").replace("-", " ")


def _layer_alias(layer_name: str, resolution: Optional[int]) -> str:
    """Build a compact alias from a RegionDiff module path when possible."""
    res = "x" if resolution is None else str(int(resolution))
    patterns = (
        (r"down_blocks\.(\d+)\.attentions\.(\d+)", "down"),
        (r"up_blocks\.(\d+)\.attentions\.(\d+)", "up"),
        (r"mid_block\.attentions\.(\d+)", "mid"),
    )
    for pattern, prefix in patterns:
        match = re.search(pattern, layer_name)
        if match is None:
            continue
        if prefix == "mid":
            return f"mid_{res}_a{match.group(1)}"
        return f"{prefix}_{res}_b{match.group(1)}_a{match.group(2)}"
    return f"region_{res}_{layer_name.replace('.', '_')}"


def _matches_layer_selection(
    *,
    layer_name: str,
    alias: str,
    selected_layers: Iterable[str],
) -> bool:
    selected = [str(item).strip() for item in selected_layers if str(item).strip()]
    if not selected:
        return True
    name_l = layer_name.lower()
    alias_l = alias.lower()
    normalized_name = name_l.replace(".region_adapter", "")
    for item in selected:
        item_l = item.lower()
        if item_l in {name_l, normalized_name, alias_l}:
            return True
        if item_l in name_l or item_l in alias_l:
            return True
    return False


class RegionDiffAttentionRecorder:
    """Context manager that captures only RegionDiff adapter attention maps."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        selected_layers: Optional[Iterable[str]] = None,
        detach: bool = False,
    ) -> None:
        self.model = model
        self.selected_layers = list(selected_layers or [])
        self.detach = bool(detach)
        self.records: Dict[str, AttentionMapRecord] = {}
        self._module_names: Dict[int, str] = {}
        self._previous_recorders: Dict[int, Any] = {}
        self._record_counts: Dict[str, int] = {}

    def clear(self) -> None:
        self.records.clear()
        self._record_counts.clear()

    def __enter__(self) -> "RegionDiffAttentionRecorder":
        self.clear()
        self._module_names.clear()
        self._previous_recorders.clear()
        for name, module in self.model.named_modules():
            if not isinstance(module, RegionSelfAttentionAdapter):
                continue
            module_id = id(module)
            self._module_names[module_id] = str(name)
            self._previous_recorders[module_id] = getattr(module, "_attention_recorder", None)
            module._attention_recorder = self
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        for module in self.model.modules():
            if not isinstance(module, RegionSelfAttentionAdapter):
                continue
            module_id = id(module)
            if module_id in self._previous_recorders:
                module._attention_recorder = self._previous_recorders[module_id]
        self._previous_recorders.clear()

    def record(
        self,
        module: RegionSelfAttentionAdapter,
        attention: torch.Tensor,
        *,
        region_token_mask: torch.Tensor,
        layout_tokens: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> None:
        del region_token_mask, layout_tokens, hidden_states
        layer_name = self._module_names.get(id(module))
        if layer_name is None:
            return
        if attention.ndim != 4:
            return
        mean_attention = attention.mean(dim=1)
        if self.detach:
            mean_attention = mean_attention.detach()
        resolution = _infer_square_resolution(int(mean_attention.shape[1]))
        alias = _layer_alias(layer_name, resolution)
        if not _matches_layer_selection(
            layer_name=layer_name,
            alias=alias,
            selected_layers=self.selected_layers,
        ):
            return

        key = layer_name
        count = self._record_counts.get(key, 0)
        self._record_counts[key] = count + 1
        if count:
            key = f"{key}#{count + 1}"
        self.records[key] = AttentionMapRecord(
            attention=mean_attention,
            layer_name=layer_name,
            alias=alias,
            resolution=resolution,
        )


def fm_timesteps_to_sd_timesteps(
    fm_t: torch.Tensor,
    *,
    num_train_timesteps: int = 1000,
) -> torch.Tensor:
    """Map FM noise-to-data time to SD data-to-noise integer timesteps."""
    max_timestep = max(int(num_train_timesteps) - 1, 0)
    return torch.round((1.0 - fm_t.detach().float()).clamp(0.0, 1.0) * max_timestep).long()


def _find_stage2_artifact_dir(path: str | Path) -> tuple[Path, Optional[Path]]:
    from src.algorithms.stable_diffusion.layout_models import (
        STAGE2_CHECKPOINT_UNET_WEIGHTS,
        STAGE2_MANIFEST_NAME,
        STAGE2_UNET_WEIGHTS,
    )

    candidate = Path(path)
    if candidate.is_dir() and (candidate / STAGE2_MANIFEST_NAME).is_file():
        return candidate, None
    if candidate.is_dir() and (candidate / STAGE2_CHECKPOINT_UNET_WEIGHTS).is_file():
        return _find_stage2_artifact_dir(candidate.parent)[0], candidate / STAGE2_CHECKPOINT_UNET_WEIGHTS
    if candidate.is_file() and candidate.name == STAGE2_UNET_WEIGHTS:
        return candidate.parent, None
    if candidate.is_file() and candidate.name == STAGE2_CHECKPOINT_UNET_WEIGHTS:
        return _find_stage2_artifact_dir(candidate.parent.parent)[0], candidate

    current = candidate if candidate.is_dir() else candidate.parent
    for parent in [current, *current.parents]:
        if (parent / STAGE2_MANIFEST_NAME).is_file():
            weights = candidate if candidate.is_file() else None
            return parent, weights
    raise FileNotFoundError(
        "Could not resolve an SD RegionDiff stage-2 artifact from "
        f"{str(path)!r}."
    )


def _load_state_dict(path: str | Path) -> Dict[str, torch.Tensor]:
    path = Path(path)
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file as safe_load_file
        except ImportError as exc:  # pragma: no cover
            raise ImportError("safetensors is required to load this teacher checkpoint") from exc
        return safe_load_file(str(path))
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    return state


class RegionDiffAttentionTeacher:
    """Frozen SD RegionDiff teacher wrapper used by FM attention KD."""

    def __init__(
        self,
        *,
        pipeline: Any,
        manifest: Mapping[str, Any],
        device: torch.device,
    ) -> None:
        self.pipeline = pipeline
        self.manifest = dict(manifest)
        self.device = torch.device(device)
        self.unet = pipeline.unet
        scheduler_config = getattr(getattr(pipeline, "scheduler", None), "config", None)
        self.num_train_timesteps = int(
            getattr(scheduler_config, "num_train_timesteps", 1000) or 1000
        )
        self.category_id_to_name = {
            int(key): str(value)
            for key, value in getattr(self.unet, "category_id_to_name", {}).items()
        }

    def freeze(self) -> None:
        for module_name in ("unet", "vae", "text_encoder"):
            module = getattr(self.pipeline, module_name, None)
            if module is None:
                continue
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def build_prompts(self, labels: torch.Tensor, object_mask: torch.Tensor) -> list[str]:
        prompt_mode = str(self.manifest.get("prompt_mode", "class_list"))
        constant_prompt = str(self.manifest.get("constant_prompt", "thermal image"))
        suffix = str(self.manifest.get("thermal_scene_suffix", "in thermal scene.")).strip()
        if suffix and not suffix.endswith("."):
            suffix = f"{suffix}."
        prompts: list[str] = []
        labels_cpu = labels.detach().cpu()
        mask_cpu = object_mask.detach().cpu()
        for batch_idx in range(int(labels_cpu.shape[0])):
            if prompt_mode == "constant":
                prompts.append(constant_prompt)
                continue
            names: list[str] = []
            seen: set[str] = set()
            for object_idx in range(int(labels_cpu.shape[1])):
                if not bool(mask_cpu[batch_idx, object_idx]):
                    continue
                label = int(labels_cpu[batch_idx, object_idx].item())
                name = str(self.category_id_to_name.get(label, f"class {label}")).replace("_", " ")
                normalized = _normalize_text(name)
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                names.append(name)
            if not names:
                prompts.append(constant_prompt)
                continue
            prompt = f"An image of {' and '.join(names)}"
            prompts.append(f"{prompt} {suffix}".strip() if suffix else prompt)
        return prompts

    @torch.no_grad()
    def encode_prompts(self, prompts: list[str]) -> torch.Tensor:
        tokenized = self.pipeline.tokenizer(
            prompts,
            max_length=self.pipeline.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokenized.input_ids.to(self.device)
        attention_mask = getattr(tokenized, "attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        outputs = self.pipeline.text_encoder(
            input_ids,
            attention_mask=attention_mask,
            return_dict=False,
        )
        return outputs[0]

    def forward_attention(
        self,
        *,
        noisy_latents: torch.Tensor,
        fm_t: torch.Tensor,
        cond_kwargs: Mapping[str, torch.Tensor],
        detach_teacher: bool = True,
    ) -> None:
        context = torch.no_grad() if detach_teacher else nullcontext()
        with context:
            sample = noisy_latents.detach().to(device=self.device)
            dtype = getattr(self.unet, "dtype", sample.dtype)
            sample = sample.to(dtype=dtype)
            timesteps = fm_timesteps_to_sd_timesteps(
                fm_t.to(device=self.device),
                num_train_timesteps=self.num_train_timesteps,
            )
            boxes = cond_kwargs["boxes_xyxy_norm"].to(device=self.device, dtype=sample.dtype)
            labels = cond_kwargs["labels"].to(device=self.device)
            object_mask = cond_kwargs["object_mask"].to(device=self.device)
            encoder_hidden_states = self.encode_prompts(
                self.build_prompts(labels=labels, object_mask=object_mask)
            ).to(dtype=sample.dtype)
            self.unet(
                sample,
                timesteps,
                encoder_hidden_states,
                cross_attention_kwargs={
                    "boxes_xyxy_norm": boxes,
                    "labels": labels,
                    "object_mask": object_mask,
                },
                return_dict=False,
            )


def load_regiondiff_attention_teacher(
    teacher_checkpoint: str,
    *,
    device: torch.device | str,
    torch_dtype: Optional[torch.dtype] = None,
) -> RegionDiffAttentionTeacher:
    """Load and freeze an SD1.5 RegionDiff teacher from a stage-2 artifact."""
    from src.algorithms.stable_diffusion.layout_models import load_stage2_layout_pipeline

    artifact_dir, override_weights = _find_stage2_artifact_dir(teacher_checkpoint)
    pipeline, manifest = load_stage2_layout_pipeline(
        stage2_dir=str(artifact_dir),
        torch_dtype=torch_dtype,
    )
    if override_weights is not None:
        missing, unexpected = pipeline.unet.load_state_dict(
            _load_state_dict(override_weights),
            strict=False,
        )
        if missing or unexpected:
            raise RuntimeError(
                "Teacher RegionDiff checkpoint did not load cleanly. "
                f"Missing keys={missing[:5]}, unexpected keys={unexpected[:5]}"
            )
    if hasattr(pipeline, "to"):
        pipeline = pipeline.to(device)
    teacher = RegionDiffAttentionTeacher(
        pipeline=pipeline,
        manifest=manifest,
        device=torch.device(device),
    )
    teacher.freeze()
    return teacher


def _coerce_record(key: str, value: AttentionMapRecord | torch.Tensor) -> AttentionMapRecord:
    if isinstance(value, AttentionMapRecord):
        return value
    resolution = _infer_square_resolution(int(value.shape[1])) if value.ndim >= 2 else None
    return AttentionMapRecord(
        attention=value,
        layer_name=key,
        alias=_layer_alias(key, resolution),
        resolution=resolution,
    )


def _filter_records(
    maps: Mapping[str, AttentionMapRecord | torch.Tensor],
    selected_layers: Iterable[str],
) -> Dict[str, AttentionMapRecord]:
    records: Dict[str, AttentionMapRecord] = {}
    for key, value in maps.items():
        record = _coerce_record(str(key), value)
        if _matches_layer_selection(
            layer_name=record.layer_name,
            alias=record.alias,
            selected_layers=selected_layers,
        ):
            records[str(key)] = record
    return records


def _match_attention_layers(
    teacher_maps: Mapping[str, AttentionMapRecord | torch.Tensor],
    student_maps: Mapping[str, AttentionMapRecord | torch.Tensor],
    *,
    selected_layers: Iterable[str],
) -> tuple[list[tuple[str, AttentionMapRecord, str, AttentionMapRecord]], list[str]]:
    teacher = _filter_records(teacher_maps, selected_layers)
    student = _filter_records(student_maps, selected_layers)
    pairs: list[tuple[str, AttentionMapRecord, str, AttentionMapRecord]] = []
    matched_teacher: set[str] = set()
    matched_student: set[str] = set()

    for key in sorted(set(teacher).intersection(student)):
        pairs.append((key, teacher[key], key, student[key]))
        matched_teacher.add(key)
        matched_student.add(key)

    def _by_alias(records: Mapping[str, AttentionMapRecord]) -> Dict[str, list[str]]:
        aliases: Dict[str, list[str]] = {}
        for key, record in sorted(records.items()):
            aliases.setdefault(record.alias, []).append(key)
        return aliases

    teacher_aliases = _by_alias(teacher)
    student_aliases = _by_alias(student)
    for alias in sorted(set(teacher_aliases).intersection(student_aliases)):
        for teacher_key, student_key in zip(teacher_aliases[alias], student_aliases[alias]):
            if teacher_key in matched_teacher or student_key in matched_student:
                continue
            pairs.append((teacher_key, teacher[teacher_key], student_key, student[student_key]))
            matched_teacher.add(teacher_key)
            matched_student.add(student_key)

    remaining_teacher = [key for key in sorted(teacher) if key not in matched_teacher]
    remaining_student = [key for key in sorted(student) if key not in matched_student]
    for student_key in list(remaining_student):
        student_record = student[student_key]
        same_resolution = [
            key for key in remaining_teacher
            if teacher[key].resolution == student_record.resolution
        ]
        if not same_resolution:
            continue
        teacher_key = same_resolution[0]
        pairs.append((teacher_key, teacher[teacher_key], student_key, student_record))
        matched_teacher.add(teacher_key)
        matched_student.add(student_key)
        remaining_teacher.remove(teacher_key)
        remaining_student.remove(student_key)

    for teacher_key, student_key in zip(remaining_teacher, remaining_student):
        if teacher_key in matched_teacher or student_key in matched_student:
            continue
        pairs.append((teacher_key, teacher[teacher_key], student_key, student[student_key]))
        matched_teacher.add(teacher_key)
        matched_student.add(student_key)

    missing = [
        *[f"teacher:{key}" for key in sorted(set(teacher) - matched_teacher)],
        *[f"student:{key}" for key in sorted(set(student) - matched_student)],
    ]
    return pairs, missing


def _selected_instance_mask(
    *,
    labels: torch.Tensor,
    object_mask: torch.Tensor,
    timesteps: Optional[torch.Tensor],
    distillation_config: Any,
    category_id_to_name: Optional[Mapping[int, str]],
) -> tuple[torch.Tensor, list[str]]:
    selected_categories = list(getattr(distillation_config, "selected_categories", []) or [])
    mask = object_mask.bool().clone()
    if timesteps is not None:
        start, end = getattr(distillation_config, "timestep_range", (0.0, 1.0))
        sample_mask = (timesteps >= float(start)) & (timesteps <= float(end))
        mask = mask & sample_mask.to(device=mask.device).view(-1, 1)

    if not selected_categories:
        return mask, []

    selected_ids: set[int] = set()
    selected_names: set[str] = set()
    for item in selected_categories:
        text = str(item).strip()
        if not text:
            continue
        try:
            selected_ids.add(int(text))
        except ValueError:
            selected_names.add(_normalize_text(text))

    category_id_to_name = {int(k): str(v) for k, v in (category_id_to_name or {}).items()}
    category_mask = torch.zeros_like(mask)
    labels_cpu = labels.detach().cpu()
    for batch_idx in range(int(labels.shape[0])):
        for object_idx in range(int(labels.shape[1])):
            label = int(labels_cpu[batch_idx, object_idx].item())
            label_name = _normalize_text(category_id_to_name.get(label, f"class {label}"))
            if label in selected_ids or label_name in selected_names:
                category_mask[batch_idx, object_idx] = True
    return mask & category_mask.to(device=mask.device), selected_categories


def _flat_bbox_mask(
    box_xyxy_norm: torch.Tensor,
    *,
    height: int,
    width: int,
) -> torch.Tensor:
    x1, y1, x2, y2 = [float(value) for value in box_xyxy_norm.detach().cpu().tolist()]
    ix1 = max(0, min(width - 1, int(math.floor(x1 * width))))
    iy1 = max(0, min(height - 1, int(math.floor(y1 * height))))
    ix2 = max(ix1 + 1, min(width, int(math.ceil(x2 * width))))
    iy2 = max(iy1 + 1, min(height, int(math.ceil(y2 * height))))
    mask = torch.zeros(height, width, dtype=torch.bool, device=box_xyxy_norm.device)
    mask[iy1:iy2, ix1:ix2] = True
    return mask.flatten()


def _resize_flat_map(
    value: torch.Tensor,
    *,
    source_resolution: int,
    target_resolution: int,
) -> torch.Tensor:
    if source_resolution == target_resolution:
        return value
    resized = F.interpolate(
        value.view(1, 1, source_resolution, source_resolution).float(),
        size=(target_resolution, target_resolution),
        mode="bilinear",
        align_corners=False,
    )
    return resized.view(-1).to(device=value.device, dtype=value.dtype)


def _zero_loss_like(
    student_maps: Mapping[str, AttentionMapRecord | torch.Tensor],
    fallback: torch.Tensor,
) -> torch.Tensor:
    for value in student_maps.values():
        record = _coerce_record("student", value)
        return record.attention.sum() * 0.0
    return fallback.sum() * 0.0


def compute_region_attention_distillation_loss(
    *,
    teacher_attention_maps: Mapping[str, AttentionMapRecord | torch.Tensor],
    student_attention_maps: Mapping[str, AttentionMapRecord | torch.Tensor],
    boxes_xyxy_norm: torch.Tensor,
    labels: torch.Tensor,
    object_mask: torch.Tensor,
    timesteps: Optional[torch.Tensor],
    distillation_config: Any,
    category_id_to_name: Optional[Mapping[int, str]] = None,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    """Compare teacher and student RegionDiff object-attention maps."""
    diagnostics: Dict[str, Any] = {
        "matched_layers": 0,
        "selected_instances": 0,
        "selected_categories": list(getattr(distillation_config, "selected_categories", []) or []),
        "skipped_layers_shape": 0,
        "skipped_layers_missing": 0,
        "skipped_layer_shape_names": [],
        "skipped_layer_missing_names": [],
        "loss_by_layer": {},
    }

    selected_mask, selected_categories = _selected_instance_mask(
        labels=labels,
        object_mask=object_mask,
        timesteps=timesteps,
        distillation_config=distillation_config,
        category_id_to_name=category_id_to_name,
    )
    diagnostics["selected_categories"] = selected_categories
    selected_indices = torch.nonzero(selected_mask, as_tuple=False)
    diagnostics["selected_instances"] = int(selected_indices.shape[0])
    if selected_indices.numel() == 0:
        return _zero_loss_like(student_attention_maps, boxes_xyxy_norm), diagnostics

    pairs, missing = _match_attention_layers(
        teacher_attention_maps,
        student_attention_maps,
        selected_layers=getattr(distillation_config, "selected_region_layers", []) or [],
    )
    diagnostics["skipped_layer_missing_names"] = missing
    diagnostics["skipped_layers_missing"] = len(missing)
    if not pairs:
        return _zero_loss_like(student_attention_maps, boxes_xyxy_norm), diagnostics

    max_objects = int(labels.shape[1])
    eps = 1e-8
    normalize_attention = bool(getattr(distillation_config, "normalize_attention", True))
    bbox_mask_only = bool(getattr(distillation_config, "bbox_mask_only", True))
    loss_type = str(getattr(distillation_config, "loss_type", "attention_kl"))
    layer_losses: list[torch.Tensor] = []

    for teacher_key, teacher_record, student_key, student_record in pairs:
        teacher_map = teacher_record.attention
        student_map = student_record.attention
        layer_name = f"{teacher_key}->{student_key}"
        if bool(getattr(distillation_config, "detach_teacher", True)):
            teacher_map = teacher_map.detach()
        if teacher_map.ndim != 3 or student_map.ndim != 3:
            diagnostics["skipped_layer_shape_names"].append(layer_name)
            continue
        if int(teacher_map.shape[0]) != int(student_map.shape[0]):
            diagnostics["skipped_layer_shape_names"].append(layer_name)
            continue
        if int(teacher_map.shape[-1]) < max_objects or int(student_map.shape[-1]) < max_objects:
            diagnostics["skipped_layer_shape_names"].append(layer_name)
            continue
        teacher_resolution = _infer_square_resolution(int(teacher_map.shape[1]))
        student_resolution = _infer_square_resolution(int(student_map.shape[1]))
        if teacher_resolution is None or student_resolution is None:
            diagnostics["skipped_layer_shape_names"].append(layer_name)
            continue

        instance_losses: list[torch.Tensor] = []
        for batch_idx_t, object_idx_t in selected_indices:
            batch_idx = int(batch_idx_t.item())
            object_idx = int(object_idx_t.item())
            teacher_obj = teacher_map[batch_idx, :, object_idx]
            student_obj = student_map[batch_idx, :, object_idx]
            teacher_obj = _resize_flat_map(
                teacher_obj,
                source_resolution=teacher_resolution,
                target_resolution=student_resolution,
            ).to(device=student_obj.device, dtype=student_obj.dtype)
            if bbox_mask_only:
                spatial_mask = _flat_bbox_mask(
                    boxes_xyxy_norm[batch_idx, object_idx],
                    height=student_resolution,
                    width=student_resolution,
                ).to(device=student_obj.device)
                if not bool(spatial_mask.any()):
                    continue
                teacher_obj = teacher_obj[spatial_mask]
                student_obj = student_obj[spatial_mask]

            teacher_obj = teacher_obj.float().clamp(min=eps)
            student_obj = student_obj.float().clamp(min=eps)
            if normalize_attention:
                teacher_obj = teacher_obj / teacher_obj.sum().clamp(min=eps)
                student_obj = student_obj / student_obj.sum().clamp(min=eps)

            if loss_type == "attention_kl":
                instance_losses.append(
                    (teacher_obj * (teacher_obj.log() - student_obj.log())).sum()
                )
            elif loss_type == "attention_l2":
                instance_losses.append(F.mse_loss(student_obj, teacher_obj, reduction="mean"))
            else:
                raise ValueError(f"Unknown distillation loss_type={loss_type!r}")

        if not instance_losses:
            continue
        layer_loss = torch.stack(instance_losses).mean()
        layer_losses.append(layer_loss)
        diagnostics["loss_by_layer"][layer_name] = float(layer_loss.detach().cpu().item())

    diagnostics["skipped_layers_shape"] = len(diagnostics["skipped_layer_shape_names"])
    diagnostics["matched_layers"] = len(layer_losses)
    if not layer_losses:
        return _zero_loss_like(student_attention_maps, boxes_xyxy_norm), diagnostics
    return torch.stack(layer_losses).mean(), diagnostics

