import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


from pipelines.flow_matching_pipeline import FlowMatchingPipeline


class NPYImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.files = sorted(
            f for f in os.listdir(root_dir) if f.endswith(".npy")
        )

        if len(self.files) == 0:
            raise RuntimeError("No .npy files found")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x = np.load(os.path.join(self.root_dir, self.files[idx]))

        if x.ndim == 2:
            x = x[None, ...]  # (1,H,W)

        x = torch.from_numpy(x).float()

        if self.transform:
            x = self.transform(x)

        return x


def _center_crop_square(x: torch.Tensor) -> torch.Tensor:
    _, h, w = x.shape
    crop = min(h, w)
    top = (h - crop) // 2
    left = (w - crop) // 2
    return x[:, top : top + crop, left : left + crop]


def _rotate_90(x: torch.Tensor, k: int) -> torch.Tensor:
    return torch.rot90(x, k, dims=(1, 2))


def _random_rotate_90(x: torch.Tensor) -> torch.Tensor:
    k = int(torch.randint(0, 4, (1,)).item())
    return _rotate_90(x, k)


def _save_tensor_image(x: torch.Tensor, out_base: str) -> None:
    x = x.detach().cpu()

    if x.min() < 0:
        x = (x + 1) / 2
    else:
        x = x - x.min()
        x = x / (x.max() + 1e-8)

    x = x.clamp(0, 1)
    x = (x * 255).byte()

    if x.shape[0] == 1:
        img = x[0].numpy()
    else:
        img = x.permute(1, 2, 0).numpy()

    np.save(f"{out_base}.npy", img)
    try:
        from PIL import Image

        Image.fromarray(img).save(f"{out_base}.png")
    except Exception:
        pass


def _resize_and_normalize_256(x: torch.Tensor) -> torch.Tensor:
    # resize
    x = F.interpolate(
        x.unsqueeze(0),
        size=(256, 256),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    # normalize to [-1,1] (simple per-image)
    x = x - x.min()
    x = x / (x.max() + 1e-8)
    x = 2 * x - 1

    return x


class ScheduledAugment256:
    def __init__(
        self,
        *,
        total_epochs: int,
        warmup_frac: float = 0.1,
        ramp_frac: float = 0.3,
        p_crop_warmup: float = 0.05,
        p_crop_max: float = 0.20,
        p_crop_final: float = 0.05,
        p_rot_warmup: float = 0.05,
        p_rot_max: float = 0.30,
        p_rot_final: float = 0.05,
    ):
        self.total_epochs = int(total_epochs)
        self.warmup_frac = float(warmup_frac)
        self.ramp_frac = float(ramp_frac)

        self.p_crop_warmup = float(p_crop_warmup)
        self.p_crop_max = float(p_crop_max)
        self.p_crop_final = float(p_crop_final)
        self.p_rot_warmup = float(p_rot_warmup)
        self.p_rot_max = float(p_rot_max)
        self.p_rot_final = float(p_rot_final)
        self.epoch = 0
        self._last_phase = None

        self._log_probs(phase="start")

    def _log_probs(self, *, phase: str) -> None:
        p_crop, p_rot = self._get_probs()
        print(
            f"[AugmentSchedule] {phase}: epoch={self.epoch} "
            f"p_crop={p_crop:.3f} p_rot={p_rot:.3f}"
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        phase = self._get_phase()
        if phase != self._last_phase:
            self._log_probs(phase=phase)
            self._last_phase = phase

    def _get_phase(self) -> str:
        warmup_end = int(self.total_epochs * self.warmup_frac)
        ramp_end = int(self.total_epochs * (self.warmup_frac + self.ramp_frac))
        if self.epoch < warmup_end:
            return "warmup"
        if self.epoch < ramp_end:
            return "ramp"
        return "decay"

    def _get_probs(self) -> tuple[float, float]:
        warmup_end = int(self.total_epochs * self.warmup_frac)
        ramp_end = int(self.total_epochs * (self.warmup_frac + self.ramp_frac))

        if self.epoch < warmup_end:
            return self.p_crop_warmup, self.p_rot_warmup

        if self.epoch < ramp_end:
            denom = max(1, ramp_end - warmup_end)
            alpha = (self.epoch - warmup_end) / denom
            p_crop = self.p_crop_warmup + alpha * (self.p_crop_max - self.p_crop_warmup)
            p_rot = self.p_rot_warmup + alpha * (self.p_rot_max - self.p_rot_warmup)
            return p_crop, p_rot

        denom = max(1, self.total_epochs - ramp_end)
        alpha = (self.epoch - ramp_end) / denom
        p_crop = self.p_crop_max + alpha * (self.p_crop_final - self.p_crop_max)
        p_rot = self.p_rot_max + alpha * (self.p_rot_final - self.p_rot_max)
        return p_crop, p_rot

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        p_crop, p_rot = self._get_probs()
        if torch.rand(()) < p_crop:
            x = _center_crop_square(x)
        if torch.rand(()) < p_rot:
            x = _random_rotate_90(x)
        return _resize_and_normalize_256(x)


def save_transform_examples(dataset: Dataset, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    x = dataset[0]
    if x.ndim == 2:
        x = x.unsqueeze(0)

    x_crop = _center_crop_square(x)
    x_rot = _rotate_90(x_crop, k=1)

    transform = getattr(dataset, "transform", None)
    if transform is not None and hasattr(transform, "set_epoch"):
        transform.set_epoch(0)
    x_final = transform(x) if transform is not None else _resize_and_normalize_256(x)

    _save_tensor_image(x_crop, os.path.join(out_dir, "example_center_crop"))
    _save_tensor_image(x_rot, os.path.join(out_dir, "example_rotate_90"))
    _save_tensor_image(x_final, os.path.join(out_dir, "example_final"))

def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    total_epochs = 10
    train_transform = ScheduledAugment256(
        total_epochs=total_epochs,
        warmup_frac=0.1,
        ramp_frac=0.3,
        p_crop_warmup=0.05,
        p_crop_max=0.20,
        p_crop_final=0.05,
        p_rot_warmup=0.05,
        p_rot_max=0.30,
        p_rot_final=0.05,
    )
    eval_transform = ScheduledAugment256(
        total_epochs=total_epochs,
        warmup_frac=0.1,
        ramp_frac=0.3,
        p_crop_warmup=0.05,
        p_crop_max=0.20,
        p_crop_final=0.05,
        p_rot_warmup=0.05,
        p_rot_max=0.30,
        p_rot_final=0.05,
    )
    train_dataset = NPYImageDataset(
        root_dir="./v18/train/",
        transform=train_transform,
    )
    eval_dataset = NPYImageDataset(
        root_dir="./v18/val/",
        transform=eval_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    pipe = FlowMatchingPipeline(
        device=device,
        t_scale=1000,
        model_dir="./non_stable_training_t_scaled/",
        sample_shape=(1, 256, 256),
    ).build_from_configs(
        unet_json="non_stable_unet_config.json",
    )

    save_transform_examples(train_dataset, os.path.join(pipe.model_dir, "transform_examples"))

    pipe.train_flow_matching(
        dataloader=train_loader,
        eval_dataloader=eval_loader,
        epochs=total_epochs,
        save_every_n_epochs=1,
        log_dir=f"{pipe.model_dir}/runs/non_stable_flow_matching_logs/",
        sample_batch_size=4,
    )
    
if __name__ == "__main__":
    main()