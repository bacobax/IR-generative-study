import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


from fm_src.pipelines.flow_matching_pipeline import StableFlowMatchingPipeline

def parse_args():
    parser = argparse.ArgumentParser(description="Stable Flow Matching Training")
    
    # Data paths
    parser.add_argument("--train_dir", type=str, default="./v18/train/", help="Path to training data")
    parser.add_argument("--val_dir", type=str, default="./v18/val/", help="Path to validation data")
    
    # Model configs
    parser.add_argument("--unet_config", type=str, default="stable_unet_config.json", help="UNet config JSON")
    parser.add_argument("--vae_config", type=str, default="vae_config.json", help="VAE config JSON")
    parser.add_argument("--vae_weights", type=str, default="./vae_best.pt", help="Pretrained VAE weights")
    
    # Output
    parser.add_argument("--model_dir", type=str, default="./serious_runs/stable_training_t_scaled/", help="Model output directory")
    
    # Training params
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--save_every_n_epochs", type=int, default=10, help="Save checkpoint every N epochs")
    parser.add_argument("--sample_batch_size", type=int, default=4, help="Batch size for sampling")
    parser.add_argument("--t_scale", type=float, default=1000, help="Time scale for UNet")
    
    # Augmentation schedule
    parser.add_argument("--warmup_frac", type=float, default=0.1, help="Warmup fraction of epochs")
    parser.add_argument("--ramp_frac", type=float, default=0.3, help="Ramp fraction of epochs")
    parser.add_argument("--p_crop_warmup", type=float, default=0.05)
    parser.add_argument("--p_crop_max", type=float, default=0.20)
    parser.add_argument("--p_crop_final", type=float, default=0.05)
    parser.add_argument("--p_rot_warmup", type=float, default=0.05)
    parser.add_argument("--p_rot_max", type=float, default=0.30)
    parser.add_argument("--p_rot_final", type=float, default=0.05)
    
    # Resume
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from (e.g. UNET/unet_fm_epoch_10_ckpt.pt)")
    
    return parser.parse_args()


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
    """Save a tensor image as .npy (uint16) and .png (uint8 grayscale).

    Handles both raw uint16 tensors and [-1, 1] normalised tensors.
    """
    x = x.detach().cpu().float()

    if x.min() < 0:  # already in [-1, 1]
        x_01 = ((x + 1) / 2).clamp(0, 1)
    else:  # raw uint16 values
        x_01 = (x / 65535.0).clamp(0, 1)

    # .npy in uint16 domain
    x_uint16 = (x_01 * 65535.0).numpy().astype(np.uint16)
    if x_uint16.ndim == 3 and x_uint16.shape[0] == 1:
        x_uint16 = x_uint16[0]
    np.save(f"{out_base}.npy", x_uint16)

    # .png in uint8 for visualisation
    x_uint8 = (x_01 * 255).clamp(0, 255).byte()
    if x_uint8.shape[0] == 1:
        img = x_uint8[0].numpy()
    else:
        img = x_uint8.permute(1, 2, 0).numpy()
    try:
        from PIL import Image
        Image.fromarray(img).save(f"{out_base}.png")
    except Exception:
        pass
    


# Full linear normalization: uint16 [0, 65535] → [-1, 1]
to_sd_tensor_and_x = lambda x: (x.to(torch.float32) / 65535.0) * 2 - 1

# Reverse: [-1, 1] → [0, 1] for display / TensorBoard
from_norm_to_display = lambda recon: (recon + 1) / 2

# Reverse: [-1, 1] → uint16 [0, 65535] for saving
from_norm_to_uint16 = lambda recon: ((recon + 1) / 2) * 65535.0




def _resize_and_normalize_256(x: torch.Tensor) -> torch.Tensor:
    # resize
    x = F.interpolate(
        x.unsqueeze(0),
        size=(256, 256),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    # normalize to [-1,1] (simple per-image)
    x = to_sd_tensor_and_x(x)

    return x


class ScheduledAugment256:
    def __init__(
        self,
        *,
        total_epochs: int,
        warmup_frac: float = 0.15,
        ramp_frac: float = 0.4,
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
    args = parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    total_epochs = args.epochs
    train_transform = ScheduledAugment256(
        total_epochs=total_epochs,
        warmup_frac=args.warmup_frac,
        ramp_frac=args.ramp_frac,
        p_crop_warmup=args.p_crop_warmup,
        p_crop_max=args.p_crop_max,
        p_crop_final=args.p_crop_final,
        p_rot_warmup=args.p_rot_warmup,
        p_rot_max=args.p_rot_max,
        p_rot_final=args.p_rot_final,
    )
    eval_transform = ScheduledAugment256(
        total_epochs=total_epochs,
        warmup_frac=args.warmup_frac,
        ramp_frac=args.ramp_frac,
        p_crop_warmup=args.p_crop_warmup,
        p_crop_max=args.p_crop_max,
        p_crop_final=args.p_crop_final,
        p_rot_warmup=args.p_rot_warmup,
        p_rot_max=args.p_rot_max,
        p_rot_final=args.p_rot_final,
    )
    train_dataset = NPYImageDataset(
        root_dir=args.train_dir,
        transform=train_transform,
    )
    eval_dataset = NPYImageDataset(
        root_dir=args.val_dir,
        transform=eval_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    pipe = StableFlowMatchingPipeline(
        device=device,
        t_scale=args.t_scale,
        model_dir=args.model_dir,
    ).build_from_configs(
        unet_json=args.unet_config,
        vae_json=args.vae_config,
    )

    if args.resume is None:
        save_transform_examples(train_dataset, os.path.join(pipe.model_dir, "transform_examples"))

    pipe.train_flow_matching(
        dataloader=train_loader,
        epochs=total_epochs,
        eval_dataloader=eval_loader,
        pretrained_vae_path=args.vae_weights,
        save_every_n_epochs=args.save_every_n_epochs,
        log_dir=f"{pipe.model_dir}/runs/stable_flow_matching_logs/",
        sample_batch_size=args.sample_batch_size,
        resume_from_checkpoint=args.resume,
    )
    
if __name__ == "__main__":
    main()