import argparse
import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from best_nvidia_gpu import get_least_used_cuda_gpu
from fm_src.pipelines.flow_matching_pipeline import StableFlowMatchingPipeline


P0001_PERCENTILE_RAW_IMAGES = 11667.0  # p0.001 percentile
P9999_PERCENTILE_RAW_IMAGES = 13944.0  # p99.999 percentile
A = P0001_PERCENTILE_RAW_IMAGES
B = P9999_PERCENTILE_RAW_IMAGES

S = B - A


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
    
# def to_sd_tensor_and_x(x: torch.Tensor) -> torch.Tensor:
#     # Full linear normalization: uint16 [0, 65535] -> [-1, 1]
#     return (x.to(torch.float32) / 65535.0) * 2 - 1


# def from_norm_to_display(recon: torch.Tensor) -> torch.Tensor:
#     # Reverse: [-1, 1] -> [0, 1] for display / TensorBoard
#     return (recon + 1) / 2


# def from_norm_to_uint16(recon: torch.Tensor) -> torch.Tensor:
#     # Reverse: [-1, 1] -> uint16 [0, 65535] for saving
#     return ((recon + 1) / 2) * 65535.0


    
def to_sd_tensor_and_x(x: torch.Tensor) -> torch.Tensor:
    # Full linear normalization: uint16 [0, 65535] -> [-1, 1]
    return torch.clamp((x.to(torch.float32) - A) / S, 0, 1) * 2 - 1

def from_norm_to_display(recon: torch.Tensor) -> torch.Tensor:
    # Reverse: [-1, 1] -> [0, 1] for display / TensorBoard
    return (recon + 1) / 2


def from_norm_to_uint16(recon: torch.Tensor) -> torch.Tensor:
    # Reverse: [-1, 1] -> uint16 [0, 65535] for saving
    return ((recon + 1) / 2) * S + A




def _resize_and_normalize(x: torch.Tensor, size: int) -> torch.Tensor:
    # Resize
    x = F.interpolate(
        x.unsqueeze(0),
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    # Normalize to [-1, 1] (simple per-image)
    x = to_sd_tensor_and_x(x)

    return x


def build_dataloader(
    root_dir: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    shuffle: bool,
) -> DataLoader:
    dataset = NPYImageDataset(
        root_dir=root_dir,
        transform=lambda x: _resize_and_normalize(x, image_size),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VAE on NPY dataset.")
    parser.add_argument("--train-dir", type=str, required=True, help="Path to train .npy files.")
    parser.add_argument("--val-dir", type=str, default=None, help="Optional path to val .npy files.")
    parser.add_argument("--image-size", type=int, default=256, help="Square image size.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--pin-memory", action="store_true", help="Enable DataLoader pin_memory.")
    parser.add_argument("--no-pin-memory", action="store_false", dest="pin_memory", help="Disable DataLoader pin_memory.")
    parser.set_defaults(pin_memory=True)
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs.")
    parser.add_argument("--t-scale", type=int, default=1000, help="Pipeline t_scale.")
    parser.add_argument("--model-dir", type=str, default="vae_runs/vae_fm_x8", help="VAE model dir.")
    parser.add_argument("--vae-json", type=str, default="fm_src/vae_config.json", help="VAE config JSON.")
    parser.add_argument("--log-dir", type=str, default="./runs/autoencoder_kl", help="TensorBoard log dir.")
    parser.add_argument("--patience", type=int, default=4, help="Early-stopping patience.")
    parser.add_argument("--min-delta", type=float, default=0.0, help="Early-stopping min delta.")
    parser.add_argument("--gpu-prefer", type=str, default="memory", help="GPU selection preference.")
    parser.add_argument("--min-free-mb", type=int, default=4096, help="Minimum free GPU memory.")
    return parser.parse_args()


def resolve_pin_memory(args: argparse.Namespace) -> bool:
    return args.pin_memory


def build_eval_loader(
    val_dir: Optional[str],
    image_size: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> Optional[DataLoader]:
    if not val_dir:
        return None
    return build_dataloader(
        root_dir=val_dir,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )


def main() -> None:
    args = parse_args()
    pin_memory = resolve_pin_memory(args)

    train_loader = build_dataloader(
        root_dir=args.train_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )
    eval_loader = build_eval_loader(
        val_dir=args.val_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    device, smi_out = get_least_used_cuda_gpu(
        prefer=args.gpu_prefer,
        min_free_mb=args.min_free_mb,
        return_type="torch",
    )
    print(f"Using device: {device}, GPU info:\n{smi_out}")

    pipeline = StableFlowMatchingPipeline(
        device=device,
        t_scale=args.t_scale,
        model_dir=args.model_dir,
        from_norm_to_display=from_norm_to_display,
    ).build_from_configs(
        vae_json=args.vae_json,
    )

    pipeline.train_vae(
        dataloader=train_loader,
        epochs=args.epochs,
        eval_dataloader=eval_loader,
        log_dir=f"{pipeline.model_dir}/runs/autoencoder_kl",
        patience=args.patience,
        min_delta=args.min_delta,
    )


if __name__ == "__main__":
    main()