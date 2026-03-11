import os
import argparse
from best_nvidia_gpu import get_least_used_cuda_gpu
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.core.constants import P0001_PERCENTILE_RAW_IMAGES, P9999_PERCENTILE_RAW_IMAGES, RAW_RANGE
from src.core.normalization import (
    raw_to_norm as to_sd_tensor_and_x,
    norm_to_display as from_norm_to_display,
    norm_to_uint16 as from_norm_to_uint16,
    resize_and_normalize_256 as _resize_and_normalize_256,
)
from src.core.data.datasets import NPYImageDataset
from src.core.data.transforms import (
    center_crop_square as _center_crop_square,
    rotate_90 as _rotate_90,
    random_rotate_90 as _random_rotate_90,
    save_tensor_image as _save_tensor_image,
    ScheduledAugment256,
    save_transform_examples,
)
from fm_src.pipelines.flow_matching_pipeline import FlowMatchingPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Flow Matching Training (pixel space, no VAE)")

    # Data paths
    parser.add_argument("--train_dir", type=str, default="./data/raw/v18/train/", help="Path to training data")
    parser.add_argument("--val_dir", type=str, default="./data/raw/v18/val/", help="Path to validation data")

    # Model configs
    parser.add_argument("--unet_config", type=str, default="configs/models/fm/non_stable_unet_config.json", help="UNet config JSON")

    # Output
    parser.add_argument("--model_dir", type=str, default="./serious_runs/pixel_fm_x0/", help="Model output directory")

    # Training params
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--save_every_n_epochs", type=int, default=10, help="Save checkpoint every N epochs")
    parser.add_argument("--sample_batch_size", type=int, default=4, help="Batch size for sampling")
    parser.add_argument("--t_scale", type=float, default=1000, help="Time scale for UNet")

    # Prediction target
    parser.add_argument("--train-target", type=str, default="v", choices=["v", "x0"],
                        help="Prediction target: 'v' for velocity, 'x0' for clean sample (default: x0)")

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


A = P0001_PERCENTILE_RAW_IMAGES
B = P9999_PERCENTILE_RAW_IMAGES
S = RAW_RANGE


def main():
    args = parse_args()


    device, smi_out = get_least_used_cuda_gpu(
        prefer="memory",
        min_free_mb=0,
        return_type="torch",
    )
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

    pipe = FlowMatchingPipeline(
        device=device,
        t_scale=args.t_scale,
        model_dir=args.model_dir,
        from_norm_to_display=from_norm_to_display,
        train_target=args.train_target,
    ).build_from_configs(
        unet_json=args.unet_config,
    )

    if args.resume is None:
        save_transform_examples(train_dataset, os.path.join(pipe.model_dir, "transform_examples"))

    pipe.train_flow_matching(
        dataloader=train_loader,
        epochs=total_epochs,
        eval_dataloader=eval_loader,
        save_every_n_epochs=args.save_every_n_epochs,
        log_dir=f"{pipe.model_dir}/runs/flow_matching_logs/",
        sample_batch_size=args.sample_batch_size,
        resume_from_checkpoint=args.resume,
    )


if __name__ == "__main__":
    main()
