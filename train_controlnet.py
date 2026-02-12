"""
Stage-2 training script: ControlNet flow matching with bounding-box conditioning.

Loads a pre-trained stage-1 pipeline (VAE + UNet), freezes both, builds a
ControlNet from the UNet encoder, and trains it with a spatial bbox mask as
conditioning signal.
"""

import os
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from fm_src.pipelines.controlnet_flow_matching_pipeline import (
    ControlNetFlowMatchingPipeline,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="ControlNet Flow Matching Training (stage 2)"
    )

    # Data
    p.add_argument("--train_dir", type=str, default="./v18/train/")
    p.add_argument("--val_dir", type=str, default="./v18/val/")
    p.add_argument(
        "--train_annotations",
        type=str,
        default="./v18/train/annotations.json",
    )
    p.add_argument(
        "--val_annotations",
        type=str,
        default="./v18/val/annotations.json",
    )

    # Stage-1 pipeline (frozen UNet + VAE)
    p.add_argument(
        "--stage1_pipeline_dir",
        type=str,
        required=True,
        help="Path to the stage-1 pipeline folder (contains UNET/ and VAE/)",
    )
    p.add_argument(
        "--vae_weights_override",
        type=str,
        default=None,
        help="Explicit path to VAE weights (overrides auto-detection)",
    )

    # Output
    p.add_argument(
        "--model_dir",
        type=str,
        default="./controlnet_runs/bbox_controlnet/",
    )

    # Training params
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--conditioning_scale", type=float, default=1.0)
    p.add_argument("--conditioning_dropout", type=float, default=0.1)
    p.add_argument("--save_every_n_epochs", type=int, default=10)
    p.add_argument("--sample_steps", type=int, default=50)
    p.add_argument("--t_scale", type=float, default=1000)
    p.add_argument("--patience", type=int, default=None)

    # Resume
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to ControlNet checkpoint to resume from",
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class BBoxConditioningDataset(Dataset):
    """
    Loads ``.npy`` images together with their COCO-format bounding-box
    annotations and returns both the pixel tensor and a binary bbox mask.

    Returns
    -------
    dict
        ``pixel_values``               – (C, 256, 256) normalised to [-1, 1]
        ``conditioning_pixel_values``   – (1, 256, 256) binary bbox mask
    """

    def __init__(
        self,
        root_dir: str,
        annotations_path: str,
        conditioning_dropout: float = 0.0,
    ):
        self.root_dir = root_dir
        self.conditioning_dropout = conditioning_dropout

        # Enumerate .npy files
        self.files = sorted(
            f for f in os.listdir(root_dir) if f.endswith(".npy")
        )
        if not self.files:
            raise RuntimeError(f"No .npy files found in {root_dir}")

        # Parse COCO annotations
        with open(annotations_path, "r") as fh:
            data = json.load(fh)

        id_to_fname: dict[str, str] = {}
        self.img_info: dict[str, dict] = {}
        for img in data["images"]:
            id_to_fname[img["id"]] = img["file_name"]
            self.img_info[img["file_name"]] = {
                "width": img["width"],
                "height": img["height"],
                "boxes": [],
            }

        for annot in data["annotations"]:
            fname = id_to_fname.get(annot["image_id"])
            if fname is not None and fname in self.img_info:
                self.img_info[fname]["boxes"].append(annot["bbox"])

        print(
            f"[BBoxDataset] {len(self.files)} files, "
            f"{len(self.img_info)} annotated images"
        )

    def __len__(self) -> int:
        return len(self.files)

    # ------------------------------------------------------------------ #

    @staticmethod
    def _make_bbox_mask(
        boxes: list, width: int, height: int
    ) -> torch.Tensor:
        """Rasterise ``[x, y, w, h]`` boxes into a binary (1, H, W) mask."""
        mask = torch.zeros(1, height, width)
        for bx, by, bw, bh in boxes:
            x0 = max(0, int(bx))
            y0 = max(0, int(by))
            x1 = min(width, int(bx + bw + 0.5))
            y1 = min(height, int(by + bh + 0.5))
            mask[0, y0:y1, x0:x1] = 1.0
        return mask

    # ------------------------------------------------------------------ #

    def __getitem__(self, idx: int) -> dict:
        fname = self.files[idx]

        # --- image -------------------------------------------------------
        x = np.load(os.path.join(self.root_dir, fname))
        if x.ndim == 2:
            x = x[None, ...]  # (1, H, W)
        x = torch.from_numpy(x).float()

        # --- bbox mask ---------------------------------------------------
        info = self.img_info.get(fname)
        if info is not None:
            w, h = info["width"], info["height"]
            boxes = info["boxes"]
        else:
            _, h, w = x.shape
            boxes = []

        mask = self._make_bbox_mask(boxes, w, h)

        # --- resize to 256×256 ------------------------------------------
        x = F.interpolate(
            x.unsqueeze(0),
            size=(256, 256),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        # Normalise to [-1, 1]
        x = x - x.min()
        x = x / (x.max() + 1e-8)
        x = 2 * x - 1

        mask = F.interpolate(
            mask.unsqueeze(0), size=(256, 256), mode="nearest"
        ).squeeze(0)

        # --- conditioning dropout ----------------------------------------
        if (
            self.conditioning_dropout > 0
            and torch.rand(()).item() < self.conditioning_dropout
        ):
            mask = torch.zeros_like(mask)

        return {
            "pixel_values": x,
            "conditioning_pixel_values": mask,
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- datasets --------------------------------------------------------
    train_dataset = BBoxConditioningDataset(
        args.train_dir,
        args.train_annotations,
        conditioning_dropout=args.conditioning_dropout,
    )
    val_dataset = BBoxConditioningDataset(
        args.val_dir,
        args.val_annotations,
        conditioning_dropout=0.0,  # no dropout for evaluation
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # --- pipeline --------------------------------------------------------
    pipe = ControlNetFlowMatchingPipeline(
        device=device,
        t_scale=args.t_scale,
        model_dir=args.model_dir,
    )

    # Load frozen stage-1 models (VAE + UNet)
    pipe.load_from_pipeline_folder_auto(
        args.stage1_pipeline_dir,
        set_eval=True,
    )

    # Optionally override VAE weights
    if args.vae_weights_override is not None:
        pipe.load_vae_weights(args.vae_weights_override)

    # Build ControlNet from the (loaded) UNet
    pipe.build_controlnet(conditioning_channels=1)

    n_cn = sum(p.numel() for p in pipe.controlnet.parameters())
    n_unet = sum(p.numel() for p in pipe.unet.parameters())
    print(f"[ControlNet] Trainable params : {n_cn:,}")
    print(f"[UNet]       Frozen params    : {n_unet:,}")

    # --- train -----------------------------------------------------------
    pipe.train_controlnet_flow_matching(
        dataloader=train_loader,
        epochs=args.epochs,
        eval_dataloader=val_loader,
        lr=args.lr,
        conditioning_scale=args.conditioning_scale,
        log_dir=os.path.join(args.model_dir, "runs", "controlnet_logs"),
        sample_steps=args.sample_steps,
        patience=args.patience,
        save_every_n_epochs=args.save_every_n_epochs,
        resume_from_checkpoint=args.resume,
    )


if __name__ == "__main__":
    main()
