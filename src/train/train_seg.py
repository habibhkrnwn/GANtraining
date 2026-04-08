from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import CUBSProcessedDataset
from src.model.segmentation import SAMAUNet, build_unet_baseline
from src.utils.run_report import append_markdown_run_report
from src.utils.visualize import overlay_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train segmentation model")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--model", choices=["unet", "sama_unet"], default="sama_unet")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-train-samples", type=int, default=256)
    parser.add_argument("--max-val-samples", type=int, default=64)
    parser.add_argument("--num-visuals", type=int, default=12)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--pos-weight", type=float, default=50.0)
    parser.add_argument("--loss-alpha", type=float, default=0.5, help="alpha for BCE in combined loss")
    return parser.parse_args()


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _maybe_subset(dataset: CUBSProcessedDataset, max_samples: int | None) -> CUBSProcessedDataset | Subset:
    if max_samples is None or max_samples <= 0 or len(dataset) <= max_samples:
        return dataset
    return Subset(dataset, list(range(max_samples)))


def _compute_batch_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-7,
) -> tuple[float, float, list[float]]:
    """Compute binary Dice/IoU for 1-channel logits [B,1,H,W]."""
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    targets_bin = (targets > 0.5).float()

    tp = (preds * targets_bin).sum(dim=(1, 2, 3))
    fp = (preds * (1.0 - targets_bin)).sum(dim=(1, 2, 3))
    fn = ((1.0 - preds) * targets_bin).sum(dim=(1, 2, 3))

    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)

    return float(dice.mean().item()), float(iou.mean().item()), [float(v) for v in dice.detach().cpu().tolist()]


def combined_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor,
    alpha: float = 0.5,
    eps: float = 1e-7,
) -> torch.Tensor:
    """BCE + soft Dice loss for thin-foreground binary segmentation."""
    alpha = float(max(0.0, min(1.0, alpha)))
    targets_bin = (targets > 0.5).float()

    bce = F.binary_cross_entropy_with_logits(logits, targets_bin, pos_weight=pos_weight)

    probs = torch.sigmoid(logits)
    tp = (probs * targets_bin).sum(dim=(1, 2, 3))
    fp = (probs * (1.0 - targets_bin)).sum(dim=(1, 2, 3))
    fn = ((1.0 - probs) * targets_bin).sum(dim=(1, 2, 3))
    soft_dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    dice_loss = 1.0 - soft_dice.mean()

    return alpha * bce + (1.0 - alpha) * dice_loss


def _save_prediction_visuals(
    model: nn.Module,
    loader: DataLoader,
    output_dir: Path,
    device: torch.device,
    max_items: int,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    saved = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            sample_ids = batch["sample_id"]

            probs = torch.sigmoid(model(images))
            pred_masks = (probs >= 0.5).float()

            images_np = images.detach().cpu().numpy()
            masks_np = masks.detach().cpu().numpy()
            preds_np = pred_masks.detach().cpu().numpy()

            for i in range(images_np.shape[0]):
                image_2d = images_np[i, 0]
                gt_2d = masks_np[i, 0]
                pred_2d = preds_np[i, 0]

                gt_overlay = overlay_mask(image_2d, gt_2d)
                pred_overlay = overlay_mask(image_2d, pred_2d)

                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(image_2d, cmap="gray")
                axes[0].set_title("Image")
                axes[0].axis("off")
                axes[1].imshow(gt_overlay)
                axes[1].set_title("GT Overlay")
                axes[1].axis("off")
                axes[2].imshow(pred_overlay)
                axes[2].set_title("Pred Overlay")
                axes[2].axis("off")
                plt.tight_layout()

                sample_id = str(sample_ids[i])
                fig.savefig(output_dir / f"{saved + 1:03d}_{sample_id}.png", dpi=150)
                plt.close(fig)

                saved += 1
                if saved >= max_items:
                    return saved
    return saved


def _run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: Any,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    debug_once: bool = False,
    debug_tag: str = "",
    debug_logs: list[str] | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    steps = 0

    debug_printed = False
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, masks)

        if is_train:
            loss.backward()
            optimizer.step()

        dice_val, iou_val, dice_per_sample = _compute_batch_metrics(logits, masks)

        if debug_once and not debug_printed:
            mask_min = float(masks.min().item())
            mask_max = float(masks.max().item())
            mask_unique = sorted(float(v) for v in torch.unique(masks.detach().cpu()).tolist())
            debug_msg = (
                f"[DEBUG {debug_tag}] logits_shape={tuple(logits.shape)} mask_min={mask_min:.4f} "
                f"mask_max={mask_max:.4f} mask_unique={mask_unique} "
                f"dice_per_sample={[round(v, 4) for v in dice_per_sample]}"
            )
            print(debug_msg)
            if debug_logs is not None:
                debug_logs.append(debug_msg)
            debug_printed = True

        total_loss += float(loss.item())
        total_dice += dice_val
        total_iou += iou_val
        steps += 1

    if steps == 0:
        return {"loss": 0.0, "dice": 0.0, "iou": 0.0}

    return {
        "loss": total_loss / steps,
        "dice": total_dice / steps,
        "iou": total_iou / steps,
    }


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    paths_cfg = config.get("paths", {})
    training_cfg = config.get("training", {})

    in_channels = int(config["segmentation"]["in_channels"])
    out_channels = int(config["segmentation"]["out_channels"])

    if args.model == "unet":
        model = build_unet_baseline(in_channels=in_channels, out_channels=out_channels)
    else:
        model = SAMAUNet(in_channels=in_channels, out_channels=out_channels)

    device = _resolve_device(args.device)
    model = model.to(device)

    processed_root = Path(paths_cfg.get("processed_root", "data/processed"))
    metadata_csv = processed_root / "metadata.csv"

    train_ds = CUBSProcessedDataset(metadata_csv=metadata_csv, processed_root=processed_root, split="train", augment=True)
    val_ds = CUBSProcessedDataset(metadata_csv=metadata_csv, processed_root=processed_root, split="val", augment=False)

    train_ds = _maybe_subset(train_ds, args.max_train_samples)
    val_ds = _maybe_subset(val_ds, args.max_val_samples)

    batch_size = int(args.batch_size or training_cfg.get("batch_size", 4))
    num_workers = int(args.num_workers if args.num_workers is not None else training_cfg.get("num_workers", 0))
    learning_rate = float(args.learning_rate or training_cfg.get("learning_rate", 1e-4))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if len(train_loader.dataset) == 0:
        raise RuntimeError("Train split is empty. Regenerate metadata.csv before training.")

    pos_weight = torch.tensor([50.0], device=device)
    loss_alpha = float(args.loss_alpha)

    def criterion(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return combined_loss(
            logits=logits,
            targets=targets,
            pos_weight=pos_weight,
            alpha=loss_alpha,
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(paths_cfg.get("seg_sanity_dir", "outputs/seg_sanity")) / run_stamp
    output_root.mkdir(parents=True, exist_ok=True)
    visuals_dir = output_root / "visuals"

    history: list[dict[str, Any]] = []
    debug_logs: list[str] = []
    for epoch in range(1, int(args.epochs) + 1):
        train_metrics = _run_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            debug_once=False,
            debug_tag="train",
            debug_logs=debug_logs,
        )
        val_metrics = _run_one_epoch(
            model,
            val_loader,
            criterion,
            None,
            device,
            debug_once=(epoch == 1),
            debug_tag="val",
            debug_logs=debug_logs,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_dice": train_metrics["dice"],
            "train_iou": train_metrics["iou"],
            "val_loss": val_metrics["loss"],
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
        }
        history.append(row)
        print(
            "Epoch {epoch}/{total} | train loss={tl:.4f} dice={td:.4f} iou={ti:.4f} | "
            "val loss={vl:.4f} dice={vd:.4f} iou={vi:.4f}".format(
                epoch=epoch,
                total=int(args.epochs),
                tl=row["train_loss"],
                td=row["train_dice"],
                ti=row["train_iou"],
                vl=row["val_loss"],
                vd=row["val_dice"],
                vi=row["val_iou"],
            )
        )

    history_df = pd.DataFrame(history)
    history_csv = output_root / "history.csv"
    history_df.to_csv(history_csv, index=False)

    checkpoint_path = output_root / "checkpoint_last.pt"
    torch.save(
        {
            "model": args.model,
            "state_dict": model.state_dict(),
            "history": history,
            "config": config,
        },
        checkpoint_path,
    )

    visual_loader = val_loader if len(val_loader.dataset) > 0 else train_loader
    visuals_saved = _save_prediction_visuals(
        model=model,
        loader=visual_loader,
        output_dir=visuals_dir,
        device=device,
        max_items=max(1, int(args.num_visuals)),
    )

    report_path = append_markdown_run_report(
        report_path=Path(paths_cfg.get("run_report_md", "reports/run_tracking.md")),
        stage="segmentation_sanity_train",
        summary={
            "model": args.model,
            "device": str(device),
            "loss": "bce_softdice",
            "pos_weight": float(args.pos_weight),
            "loss_alpha": loss_alpha,
            "epochs": int(args.epochs),
            "train_samples": len(train_loader.dataset),
            "val_samples": len(val_loader.dataset),
            "batch_size": batch_size,
            "visuals_saved": visuals_saved,
            "best_val_dice": float(history_df["val_dice"].max()) if not history_df.empty else 0.0,
            "last_val_dice": float(history_df["val_dice"].iloc[-1]) if not history_df.empty else 0.0,
        },
        details={
            "output_root": str(output_root),
            "history_csv": str(history_csv),
            "checkpoint": str(checkpoint_path),
            "visuals_dir": str(visuals_dir),
            "debug_batch": " | ".join(debug_logs) if debug_logs else "n/a",
        },
    )

    print(f"Saved training history: {history_csv}")
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved visuals: {visuals_saved} -> {visuals_dir}")
    print(f"Saved run report: {report_path}")


if __name__ == "__main__":
    main()
