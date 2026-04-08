from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from src.data.dataset import CUBSProcessedDataset
from src.utils.run_report import append_markdown_run_report


def main(config_path: str, split: str = "train", batch_size: int = 2, augment: bool = True) -> None:
    with Path(config_path).open("r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    paths_cfg = cfg.get("paths", {})
    processed_root = Path(paths_cfg.get("processed_root", "data/processed"))
    report_md = Path(paths_cfg.get("run_report_md", "reports/run_tracking.md"))

    metadata_csv = processed_root / "metadata.csv"
    dataset = CUBSProcessedDataset(
        metadata_csv=metadata_csv,
        processed_root=processed_root,
        split=split,
        augment=augment,
    )

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    batch = next(iter(loader))

    image_shape = tuple(batch["image"].shape)
    mask_shape = tuple(batch["mask"].shape)
    hard_thin_unique = sorted(int(v) for v in torch.unique(batch["hard_thin"]).tolist())
    hard_thin_ok = set(hard_thin_unique).issubset({0, 1})

    summary = {
        "split": split,
        "augment": augment,
        "dataset_len": len(dataset),
        "batch_size": batch_size,
        "image_shape": image_shape,
        "mask_shape": mask_shape,
        "hard_thin_unique": hard_thin_unique,
        "hard_thin_is_binary": hard_thin_ok,
    }
    details = {
        "metadata_csv": str(metadata_csv),
        "processed_root": str(processed_root),
        "example_imt_mm": [round(float(v), 6) for v in batch["imt_mm"][:5].tolist()],
        "example_hard_thin": [int(v) for v in batch["hard_thin"][:5].tolist()],
    }

    report_path = append_markdown_run_report(
        report_path=report_md,
        stage="dataloader_check",
        summary=summary,
        details=details,
    )

    print(f"len {split} = {len(dataset)}")
    print(image_shape, mask_shape, batch["imt_mm"][:5], batch["hard_thin"][:5])
    print(f"Saved run report to: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DataLoader sanity check and log results to markdown")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--augment", action="store_true", default=False)
    args = parser.parse_args()

    main(
        config_path=args.config,
        split=args.split,
        batch_size=args.batch_size,
        augment=args.augment,
    )
