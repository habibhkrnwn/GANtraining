from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.model.segmentation import SAMAUNet, build_unet_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train segmentation model")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--model", choices=["unet", "sama_unet"], default="sama_unet")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    in_channels = int(config["segmentation"]["in_channels"])
    out_channels = int(config["segmentation"]["out_channels"])

    if args.model == "unet":
        model = build_unet_baseline(in_channels=in_channels, out_channels=out_channels)
    else:
        model = SAMAUNet(in_channels=in_channels, out_channels=out_channels)

    _ = model


if __name__ == "__main__":
    main()
