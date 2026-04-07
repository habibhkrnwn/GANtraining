from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.model.diffusion import CarotidDiffusionAugmentor, DiffusionConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Carotid Diffusion Augmentor")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    diffusion_cfg = DiffusionConfig(
        model_id=config["diffusion"]["base_model"],
        use_controlnet=bool(config["diffusion"]["use_controlnet"]),
        learning_rate=float(config["training"]["learning_rate"]),
    )

    model = CarotidDiffusionAugmentor(diffusion_cfg)
    model.build_pipeline()


if __name__ == "__main__":
    main()
