from __future__ import annotations

import torch
from monai.networks.nets import UNet
from torch import nn


def build_unet_baseline(in_channels: int = 1, out_channels: int = 1) -> nn.Module:
    """Baseline U-Net for IMC segmentation."""
    return UNet(
        spatial_dims=2,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )


class SAMAUNet(nn.Module):
    """Placeholder for SAMA-UNet implementation used in experiments."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1) -> None:
        super().__init__()
        self.backbone = build_unet_baseline(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
