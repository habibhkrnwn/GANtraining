from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    """Build an RGB overlay image for quick qualitative checks."""
    image_2d = image.squeeze().astype(np.float32)
    mask_2d = (mask.squeeze() > 0).astype(np.float32)

    image_norm = image_2d - image_2d.min()
    if image_norm.max() > 0:
        image_norm = image_norm / image_norm.max()

    rgb = np.stack([image_norm, image_norm, image_norm], axis=-1)
    red = np.zeros_like(rgb)
    red[..., 0] = 1.0

    return rgb * (1.0 - alpha * mask_2d[..., None]) + red * (alpha * mask_2d[..., None])


def show_overlay(image: np.ndarray, mask: np.ndarray, title: str = "IMC Overlay") -> None:
    overlay = overlay_mask(image, mask)
    plt.figure(figsize=(8, 4))
    plt.imshow(overlay)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
