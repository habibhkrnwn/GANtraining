from __future__ import annotations

from monai.transforms import Compose, RandFlipd, RandRotate90d, RandZoomd


def get_classical_augmentations() -> Compose:
    """Return baseline augmentation pipeline for image-mask dictionaries."""
    return Compose(
        [
            RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=1),
            RandRotate90d(keys=["image", "mask"], prob=0.3, max_k=3),
            RandZoomd(
                keys=["image", "mask"],
                prob=0.3,
                min_zoom=0.9,
                max_zoom=1.1,
                mode=("bilinear", "nearest"),
            ),
        ]
    )
