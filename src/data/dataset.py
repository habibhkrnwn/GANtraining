from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.utils.io import read_tiff


class CUBSDataset(Dataset[dict[str, Any]]):
    """PyTorch dataset for CUBS image-mask pairs and metadata."""

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        image_size: tuple[int, int] = (256, 256),
        normalize_to_minus1_1: bool = True,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.records = list(records)
        self.image_size = image_size
        self.normalize_to_minus1_1 = normalize_to_minus1_1
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records[index]

        image = read_tiff(Path(row["image_path"]))
        mask = read_tiff(Path(row["mask_path"]))

        image = self._prepare_image(image)
        mask = self._prepare_mask(mask)

        sample: dict[str, Any] = {
            "image": image,
            "mask": mask,
            "center_id": row.get("center_id", "unknown"),
            "snr_label": row.get("snr_label", "unknown"),
            "split": row.get("split", "train"),
            "is_thin_imc": bool(row.get("is_thin_imc", False)),
            "is_ambiguous": bool(row.get("is_ambiguous", False)),
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def _prepare_image(self, image: np.ndarray) -> torch.Tensor:
        if image.ndim == 3:
            image = image.squeeze()

        image = self._resize_array(image, is_mask=False)
        image = image.astype(np.float32)

        image_min = float(np.min(image))
        image_max = float(np.max(image))
        if image_max > image_min:
            image = (image - image_min) / (image_max - image_min)

        if self.normalize_to_minus1_1:
            image = image * 2.0 - 1.0

        return torch.from_numpy(image).unsqueeze(0)

    def _prepare_mask(self, mask: np.ndarray) -> torch.Tensor:
        if mask.ndim == 3:
            mask = mask.squeeze()

        mask = self._resize_array(mask, is_mask=True)
        mask = (mask > 0).astype(np.float32)
        return torch.from_numpy(mask).unsqueeze(0)

    def _resize_array(self, arr: np.ndarray, is_mask: bool) -> np.ndarray:
        pil = Image.fromarray(arr.astype(np.float32))
        interpolation = Image.NEAREST if is_mask else Image.BILINEAR
        resized = pil.resize((self.image_size[1], self.image_size[0]), interpolation)
        return np.asarray(resized, dtype=np.float32)
