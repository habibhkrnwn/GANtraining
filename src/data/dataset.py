from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
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


class CUBSProcessedDataset(Dataset[dict[str, Any]]):
    """Load processed CUBS samples from metadata.csv and NPZ files."""

    def __init__(
        self,
        metadata_csv: str | Path,
        processed_root: str | Path,
        split: str | None = None,
        augment: bool = False,
        normalize_to_minus1_1: bool = True,
    ) -> None:
        self.metadata_csv = Path(metadata_csv)
        self.processed_root = Path(processed_root)
        self.split = split
        self.augment = augment
        self.normalize_to_minus1_1 = normalize_to_minus1_1

        if not self.metadata_csv.exists():
            raise FileNotFoundError(f"metadata.csv not found: {self.metadata_csv}")

        df = pd.read_csv(self.metadata_csv)
        if split is not None:
            df = df[df["split"] == split]
        self.records = df.reset_index(drop=True).to_dict(orient="records")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records[index]
        npz_path = self._resolve_npz_path(row)

        with np.load(npz_path) as npz_data:
            image = np.asarray(npz_data["image"], dtype=np.float32)
            mask = np.asarray(npz_data["mask"], dtype=np.float32)

        image = self._normalize_image(image)
        mask = (mask > 0).astype(np.float32)

        image_t = torch.from_numpy(image).unsqueeze(0)
        mask_t = torch.from_numpy(mask).unsqueeze(0)

        if self.augment:
            image_t, mask_t = self._apply_simple_augment(image_t, mask_t)

        sample: dict[str, Any] = {
            "image": image_t,
            "mask": mask_t,
            "imt_mm": torch.tensor(float(row.get("imt_mm", 0.0)), dtype=torch.float32),
            "hard_thin": torch.tensor(self._to_int01(row.get("hard_thin", 0)), dtype=torch.int64),
            "is_ambiguous": torch.tensor(self._to_int01(row.get("is_ambiguous", 0)), dtype=torch.int64),
            "sample_id": str(row.get("sample_id", "unknown")),
            "dataset": str(row.get("dataset", "unknown")),
            "split": str(row.get("split", self.split or "unknown")),
        }
        return sample

    def _resolve_npz_path(self, row: Mapping[str, Any]) -> Path:
        raw_path = str(row.get("npz_path", "")).strip()
        if not raw_path:
            raise ValueError(f"Missing npz_path for sample {row.get('sample_id', 'unknown')}")

        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = self.processed_root / candidate

        if not candidate.exists():
            fallback = self.processed_root / f"{row.get('sample_id', '')}.npz"
            if fallback.exists():
                return fallback
            raise FileNotFoundError(f"NPZ file not found: {candidate}")

        return candidate

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        image_min = float(np.min(image))
        image_max = float(np.max(image))
        if image_max > image_min:
            image = (image - image_min) / (image_max - image_min)
        else:
            image = np.zeros_like(image, dtype=np.float32)

        if self.normalize_to_minus1_1:
            image = image * 2.0 - 1.0
        return image.astype(np.float32)

    def _apply_simple_augment(self, image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Keep identical random geometry transform for image/mask pair.
        if torch.rand(1).item() < 0.5:
            image = torch.flip(image, dims=(-1,))
            mask = torch.flip(mask, dims=(-1,))

        k = int(torch.randint(low=0, high=4, size=(1,)).item())
        if k > 0:
            image = torch.rot90(image, k=k, dims=(-2, -1))
            mask = torch.rot90(mask, k=k, dims=(-2, -1))
        return image, mask

    @staticmethod
    def _to_int01(value: Any) -> int:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, np.integer)):
            return 1 if int(value) != 0 else 0

        text = str(value).strip().lower()
        return 1 if text in {"1", "true", "yes"} else 0
