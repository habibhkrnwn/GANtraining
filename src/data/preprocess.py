from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from PIL import Image
from scipy.interpolate import UnivariateSpline

from src.utils.io import read_cf_txt, read_profile_txt, read_tiff


@dataclass
class IMTStats:
    imt_mean_mm: float
    imt_std_mm: float
    imt_mean_px: float
    imt_std_px: float


def _build_profile_spline(profile_xy: np.ndarray) -> UnivariateSpline:
    """Build spline from profile points while handling duplicate x values."""
    if profile_xy.ndim != 2 or profile_xy.shape[1] < 2:
        raise ValueError("Profile must have shape (N, 2).")

    x = profile_xy[:, 0].astype(np.float64)
    y = profile_xy[:, 1].astype(np.float64)

    x_unique, unique_indices = np.unique(x, return_index=True)
    y_unique = y[unique_indices]

    if len(x_unique) < 2:
        raise ValueError("Profile must contain at least two unique x coordinates.")

    k = 1 if len(x_unique) < 4 else 3
    return UnivariateSpline(x_unique, y_unique, k=k, s=0)


def build_imc_mask(
    li_profile_xy: np.ndarray,
    ma_profile_xy: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Create binary IMC mask by filling the area between LI and MA splines."""
    height, width = image_shape

    li_spline = _build_profile_spline(li_profile_xy)
    ma_spline = _build_profile_spline(ma_profile_xy)

    li_x = li_profile_xy[:, 0].astype(np.float64)
    ma_x = ma_profile_xy[:, 0].astype(np.float64)
    x_min = int(np.ceil(max(float(np.min(li_x)), float(np.min(ma_x)), 0.0)))
    x_max = int(np.floor(min(float(np.max(li_x)), float(np.max(ma_x)), float(width - 1))))
    if x_max < x_min:
        raise ValueError("LI and MA profiles do not overlap on x-axis.")

    x_grid = np.arange(x_min, x_max + 1, dtype=np.float64)

    li_y = np.clip(li_spline(x_grid), 0, height - 1)
    ma_y = np.clip(ma_spline(x_grid), 0, height - 1)

    upper = np.minimum(li_y, ma_y).astype(np.int32)
    lower = np.maximum(li_y, ma_y).astype(np.int32)

    mask = np.zeros((height, width), dtype=np.uint8)
    for local_idx, x_val in enumerate(x_grid.astype(np.int32)):
        mask[upper[local_idx] : lower[local_idx] + 1, x_val] = 1
    return mask


def compute_imt_stats(
    li_profile_xy: np.ndarray,
    ma_profile_xy: np.ndarray,
    cf_mm_per_pixel: float,
    width: int,
) -> IMTStats:
    """Compute IMT statistics in pixels and millimeters over overlapping LI/MA x-range."""
    li_spline = _build_profile_spline(li_profile_xy)
    ma_spline = _build_profile_spline(ma_profile_xy)

    li_x = li_profile_xy[:, 0].astype(np.float64)
    ma_x = ma_profile_xy[:, 0].astype(np.float64)
    x_start = max(float(np.min(li_x)), float(np.min(ma_x)))
    x_end = min(float(np.max(li_x)), float(np.max(ma_x)))
    if x_end <= x_start:
        raise ValueError("LI and MA profiles do not overlap on x-axis.")

    sample_count = max(int(width), 2)
    x_grid = np.linspace(x_start, x_end, num=sample_count, dtype=np.float64)

    thickness_px = np.abs(ma_spline(x_grid) - li_spline(x_grid))
    thickness_mm = thickness_px * cf_mm_per_pixel

    return IMTStats(
        imt_mean_mm=float(np.mean(thickness_mm)),
        imt_std_mm=float(np.std(thickness_mm)),
        imt_mean_px=float(np.mean(thickness_px)),
        imt_std_px=float(np.std(thickness_px)),
    )


def classify_case(
    imt_mean_mm: float,
    imt_std_mm: float,
    thin_threshold_mm: float = 0.5,
    ambiguous_std_threshold_mm: float = 0.15,
) -> dict[str, bool]:
    """Generate clinical flags used for downstream conditioning/analysis."""
    return {
        "is_thin_imc": imt_mean_mm < thin_threshold_mm,
        "is_ambiguous": imt_std_mm > ambiguous_std_threshold_mm,
    }


def _ensure_2d_image(image: np.ndarray) -> np.ndarray:
    """Convert loaded TIFF image to a single-channel 2D array."""
    arr = np.asarray(image)
    if arr.ndim == 2:
        return arr.astype(np.float32)

    if arr.ndim == 3:
        if arr.shape[0] == 1:
            return arr[0].astype(np.float32)
        if arr.shape[-1] == 1:
            return arr[..., 0].astype(np.float32)
        return arr[..., 0].astype(np.float32)

    raise ValueError(f"Unsupported image shape: {arr.shape}")


def _resize_array(arr: np.ndarray, out_size: tuple[int, int], is_mask: bool) -> np.ndarray:
    """Resize image or mask to (height, width) using proper interpolation."""
    out_h, out_w = out_size
    if is_mask:
        pil = Image.fromarray(arr.astype(np.uint8))
        resized = pil.resize((out_w, out_h), Image.NEAREST)
        return (np.asarray(resized, dtype=np.uint8) > 0).astype(np.uint8)

    pil = Image.fromarray(arr.astype(np.float32))
    resized = pil.resize((out_w, out_h), Image.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


def build_sample_record(
    sample_id: str,
    image_path: str | Path,
    li_path: str | Path,
    ma_path: str | Path,
    cf_path: str | Path,
    out_size: tuple[int, int] = (256, 256),
    num_points: int | None = None,
    imt_threshold_mm: float = 0.5,
    ambiguous_std_threshold_mm: float = 0.15,
) -> dict[str, Any]:
    """Build one processed sample record from raw image/profile/CF files."""
    image_raw = _ensure_2d_image(read_tiff(image_path))
    li_profile_xy = read_profile_txt(li_path)
    ma_profile_xy = read_profile_txt(ma_path)
    cf_mm_per_pixel = read_cf_txt(cf_path)

    raw_h, raw_w = image_raw.shape
    raw_mask = build_imc_mask(li_profile_xy, ma_profile_xy, (raw_h, raw_w))

    stat_width = int(num_points) if num_points is not None and int(num_points) > 1 else raw_w
    imt_stats = compute_imt_stats(
        li_profile_xy=li_profile_xy,
        ma_profile_xy=ma_profile_xy,
        cf_mm_per_pixel=cf_mm_per_pixel,
        width=stat_width,
    )
    flags = classify_case(
        imt_mean_mm=imt_stats.imt_mean_mm,
        imt_std_mm=imt_stats.imt_std_mm,
        thin_threshold_mm=imt_threshold_mm,
        ambiguous_std_threshold_mm=ambiguous_std_threshold_mm,
    )

    image_out = _resize_array(image_raw, out_size, is_mask=False)
    mask_out = _resize_array(raw_mask, out_size, is_mask=True)

    return {
        "sample_id": sample_id,
        "image": image_out.astype(np.float32),
        "mask": mask_out.astype(np.uint8),
        "raw_shape": (int(raw_h), int(raw_w)),
        "out_shape": (int(out_size[0]), int(out_size[1])),
        "cf_mm_per_pixel": float(cf_mm_per_pixel),
        "imt_mm": float(imt_stats.imt_mean_mm),
        "imt_std_mm": float(imt_stats.imt_std_mm),
        "imt_px": float(imt_stats.imt_mean_px),
        "imt_std_px": float(imt_stats.imt_std_px),
        "is_thin_imc": bool(flags["is_thin_imc"]),
        "is_ambiguous": bool(flags["is_ambiguous"]),
        "hard_thin": bool(flags["is_thin_imc"]),
    }


def save_sample_npz(record: Mapping[str, Any], processed_root: str | Path) -> Path:
    """Persist one processed sample as compressed NPZ."""
    out_dir = Path(processed_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_id = str(record["sample_id"])
    npz_path = out_dir / f"{sample_id}.npz"
    np.savez_compressed(
        npz_path,
        sample_id=sample_id,
        image=np.asarray(record["image"], dtype=np.float32),
        mask=np.asarray(record["mask"], dtype=np.uint8),
        cf_mm_per_pixel=np.float32(record["cf_mm_per_pixel"]),
        imt_mm=np.float32(record["imt_mm"]),
        imt_std_mm=np.float32(record["imt_std_mm"]),
        imt_px=np.float32(record["imt_px"]),
        imt_std_px=np.float32(record["imt_std_px"]),
        hard_thin=np.uint8(bool(record["hard_thin"])),
        is_thin_imc=np.uint8(bool(record["is_thin_imc"])),
        is_ambiguous=np.uint8(bool(record["is_ambiguous"])),
    )
    return npz_path


def save_metadata_csv(rows: Sequence[Mapping[str, Any]], out_path: str | Path) -> Path:
    """Save metadata table for all processed samples."""
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    if not df.empty and "sample_id" in df.columns:
        df = df.sort_values([col for col in ["dataset", "split", "sample_id"] if col in df.columns])
    df.to_csv(out_file, index=False)
    return out_file
