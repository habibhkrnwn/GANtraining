from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

_FLOAT_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def read_tiff(path: str | Path) -> np.ndarray:
    """Read a TIFF image file as a float32 numpy array."""
    image = tifffile.imread(str(path))
    return np.asarray(image, dtype=np.float32)


def read_profile_txt(path: str | Path) -> np.ndarray:
    """Read LI/MA profile file and return Nx2 array of (x, y) coordinates."""
    coords: list[tuple[float, float]] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if not stripped:
                continue
            numbers = _FLOAT_PATTERN.findall(stripped.replace(",", " "))
            if len(numbers) < 2:
                continue
            coords.append((float(numbers[0]), float(numbers[1])))

    if not coords:
        raise ValueError(f"No coordinates found in profile file: {path}")

    arr = np.asarray(coords, dtype=np.float32)
    order = np.argsort(arr[:, 0])
    return arr[order]


def read_cf_txt(path: str | Path) -> float:
    """Read conversion factor (mm/pixel) from a CF text file."""
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    numbers = _FLOAT_PATTERN.findall(text)
    if not numbers:
        raise ValueError(f"Cannot parse conversion factor from file: {path}")
    return float(numbers[0])


def load_split_info(path: str | Path) -> dict[str, Any]:
    """Load split information from a JSON file."""
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def read_split_info(path: str | Path) -> dict[str, Any]:
    """Backward-compatible alias for loading split information."""
    return load_split_info(path)
