from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.utils.io import read_split_info
from src.utils.run_report import append_markdown_run_report


def _build_paths_2021(root: Path, sample_id: str, mask_source: str = "A1") -> dict[str, Path]:
    seg_dir = root / "SEGMENTATIONS" / f"Manual-{mask_source}"
    return {
        "image": root / "IMAGES" / f"{sample_id}.tiff",
        "li": seg_dir / f"{sample_id}-LI.txt",
        "ma": seg_dir / f"{sample_id}-MA.txt",
        "cf": root / "CF" / f"{sample_id}_CF.txt",
    }


def _build_paths_2022(root: Path, sample_id: str, mask_source: str = "A1") -> dict[str, Path]:
    seg_dir = root / "LIMA-Profiles" / f"Manual-{mask_source}"
    return {
        "image": root / "images" / f"{sample_id}.tiff",
        "li": seg_dir / f"{sample_id}-LI.txt",
        "ma": seg_dir / f"{sample_id}-MA.txt",
        "cf": root / "CF" / f"{sample_id}_CF.txt",
    }


def _extract_npz_scalar(npz: Any, key: str, default: float = 0.0) -> float:
    if key not in npz:
        return float(default)
    return float(np.asarray(npz[key]).item())


def _extract_npz_int(npz: Any, key: str, default: int = 0) -> int:
    if key not in npz:
        return int(default)
    return int(np.asarray(npz[key]).item())


def main(config_path: str) -> None:
    root = Path(__file__).resolve().parent
    config = yaml.safe_load((root / config_path).read_text(encoding="utf-8"))

    paths_cfg = config["paths"]
    preprocess_cfg = config.get("preprocess", {})

    processed_root = root / paths_cfg["processed_root"]
    split_info_path = root / paths_cfg["split_info_json"]
    run_report_md = root / paths_cfg.get("run_report_md", "reports/run_tracking.md")

    cubs_2021_root = root / paths_cfg["cubs_2021_dir"]
    cubs_2022_root = root / paths_cfg["cubs_2022_dir"]
    mask_source = str(preprocess_cfg.get("mask_source", "A1"))

    split_info = read_split_info(split_info_path)

    records: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    for dataset_name, block in (("cubs_2021", split_info["cubs_2021"]), ("cubs_2022", split_info["cubs_2022"])):
        builder = _build_paths_2021 if dataset_name == "cubs_2021" else _build_paths_2022
        dataset_root = cubs_2021_root if dataset_name == "cubs_2021" else cubs_2022_root

        for split_name in ("train", "val", "test"):
            for sample_id in block.get(split_name, []):
                npz_path = processed_root / f"{sample_id}.npz"
                raw_paths = builder(dataset_root, sample_id, mask_source)
                if not npz_path.exists():
                    failed.append(
                        {
                            "dataset": dataset_name,
                            "sample_id": sample_id,
                            "split": split_name,
                            "reason": "missing npz",
                        }
                    )
                    continue

                with np.load(npz_path) as npz:
                    records.append(
                        {
                            "dataset": dataset_name,
                            "sample_id": sample_id,
                            "split": split_name,
                            "imt_mm": _extract_npz_scalar(npz, "imt_mm", 0.0),
                            "imt_std_mm": _extract_npz_scalar(npz, "imt_std_mm", 0.0),
                            "imt_px": _extract_npz_scalar(npz, "imt_px", 0.0),
                            "imt_std_px": _extract_npz_scalar(npz, "imt_std_px", 0.0),
                            "hard_thin": _extract_npz_int(npz, "hard_thin", 0),
                            "is_ambiguous": _extract_npz_int(npz, "is_ambiguous", 0),
                            "npz_path": str(npz_path),
                            "image_path": str(raw_paths["image"]),
                            "li_path": str(raw_paths["li"]),
                            "ma_path": str(raw_paths["ma"]),
                            "cf_path": str(raw_paths["cf"]),
                        }
                    )

    metadata_df = pd.DataFrame(records)
    if not metadata_df.empty:
        metadata_df = metadata_df.sort_values(["dataset", "split", "sample_id"])
    metadata_path = processed_root / "metadata.csv"
    metadata_df.to_csv(metadata_path, index=False)

    failed_df = pd.DataFrame(failed, columns=["dataset", "sample_id", "split", "reason"])
    failed_path = processed_root / "failed_samples.csv"
    failed_df.to_csv(failed_path, index=False)

    summary = {
        "metadata_rows": int(len(metadata_df)),
        "failed_rows": int(len(failed_df)),
        "train_rows": int((metadata_df["split"] == "train").sum()) if not metadata_df.empty else 0,
        "val_rows": int((metadata_df["split"] == "val").sum()) if not metadata_df.empty else 0,
        "test_rows": int((metadata_df["split"] == "test").sum()) if not metadata_df.empty else 0,
    }
    details = {
        "metadata_csv": str(metadata_path),
        "failed_csv": str(failed_path),
        "split_info": str(split_info_path),
    }
    report_path = append_markdown_run_report(run_report_md, "metadata_refresh", summary, details)

    print(summary)
    print(f"Saved metadata: {metadata_path}")
    print(f"Saved failed samples: {failed_path}")
    print(f"Saved run report: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rebuild metadata.csv from processed NPZ and split_info.json")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
