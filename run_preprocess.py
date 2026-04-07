from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **_: Any):  # type: ignore[no-redef]
        return iterable

from src.data.preprocess import build_sample_record, save_metadata_csv, save_sample_npz
from src.utils.io import read_split_info


def resolve_split_items(split_block: dict[str, list[str]]) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for split_name in ["train", "val", "test"]:
        for sample_id in split_block.get(split_name, []):
            rows.append((sample_id, split_name))
    return rows


def build_paths_2021(root: Path, sample_id: str, mask_source: str = "A1") -> dict[str, Path]:
    seg_dir = root / "SEGMENTATIONS" / f"Manual-{mask_source}"
    return {
        "image": root / "IMAGES" / f"{sample_id}.tiff",
        "li": seg_dir / f"{sample_id}-LI.txt",
        "ma": seg_dir / f"{sample_id}-MA.txt",
        "cf": root / "CF" / f"{sample_id}_CF.txt",
    }


def build_paths_2022(root: Path, sample_id: str, mask_source: str = "A1") -> dict[str, Path]:
    seg_dir = root / "LIMA-Profiles" / f"Manual-{mask_source}"
    return {
        "image": root / "images" / f"{sample_id}.tiff",
        "li": seg_dir / f"{sample_id}-LI.txt",
        "ma": seg_dir / f"{sample_id}-MA.txt",
        "cf": root / "CF" / f"{sample_id}_CF.txt",
    }


def all_exist(paths: dict[str, Path]) -> tuple[bool, list[str]]:
    missing = [name for name, path in paths.items() if not path.exists()]
    return len(missing) == 0, missing


def save_overlay_png(image: Any, mask: Any, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 5))
    plt.imshow(image, cmap="gray")
    plt.imshow(mask, cmap="jet", alpha=0.35)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()


def process_dataset(
    dataset_name: str,
    dataset_root: Path,
    split_items: Iterable[tuple[str, str]],
    processed_root: Path,
    mask_source: str,
    image_size: tuple[int, int],
    interpolation_points: int,
    imt_threshold_mm: float,
    ambiguous_std_threshold_mm: float,
    save_overlay_preview: bool,
    preview_limit: int = 20,
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metadata_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    preview_count = 0

    path_builder = build_paths_2021 if dataset_name == "cubs_2021" else build_paths_2022

    rows = list(split_items)
    if limit is not None:
        rows = rows[:limit]

    for sample_id, split_name in tqdm(rows, desc=f"Processing {dataset_name}"):
        try:
            paths = path_builder(dataset_root, sample_id, mask_source)
            exists, missing = all_exist(paths)
            if not exists:
                failed_rows.append(
                    {
                        "dataset": dataset_name,
                        "sample_id": sample_id,
                        "split": split_name,
                        "reason": f"missing files: {','.join(missing)}",
                    }
                )
                continue

            record = build_sample_record(
                sample_id=sample_id,
                image_path=paths["image"],
                li_path=paths["li"],
                ma_path=paths["ma"],
                cf_path=paths["cf"],
                out_size=image_size,
                num_points=interpolation_points,
                imt_threshold_mm=imt_threshold_mm,
                ambiguous_std_threshold_mm=ambiguous_std_threshold_mm,
            )
            npz_path = save_sample_npz(record, processed_root)

            metadata_rows.append(
                {
                    "dataset": dataset_name,
                    "sample_id": sample_id,
                    "split": split_name,
                    "imt_mm": record["imt_mm"],
                    "imt_std_mm": record["imt_std_mm"],
                    "imt_px": record["imt_px"],
                    "imt_std_px": record["imt_std_px"],
                    "hard_thin": record["hard_thin"],
                    "is_ambiguous": record["is_ambiguous"],
                    "npz_path": str(npz_path),
                    "image_path": str(paths["image"]),
                    "li_path": str(paths["li"]),
                    "ma_path": str(paths["ma"]),
                    "cf_path": str(paths["cf"]),
                }
            )

            if save_overlay_preview and preview_count < preview_limit:
                preview_dir = processed_root / "preview_overlays" / dataset_name / split_name
                save_overlay_png(record["image"], record["mask"], preview_dir / f"{sample_id}.png")
                preview_count += 1

        except Exception as exc:  # pragma: no cover
            failed_rows.append(
                {
                    "dataset": dataset_name,
                    "sample_id": sample_id,
                    "split": split_name,
                    "reason": str(exc),
                }
            )

    return metadata_rows, failed_rows


def main(config_path: str, limit: int | None = None) -> None:
    with Path(config_path).open("r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    paths_cfg = cfg.get("paths", {})
    data_cfg = cfg.get("data", {})
    preprocess_cfg = cfg.get("preprocess", {})

    processed_root = ROOT / paths_cfg["processed_dir"]
    processed_root.mkdir(parents=True, exist_ok=True)

    split_info_path = ROOT / paths_cfg["split_info_json"]
    split_info = read_split_info(split_info_path)

    cubs_2021_root = ROOT / paths_cfg["cubs_2021_dir"]
    cubs_2022_root = ROOT / paths_cfg["cubs_2022_dir"]

    image_size = tuple(preprocess_cfg.get("image_size", data_cfg.get("image_size", [256, 256])))
    if len(image_size) != 2:
        raise ValueError("image_size must contain two integers [H, W].")

    interpolation_points = int(preprocess_cfg.get("interpolation_points", image_size[1]))
    imt_threshold_mm = float(preprocess_cfg.get("imt_threshold_mm", data_cfg.get("thin_imc_threshold_mm", 0.5)))
    ambiguous_std_threshold_mm = float(
        preprocess_cfg.get(
            "ambiguous_std_threshold_mm",
            data_cfg.get("ambiguous_std_threshold_mm", 0.15),
        )
    )
    mask_source = str(preprocess_cfg.get("mask_source", "A1"))
    save_overlay_preview = bool(preprocess_cfg.get("save_overlay_preview", True))
    preview_limit = int(preprocess_cfg.get("preview_limit", 20))

    rows_2021 = resolve_split_items(split_info["cubs_2021"])
    rows_2022 = resolve_split_items(split_info["cubs_2022"])

    metadata_2021, failed_2021 = process_dataset(
        dataset_name="cubs_2021",
        dataset_root=cubs_2021_root,
        split_items=rows_2021,
        processed_root=processed_root,
        mask_source=mask_source,
        image_size=(int(image_size[0]), int(image_size[1])),
        interpolation_points=interpolation_points,
        imt_threshold_mm=imt_threshold_mm,
        ambiguous_std_threshold_mm=ambiguous_std_threshold_mm,
        save_overlay_preview=save_overlay_preview,
        preview_limit=preview_limit,
        limit=limit,
    )
    metadata_2022, failed_2022 = process_dataset(
        dataset_name="cubs_2022",
        dataset_root=cubs_2022_root,
        split_items=rows_2022,
        processed_root=processed_root,
        mask_source=mask_source,
        image_size=(int(image_size[0]), int(image_size[1])),
        interpolation_points=interpolation_points,
        imt_threshold_mm=imt_threshold_mm,
        ambiguous_std_threshold_mm=ambiguous_std_threshold_mm,
        save_overlay_preview=save_overlay_preview,
        preview_limit=preview_limit,
        limit=limit,
    )

    metadata_rows = metadata_2021 + metadata_2022
    failed_rows = failed_2021 + failed_2022

    metadata_path = save_metadata_csv(metadata_rows, processed_root / "metadata.csv")
    failed_df = pd.DataFrame(failed_rows, columns=["dataset", "sample_id", "split", "reason"])
    failed_path = processed_root / "failed_samples.csv"
    failed_df.to_csv(failed_path, index=False)

    summary = {
        "processed_total": len(metadata_rows),
        "failed_total": len(failed_rows),
        "processed_cubs_2021": len(metadata_2021),
        "processed_cubs_2022": len(metadata_2022),
    }
    print(summary)
    print(f"Saved metadata to: {metadata_path}")
    print(f"Saved failed log to: {failed_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess CUBS 2021+2022 into NPZ + metadata CSV")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit sample count per dataset for quick testing.",
    )
    args = parser.parse_args()
    main(args.config, args.limit)
