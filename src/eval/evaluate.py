from __future__ import annotations

from typing import Iterable

import numpy as np

from src.eval.metrics import dice_score, iou_score


def evaluate_segmentation_batch(
    y_pred_batch: np.ndarray,
    y_true_batch: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute mean Dice and IoU for a batch of predictions."""
    if y_pred_batch.shape != y_true_batch.shape:
        raise ValueError("Prediction and target batch must have the same shape.")

    dice_list: list[float] = []
    iou_list: list[float] = []

    for pred, target in zip(y_pred_batch, y_true_batch):
        binary_pred = (pred >= threshold).astype(np.uint8)
        binary_target = (target >= 0.5).astype(np.uint8)
        dice_list.append(dice_score(binary_pred, binary_target))
        iou_list.append(iou_score(binary_pred, binary_target))

    return {
        "dice": float(np.mean(dice_list)),
        "iou": float(np.mean(iou_list)),
    }


def summarize_metric_history(history: Iterable[dict[str, float]]) -> dict[str, float]:
    history = list(history)
    if not history:
        return {"dice": 0.0, "iou": 0.0}

    return {
        "dice": float(np.mean([step["dice"] for step in history])),
        "iou": float(np.mean([step["iou"] for step in history])),
    }
