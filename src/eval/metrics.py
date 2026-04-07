from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity


def dice_score(y_pred: np.ndarray, y_true: np.ndarray, eps: float = 1e-7) -> float:
    y_pred = (y_pred > 0).astype(np.float32)
    y_true = (y_true > 0).astype(np.float32)

    intersection = float(np.sum(y_pred * y_true))
    denominator = float(np.sum(y_pred) + np.sum(y_true))
    return (2.0 * intersection + eps) / (denominator + eps)


def iou_score(y_pred: np.ndarray, y_true: np.ndarray, eps: float = 1e-7) -> float:
    y_pred = (y_pred > 0).astype(np.float32)
    y_true = (y_true > 0).astype(np.float32)

    intersection = float(np.sum(y_pred * y_true))
    union = float(np.sum((y_pred + y_true) > 0))
    return (intersection + eps) / (union + eps)


def imt_error_mm(imt_pred_mm: float, imt_true_mm: float) -> float:
    return float(abs(imt_pred_mm - imt_true_mm))


def ssim_score(image_a: np.ndarray, image_b: np.ndarray) -> float:
    a = image_a.astype(np.float32)
    b = image_b.astype(np.float32)
    data_range = float(max(a.max(), b.max()) - min(a.min(), b.min()))
    if data_range == 0:
        data_range = 1.0
    return float(structural_similarity(a, b, data_range=data_range))
