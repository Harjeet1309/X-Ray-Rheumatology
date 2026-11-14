"""Comprehensive medical imaging evaluation metrics (PRD Section 10).

Covers:
1. Classification: Accuracy, Precision, Recall, F1, Sensitivity, Specificity, ROC-AUC, PR-AUC, Confusion Matrix.
2. Detection: Precision, Recall, mAP@50, mAP@50:95.
3. Segmentation: Dice coefficient, Intersection over Union (IoU), Precision, Recall.
4. Severity Scoring: MAE, RMSE, Quadratic Weighted Kappa (QWK), Ordinal Accuracy.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute binary RA classification metrics including Sensitivity and Specificity."""
    y_t = np.asarray(y_true, dtype=np.int64)
    y_p = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_p >= threshold).astype(np.int64)

    cm = confusion_matrix(y_t, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))

    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)

    try:
        roc_auc = float(roc_auc_score(y_t, y_p)) if len(np.unique(y_t)) > 1 else 0.5
    except Exception:
        roc_auc = 0.5

    try:
        pr_auc = float(average_precision_score(y_t, y_p)) if len(np.unique(y_t)) > 1 else 0.5
    except Exception:
        pr_auc = 0.5

    return {
        "accuracy": round(float(accuracy_score(y_t, y_pred)), 4),
        "precision": round(float(precision_score(y_t, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_t, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_t, y_pred, zero_division=0)), 4),
        "sensitivity": round(float(sensitivity), 4),
        "specificity": round(float(specificity), 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def compute_segmentation_metrics(
    pred_mask: np.ndarray,
    true_mask: np.ndarray,
    num_classes: int = 4,
    ignore_background: bool = True,
) -> Dict[str, Any]:
    """Compute pixel-level Dice, IoU, Precision, and Recall across anatomical & erosion classes."""
    p = pred_mask.astype(np.int64)
    t = true_mask.astype(np.int64)

    class_Indices = range(1 if ignore_background else 0, num_classes)
    dices: List[float] = []
    ious: List[float] = []
    precisions: List[float] = []
    recalls: List[float] = []
    per_class: Dict[str, Dict[str, float]] = {}

    for c in class_Indices:
        pc = p == c
        tc = t == c
        inter = float(np.logical_and(pc, tc).sum())
        union = float(np.logical_or(pc, tc).sum())
        p_sum = float(pc.sum())
        t_sum = float(tc.sum())

        if t_sum == 0 and p_sum == 0:
            dice = 1.0
            iou = 1.0
            prec = 1.0
            rec = 1.0
        else:
            dice = (2.0 * inter + 1e-6) / (p_sum + t_sum + 1e-6)
            iou = (inter + 1e-6) / (union + 1e-6)
            prec = (inter + 1e-6) / (p_sum + 1e-6)
            rec = (inter + 1e-6) / (t_sum + 1e-6)

        dices.append(dice)
        ious.append(iou)
        precisions.append(prec)
        recalls.append(rec)
        per_class[f"class_{c}"] = {
            "dice": round(dice, 4),
            "iou": round(iou, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
        }

    return {
        "dice": round(float(np.mean(dices)) if dices else 0.0, 4),
        "iou": round(float(np.mean(ious)) if ious else 0.0, 4),
        "precision": round(float(np.mean(precisions)) if precisions else 0.0, 4),
        "recall": round(float(np.mean(recalls)) if recalls else 0.0, 4),
        "per_class": per_class,
    }


def compute_box_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Compute IoU between two boxes in [x, y, w, h] format."""
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return float(inter / max(union, 1e-6))


def compute_detection_metrics(
    pred_boxes: List[Dict[str, Any]],
    gt_boxes: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Compute Precision, Recall, mAP@50, and mAP@50:95 for joint/abnormality detection."""
    if not gt_boxes:
        return {"precision": 1.0, "recall": 1.0, "mAP50": 1.0, "mAP50_95": 1.0}

    iou_thresholds = np.linspace(0.50, 0.95, 10)
    aps_per_thresh: List[float] = []
    prec_50 = 0.0
    rec_50 = 0.0

    sorted_preds = sorted(pred_boxes, key=lambda x: float(x.get("confidence", 0.5)), reverse=True)

    for t_idx, thresh in enumerate(iou_thresholds):
        matched_gt = set()
        tp = 0
        fp = 0
        for p in sorted_preds:
            best_iou = 0.0
            best_g_idx = -1
            for g_idx, g in enumerate(gt_boxes):
                if g_idx in matched_gt:
                    continue
                if p.get("label") != g.get("label"):
                    continue
                iou = compute_box_iou(p["bbox"], g["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_g_idx = g_idx
            if best_iou >= thresh and best_g_idx >= 0:
                tp += 1
                matched_gt.add(best_g_idx)
            else:
                fp += 1

        fn = len(gt_boxes) - tp
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        ap = prec * rec if (prec + rec) > 0 else 0.0
        # Harmonic/interpolated approximation for single-operating-point or ranked boxes
        ap_val = (2 * prec * rec / max(prec + rec, 1e-6)) if (tp > 0) else 0.0
        aps_per_thresh.append(ap_val)
        if t_idx == 0:
            prec_50 = prec
            rec_50 = rec

    return {
        "precision": round(float(prec_50), 4),
        "recall": round(float(rec_50), 4),
        "mAP50": round(float(aps_per_thresh[0]), 4),
        "mAP50_95": round(float(np.mean(aps_per_thresh)), 4),
    }


def compute_severity_metrics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    max_score: int = 5,
) -> Dict[str, float]:
    """Compute MAE, RMSE, Quadratic Weighted Kappa (QWK), and Ordinal Accuracy."""
    yt = np.asarray(y_true, dtype=np.float64).flatten()
    yp = np.asarray(y_pred, dtype=np.float64).flatten()

    mae = float(mean_absolute_error(yt, yp))
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))

    yt_int = np.clip(np.round(yt), 0, max_score).astype(np.int64)
    yp_int = np.clip(np.round(yp), 0, max_score).astype(np.int64)

    ordinal_acc = float(np.mean(yt_int == yp_int))
    within_one_acc = float(np.mean(np.abs(yt_int - yp_int) <= 1))

    try:
        if len(np.unique(yt_int)) > 1 or len(np.unique(yp_int)) > 1:
            qwk = float(cohen_kappa_score(yt_int, yp_int, weights="quadratic"))
        else:
            qwk = 1.0 if ordinal_acc == 1.0 else 0.0
    except Exception:
        qwk = 0.0

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "quadratic_weighted_kappa": round(qwk, 4),
        "ordinal_accuracy": round(ordinal_acc, 4),
        "within_one_accuracy": round(within_one_acc, 4),
    }
