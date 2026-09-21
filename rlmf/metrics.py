from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def compute_binary_metrics(
    labels: Iterable[int],
    probs: Iterable[float],
    threshold: float | Iterable[float] = 0.5,
) -> dict[str, float]:
    try:
        from sklearn.metrics import (
            accuracy_score,
            average_precision_score,
            confusion_matrix,
            f1_score,
            matthews_corrcoef,
            precision_score,
            recall_score,
            roc_auc_score,
        )
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for metric calculation. "
            "Install dependencies with: pip install -r requirements_birna.txt"
        ) from exc

    y_true = np.asarray(list(labels), dtype=int)
    y_prob = np.asarray(list(probs), dtype=float)
    if y_true.size == 0:
        raise ValueError("Cannot compute metrics on an empty label array.")
    if y_true.shape[0] != y_prob.shape[0]:
        raise ValueError(f"Metric input size mismatch: labels={y_true.shape[0]}, probs={y_prob.shape[0]}")

    if np.isscalar(threshold):
        threshold_array = float(threshold)
        if not math.isfinite(threshold_array):
            raise ValueError("Metric threshold must be finite.")
    else:
        threshold_array = np.asarray(list(threshold), dtype=float)
        if threshold_array.shape != y_prob.shape:
            raise ValueError(
                f"Metric threshold size mismatch: thresholds={threshold_array.shape[0]}, probs={y_prob.shape[0]}"
            )
        if not np.isfinite(threshold_array).all():
            raise ValueError("Metric thresholds must be finite.")

    y_pred = (y_prob >= threshold_array).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if tn + fp else math.nan
    sensitivity = float(tp / (tp + fn)) if tp + fn else math.nan
    metrics = {
        "ACC": float(accuracy_score(y_true, y_pred)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "AUC": math.nan,
        "AUPRC": math.nan,
    }
    try:
        metrics["AUC"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        pass
    try:
        metrics["AUPRC"] = float(average_precision_score(y_true, y_prob))
    except ValueError:
        pass
    return metrics


def format_metrics(metrics: dict[str, float]) -> str:
    ordered_keys = [
        "ACC", "MCC", "AUC", "AUPRC", "F1", "Precision", "Recall", "Specificity"
    ]
    parts = []
    for key in ordered_keys:
        value = metrics.get(key, math.nan)
        if isinstance(value, float) and math.isnan(value):
            parts.append(f"{key}=nan")
        else:
            parts.append(f"{key}={value:.4f}")
    return " ".join(parts)


def json_safe_metrics(obj):
    if isinstance(obj, dict):
        return {key: json_safe_metrics(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [json_safe_metrics(value) for value in obj]
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj
