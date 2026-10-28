"""Threshold optimization and comprehensive forensic metric computation.

Optimizes decision threshold tau on validation set to maximize Balanced Accuracy
and Macro-F1, then evaluates held-out test sets at both default (0.5) and optimal tau*.
"""

from typing import Dict, Tuple, Any
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)
from .metrics import compute_eer, compute_bootstrap_ci


def find_optimal_threshold(
    y_val_true: np.ndarray,
    y_val_score: np.ndarray,
    metric: str = "balanced_accuracy",
) -> float:
    """Scans thresholds in [0.05, 0.95] to find the optimal operating threshold on validation data."""
    best_thresh = 0.5
    best_score = -1.0

    thresholds = np.linspace(0.05, 0.95, 91)
    for t in thresholds:
        y_pred = (y_val_score >= t).astype(int)
        if metric == "balanced_accuracy":
            score = balanced_accuracy_score(y_val_true, y_pred)
        elif metric == "f1":
            score = f1_score(y_val_true, y_pred, average="macro", zero_division=0)
        else:
            score = accuracy_score(y_val_true, y_pred)

        if score > best_score:
            best_score = score
            best_thresh = float(t)

    return best_thresh


def evaluate_comprehensive_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Computes all essential classification metrics at a given threshold."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Rate calculations
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    try:
        auroc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        auroc = 0.5

    eer, eer_threshold = compute_eer(y_true, y_score)

    return {
        "accuracy": round(float(accuracy), 4),
        "balanced_accuracy": round(float(balanced_acc), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "specificity": round(float(specificity), 4),
        "macro_f1": round(float(macro_f1), 4),
        "auroc": round(float(auroc), 4),
        "eer": round(float(eer), 4),
        "eer_threshold": round(float(eer_threshold), 4),
        "applied_threshold": round(float(threshold), 4),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }
