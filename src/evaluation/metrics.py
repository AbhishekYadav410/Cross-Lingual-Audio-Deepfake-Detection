"""Standardized anti-spoofing and audio deepfake evaluation metrics.

Implements:
1. Area Under ROC Curve (AUROC).
2. Equal Error Rate (EER) via exact ROC bisection.
3. Balanced Accuracy and Macro-F1.
4. Non-parametric 1,000-sample bootstrap 95% confidence intervals.
"""

from typing import Dict, Tuple, Any
import numpy as np
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
)
from scipy.optimize import brentq
from scipy.interpolate import interp1d


def compute_eer(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    """Computes Equal Error Rate (EER) and the optimal threshold tau_EER.

    Args:
        y_true: Ground truth binary labels (0 = Real, 1 = Fake).
        y_score: Continuous prediction scores / probabilities for Fake class.

    Returns:
        (eer_value, eer_threshold)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1.0 - tpr

    # Handle edge case where FPR and FNR don't cross cleanly
    try:
        eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
        thresh_interp = interp1d(fpr, thresholds, bounds_error=False, fill_value=(thresholds[0], thresholds[-1]))
        threshold = float(thresh_interp(eer))
    except (ValueError, ZeroDivisionError):
        # Fallback to closest index
        diffs = np.abs(fpr - fnr)
        idx = np.argmin(diffs)
        eer = float((fpr[idx] + fnr[idx]) / 2.0)
        threshold = float(thresholds[idx])

    return float(eer), float(threshold)


def evaluate_predictions(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Computes all primary anti-spoofing performance metrics."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    # 1. AUROC
    try:
        auroc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        auroc = 0.5

    # 2. EER
    eer, eer_threshold = compute_eer(y_true, y_score)

    # 3. Predictions at default tau = 0.5
    y_pred_default = (y_score >= threshold).astype(int)
    bal_acc_default = float(balanced_accuracy_score(y_true, y_pred_default))
    macro_f1_default = float(f1_score(y_true, y_pred_default, average="macro", zero_division=0))

    # 4. Predictions at tau_EER
    y_pred_eer = (y_score >= eer_threshold).astype(int)
    bal_acc_eer = float(balanced_accuracy_score(y_true, y_pred_eer))

    # Confusion matrix elements at tau=0.5
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_default, labels=[0, 1]).ravel()

    return {
        "auroc": auroc,
        "eer": eer,
        "eer_threshold": eer_threshold,
        "balanced_accuracy_tau0.5": bal_acc_default,
        "balanced_accuracy_tau_eer": bal_acc_eer,
        "macro_f1": macro_f1_default,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Calculates non-parametric bootstrap confidence intervals for AUROC and EER.

    Returns:
        Dict with mean, lower, and upper bounds for each metric.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n_samples = len(y_true)

    rng = np.random.RandomState(random_seed)
    auroc_boot = []
    eer_boot = []

    for _ in range(n_resamples):
        indices = rng.randint(0, n_samples, n_samples)
        sample_y = y_true[indices]
        sample_scores = y_score[indices]

        # Only compute if both classes exist in resample
        if len(np.unique(sample_y)) < 2:
            continue

        try:
            auroc_boot.append(roc_auc_score(sample_y, sample_scores))
            eer, _ = compute_eer(sample_y, sample_scores)
            eer_boot.append(eer)
        except Exception:
            continue

    alpha = (1.0 - confidence_level) / 2.0
    lower_pct = alpha * 100
    upper_pct = (1.0 - alpha) * 100

    results = {
        "auroc": {
            "mean": float(np.mean(auroc_boot)),
            "lower": float(np.percentile(auroc_boot, lower_pct)),
            "upper": float(np.percentile(auroc_boot, upper_pct)),
        },
        "eer": {
            "mean": float(np.mean(eer_boot)),
            "lower": float(np.percentile(eer_boot, lower_pct)),
            "upper": float(np.percentile(eer_boot, upper_pct)),
        },
    }
    return results
