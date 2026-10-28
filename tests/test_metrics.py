"""Tests for anti-spoofing evaluation metrics."""

import numpy as np
import pytest
from src.evaluation.metrics import compute_eer, evaluate_predictions, compute_bootstrap_ci


def test_perfect_predictions():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.15, 0.85, 0.9, 0.95])

    metrics = evaluate_predictions(y_true, y_score)
    assert metrics["auroc"] == 1.0
    assert metrics["eer"] == 0.0
    assert metrics["balanced_accuracy_tau0.5"] == 1.0


def test_random_predictions():
    np.random.seed(42)
    y_true = np.random.randint(0, 2, size=200)
    y_score = np.random.rand(200)

    metrics = evaluate_predictions(y_true, y_score)
    # Random guess AUROC should be roughly around 0.5
    assert 0.35 <= metrics["auroc"] <= 0.65
    assert 0.35 <= metrics["eer"] <= 0.65


def test_bootstrap_ci():
    y_true = np.array([0] * 50 + [1] * 50)
    y_score = np.concatenate([np.random.normal(0.2, 0.1, 50), np.random.normal(0.8, 0.1, 50)])

    ci = compute_bootstrap_ci(y_true, y_score, n_resamples=100)
    assert "auroc" in ci and "eer" in ci
    assert ci["auroc"]["lower"] <= ci["auroc"]["mean"] <= ci["auroc"]["upper"]
