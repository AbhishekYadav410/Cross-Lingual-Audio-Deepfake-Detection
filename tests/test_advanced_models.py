"""Tests for advanced architectures: LCNN (Model B2) and SE-ResNet (Model B3)."""

import torch
import numpy as np
import pytest
from src.models.lcnn import LCNN, MFM
from src.models.se_resnet import SEResNet
from src.models.ensemble import ForensicEnsemble
from src.features.spec_augment import SpecAugment
from src.evaluation.threshold_optimizer import find_optimal_threshold, evaluate_comprehensive_metrics


def test_lcnn_forward_and_params():
    model = LCNN(in_channels=1, dropout_rate=0.3)
    params = model.count_parameters()

    assert params < 600000, f"LCNN params {params} exceed 600k!"
    assert params > 150000

    dummy_input = torch.randn(4, 1, 80, 94)
    logits = model(dummy_input)
    assert logits.shape == (4, 1)

    probs = model.predict_proba(dummy_input)
    assert probs.shape == (4, 1)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_se_resnet_forward_and_params():
    model = SEResNet(in_channels=1, base_channels=24)
    params = model.count_parameters()

    assert params < 500000, f"SE-ResNet params {params} exceed 500k!"
    assert params > 150000

    dummy_input = torch.randn(4, 1, 80, 94)
    logits = model(dummy_input)
    assert logits.shape == (4, 1)

    probs = model.predict_proba(dummy_input)
    assert probs.shape == (4, 1)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_spec_augment():
    augmenter = SpecAugment(freq_mask_param=12, time_mask_param=16, prob=1.0)
    spec = np.ones((80, 94), dtype=np.float32)
    augmented = augmenter(spec)

    assert augmented.shape == (80, 94)
    # The augmented spec should have elements replaced with mean
    assert not np.isnan(augmented).any()


def test_threshold_optimizer_and_comprehensive_metrics():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

    best_thresh = find_optimal_threshold(y_true, y_score, metric="balanced_accuracy")
    assert 0.4 <= best_thresh <= 0.6

    metrics = evaluate_comprehensive_metrics(y_true, y_score, threshold=best_thresh)
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "specificity" in metrics
    assert "balanced_accuracy" in metrics
    assert "macro_f1" in metrics
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0


def test_forensic_ensemble():
    ensemble = ForensicEnsemble(weights={"svm": 0.4, "cnn": 0.6})
    probs_dict = {
        "svm": np.array([0.2, 0.8]),
        "cnn": np.array([0.4, 0.6]),
    }
    blended = ensemble.predict_proba(probs_dict)
    assert len(blended) == 2
    assert np.isclose(blended[0], 0.4 * 0.2 + 0.6 * 0.4)
    assert np.isclose(blended[1], 0.4 * 0.8 + 0.6 * 0.6)
