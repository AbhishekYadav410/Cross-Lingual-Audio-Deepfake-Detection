"""Tests for Model A (Classical) and Model B (Lightweight CNN)."""

import torch
import numpy as np
import pytest
from src.models.lightweight_cnn import LightweightAudioCNN
from src.models.classical_baseline import ClassicalBaseline


def test_cnn_architecture_and_params():
    model = LightweightAudioCNN(base_channels=16, fc_hidden=64)
    param_count = model.count_parameters()

    # Must be lightweight: < 500k parameters
    assert param_count < 500000, f"Parameter count {param_count} exceeds 500k!"
    assert param_count > 100000

    # Test forward pass with batch size 4, 80 mel bands, 94 time frames
    dummy_input = torch.randn(4, 1, 80, 94)
    logits = model(dummy_input)
    assert logits.shape == (4, 1)

    probs = model.predict_proba(dummy_input)
    assert probs.shape == (4, 1)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_classical_baseline_fit_predict():
    # 50 samples, 180 dimensions
    X = np.random.randn(50, 180).astype(np.float32)
    y = np.random.randint(0, 2, size=50)

    # Test SVM
    svm = ClassicalBaseline(model_type="svm_rbf")
    svm.fit(X, y)
    probs_svm = svm.predict_proba(X)
    assert probs_svm.shape == (50, 2)
    preds_svm = svm.predict(X)
    assert len(preds_svm) == 50

    # Test Logistic Regression
    lr = ClassicalBaseline(model_type="logistic_regression")
    lr.fit(X, y)
    probs_lr = lr.predict_proba(X)
    assert probs_lr.shape == (50, 2)
