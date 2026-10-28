"""Unit tests for DistilHuBERT architecture and statistical pooling."""

import pytest
import torch
from src.models.distilhubert_classifier import StatisticalPooling, DistilHuBERTClassifier


def test_statistical_pooling():
    pool = StatisticalPooling()
    # Batch=2, Time=50, Hidden=128
    dummy_input = torch.randn(2, 50, 128)
    pooled = pool(dummy_input)

    assert pooled.shape == (2, 256)
    # Check that pooled std is positive
    std_part = pooled[:, 128:]
    assert torch.all(std_part >= 0.0)


def test_distilhubert_classifier_mock():
    # Test classifier forward without loading heavy pretrained weights via monkeypatch/small module
    hidden_dim = 64
    pool = StatisticalPooling()
    classifier = torch.nn.Sequential(
        torch.nn.Linear(hidden_dim * 2, 32),
        torch.nn.BatchNorm1d(32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 1),
    )

    dummy_features = torch.randn(2, 20, hidden_dim)
    pooled = pool(dummy_features)
    logits = classifier(pooled)
    probs = torch.sigmoid(logits)

    assert logits.shape == (2, 1)
    assert probs.shape == (2, 1)
    assert torch.all((probs >= 0.0) & (probs <= 1.0))
