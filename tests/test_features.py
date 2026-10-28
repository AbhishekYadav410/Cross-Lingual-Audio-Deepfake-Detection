"""Tests for spectral feature extractors."""

import numpy as np
import pytest
from src.features.extractors import FeatureExtractor
from src.features.classical_features import ClassicalFeatureExtractor


def test_log_mel_extraction():
    extractor = FeatureExtractor(sample_rate=16000, n_mels=80)
    audio = np.random.randn(48000).astype(np.float32)

    mel = extractor.extract_log_mel(audio)
    assert mel.shape[0] == 80
    assert mel.shape[1] > 0
    assert not np.isnan(mel).any()
    assert not np.isinf(mel).any()


def test_lfcc_extraction():
    extractor = FeatureExtractor(sample_rate=16000, n_lfcc=80)
    audio = np.random.randn(48000).astype(np.float32)

    lfcc = extractor.extract_lfcc(audio)
    assert lfcc.shape[0] == 80
    assert lfcc.shape[1] > 0
    assert not np.isnan(lfcc).any()
    assert not np.isinf(lfcc).any()


def test_classical_feature_vector():
    extractor = ClassicalFeatureExtractor(sample_rate=16000, n_coefficients=30)
    audio = np.random.randn(48000).astype(np.float32)

    vec = extractor.extract_vector(audio)
    # 30 static + 30 delta + 30 delta2 = 90; Mean (90) + Std (90) = 180
    assert vec.shape == (180,)
    assert not np.isnan(vec).any()
