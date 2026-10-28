"""Tests for audio loading and preprocessing pipeline."""

import numpy as np
import pytest
from src.preprocessing.audio_loader import AudioStandardizer, load_and_standardize_audio


def test_standardizer_duration_cropping():
    standardizer = AudioStandardizer(target_sample_rate=16000, target_duration_sec=3.0)
    # Audio of 5 seconds (80,000 samples)
    audio_long = np.random.randn(80000).astype(np.float32)
    processed = standardizer.process(audio_long, orig_sr=16000)

    assert len(processed) == 48000
    assert processed.dtype == np.float32


def test_standardizer_duration_padding():
    standardizer = AudioStandardizer(target_sample_rate=16000, target_duration_sec=3.0)
    # Audio of 1.5 seconds (24,000 samples)
    audio_short = np.random.randn(24000).astype(np.float32)
    processed = standardizer.process(audio_short, orig_sr=16000)

    assert len(processed) == 48000
    assert np.max(np.abs(processed)) <= 1.0


def test_standardizer_stereo_to_mono():
    standardizer = AudioStandardizer(target_sample_rate=16000, target_duration_sec=3.0)
    # Stereo audio (2 channels, 48,000 samples)
    audio_stereo = np.random.randn(2, 48000).astype(np.float32)
    processed = standardizer.process(audio_stereo, orig_sr=16000)

    assert processed.ndim == 1
    assert len(processed) == 48000


def test_standardizer_resampling():
    standardizer = AudioStandardizer(target_sample_rate=16000, target_duration_sec=3.0)
    # Audio recorded at 44.1 kHz (3 seconds = 132,300 samples)
    audio_44k = np.random.randn(132300).astype(np.float32)
    processed = standardizer.process(audio_44k, orig_sr=44100)

    assert len(processed) == 48000
