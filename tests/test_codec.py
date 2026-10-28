"""Tests for FFmpeg audio codec degradation engine."""

import numpy as np
import pytest
from src.robustness.ffmpeg_codec import AudioCodecSimulator, CodecAugmenter


def test_codec_simulator_mp3_and_opus():
    simulator = AudioCodecSimulator()

    # Generate 1 second of 440Hz sine wave (A4 note) at 16kHz
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    sine_audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    # Test MP3 64k
    mp3_audio = simulator.compress_decompress(sine_audio, sample_rate=sr, codec_profile="mp3_64k")
    assert len(mp3_audio) == len(sine_audio)
    assert not np.isnan(mp3_audio).any()

    # Test Opus 24k
    opus_audio = simulator.compress_decompress(sine_audio, sample_rate=sr, codec_profile="opus_24k")
    assert len(opus_audio) == len(sine_audio)
    assert not np.isnan(opus_audio).any()


def test_codec_augmenter():
    augmenter = CodecAugmenter(aug_prob=1.0)
    audio = np.random.randn(16000).astype(np.float32) * 0.5
    augmented, profile = augmenter(audio)

    assert len(augmented) == len(audio)
    assert profile in ["mp3_64k", "mp3_32k", "opus_24k", "opus_16k"]
