"""Classical feature extraction for Model A (SVM / Logistic Regression).

Computes static, delta, and delta-delta coefficients from LFCC, followed by
temporal mean and standard deviation aggregation to produce a fixed 180-d vector.
"""

from typing import Tuple
import numpy as np
import scipy.signal
from .extractors import FeatureExtractor


def compute_deltas(features: np.ndarray, width: int = 5) -> np.ndarray:
    """Computes delta coefficients along time axis using standard regression formula."""
    half_width = width // 2
    padded = np.pad(features, ((0, 0), (half_width, half_width)), mode="edge")
    kernel = np.arange(-half_width, half_width + 1)
    denom = 2 * sum(i**2 for i in range(1, half_width + 1))

    deltas = np.zeros_like(features)
    for t in range(features.shape[1]):
        window = padded[:, t : t + width]
        deltas[:, t] = np.dot(window, kernel) / denom
    return deltas


class ClassicalFeatureExtractor:
    """Extracts a fixed 180-dimensional feature vector per audio clip."""

    def __init__(self, sample_rate: int = 16000, n_coefficients: int = 30):
        self.sample_rate = sample_rate
        self.n_coefficients = n_coefficients
        self.base_extractor = FeatureExtractor(
            sample_rate=sample_rate,
            n_lfcc=n_coefficients,
            f_min=0.0,
            f_max=8000.0,
        )

    def extract_vector(self, audio: np.ndarray) -> np.ndarray:
        """Transforms 1D audio into a fixed 180-dimensional feature vector.

        30 static + 30 delta + 30 double-delta = 90 features.
        Temporal Mean (90) + Temporal Std (90) = 180 dimensions.
        """
        # Shape: (n_coefficients, time_frames)
        static = self.base_extractor.extract_lfcc(audio)[: self.n_coefficients, :]
        delta = compute_deltas(static)
        delta2 = compute_deltas(delta)

        # Concatenate across coefficient dimension: (90, time_frames)
        stacked = np.vstack([static, delta, delta2])

        # Temporal aggregation
        mean_vec = np.mean(stacked, axis=1)
        std_vec = np.std(stacked, axis=1)

        feature_vector = np.concatenate([mean_vec, std_vec])
        return feature_vector.astype(np.float32)
