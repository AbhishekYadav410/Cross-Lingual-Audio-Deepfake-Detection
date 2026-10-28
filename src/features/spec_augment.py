"""SpecAugment data augmentation for spectral representations.

Implements random Frequency Masking and Time Masking on 2D spectrograms,
preventing overfitting to language-specific vowel formants and temporal cadence.
"""

from typing import Tuple, Optional
import numpy as np
import torch


class SpecAugment:
    """Applies frequency and time masking to 2D spectral representations."""

    def __init__(
        self,
        freq_mask_param: int = 12,
        time_mask_param: int = 16,
        n_freq_masks: int = 2,
        n_time_masks: int = 2,
        prob: float = 0.5,
    ):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
        self.prob = prob

    def __call__(self, spec: np.ndarray) -> np.ndarray:
        """Args:
            spec: 2D NumPy array of shape (n_freq_bins, time_frames)
        """
        if np.random.rand() > self.prob:
            return spec

        augmented = spec.copy()
        n_freq, n_time = augmented.shape
        fill_val = float(np.mean(augmented))

        # 1. Frequency masking
        for _ in range(self.n_freq_masks):
            f_len = np.random.randint(1, self.freq_mask_param + 1)
            if f_len >= n_freq:
                continue
            f0 = np.random.randint(0, n_freq - f_len)
            augmented[f0 : f0 + f_len, :] = fill_val

        # 2. Time masking
        for _ in range(self.n_time_masks):
            t_len = np.random.randint(1, self.time_mask_param + 1)
            if t_len >= n_time:
                continue
            t0 = np.random.randint(0, n_time - t_len)
            augmented[:, t0 : t0 + t_len] = fill_val

        return augmented
