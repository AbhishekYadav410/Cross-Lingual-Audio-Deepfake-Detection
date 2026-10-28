"""Hybrid forensic ensemble blending classical and deep architectures.

Combines calibrated probabilities from Model A (LFCC-SVM) and Model B (LCNN / SE-ResNet)
via soft-voting to improve precision, recall, and cross-lingual robustness.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import torch


class ForensicEnsemble:
    """Soft-voting ensemble of classical and neural anti-spoofing detectors."""

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.weights = weights or {"svm": 0.35, "lcnn": 0.35, "se_resnet": 0.30}
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}

    def predict_proba(
        self,
        probs_dict: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Computes weighted probability average.

        Args:
            probs_dict: Dictionary mapping model names (e.g. 'svm', 'lcnn') to 1D arrays of P(Fake).

        Returns:
            1D array of blended fake probabilities.
        """
        blended = np.zeros_like(next(iter(probs_dict.values())), dtype=np.float32)

        for name, probs in probs_dict.items():
            w = self.weights.get(name, 0.0)
            blended += w * np.asarray(probs, dtype=np.float32)

        return np.clip(blended, 0.0, 1.0)
