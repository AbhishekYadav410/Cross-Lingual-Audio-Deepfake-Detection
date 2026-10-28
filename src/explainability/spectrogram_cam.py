"""Spectrogram occlusion attribution and faithfulness verification (RQ4).

Implements:
1. Time-frequency grid occlusion attribution.
2. Deletion faithfulness testing (measuring prediction decay as salient bins are masked).
"""

from typing import Tuple, Dict, Any, List
import numpy as np
import torch


class SpectrogramOcclusionExplainer:
    """Explains 2D-CNN predictions via grid-based time-frequency occlusion."""

    def __init__(self, model: torch.nn.Module, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()

    def attribute(
        self,
        spectrogram: np.ndarray,
        patch_size: Tuple[int, int] = (10, 10),
    ) -> np.ndarray:
        """Computes saliency map via sliding patch occlusion.

        Args:
            spectrogram: 2D array of shape (n_freq_bins, time_frames).
            patch_size: (freq_patch, time_patch) tuple.

        Returns:
            2D heatmap array of shape (n_freq_bins, time_frames).
        """
        n_freq, n_time = spectrogram.shape
        f_patch, t_patch = patch_size

        # Base prediction
        base_tensor = torch.from_numpy(spectrogram).unsqueeze(0).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            base_prob = float(self.model.predict_proba(base_tensor)[0, 0])

        saliency_map = np.zeros_like(spectrogram, dtype=np.float32)
        baseline_val = float(np.mean(spectrogram))

        # Slide occlusion patch across time and frequency
        for f in range(0, n_freq, f_patch):
            for t in range(0, n_time, t_patch):
                occluded = spectrogram.copy()
                f_end = min(f + f_patch, n_freq)
                t_end = min(t + t_patch, n_time)
                occluded[f:f_end, t:t_end] = baseline_val

                occ_tensor = torch.from_numpy(occluded).unsqueeze(0).unsqueeze(0).float().to(self.device)
                with torch.no_grad():
                    occ_prob = float(self.model.predict_proba(occ_tensor)[0, 0])

                # Attribution = Drop in fake probability when region is removed
                attribution = max(0.0, base_prob - occ_prob)
                saliency_map[f:f_end, t:t_end] = attribution

        # Normalize heatmap to [0.0, 1.0]
        max_val = np.max(saliency_map)
        if max_val > 0:
            saliency_map /= max_val

        return saliency_map

    def verify_faithfulness(
        self,
        spectrogram: np.ndarray,
        saliency_map: np.ndarray,
        fractions: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5],
    ) -> Dict[str, float]:
        """Deletion faithfulness curve: tests score decay when top-k salient bins are masked."""
        flat_saliency = saliency_map.ravel()
        flat_spec = spectrogram.ravel()
        sorted_indices = np.argsort(flat_saliency)[::-1]  # Most salient first
        baseline_val = float(np.mean(spectrogram))

        base_tensor = torch.from_numpy(spectrogram).unsqueeze(0).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            base_prob = float(self.model.predict_proba(base_tensor)[0, 0])

        scores = {"base_prob": round(base_prob, 4)}

        for frac in fractions:
            k = int(len(flat_spec) * frac)
            masked_flat = flat_spec.copy()
            masked_flat[sorted_indices[:k]] = baseline_val
            masked_spec = masked_flat.reshape(spectrogram.shape)

            t_masked = torch.from_numpy(masked_spec).unsqueeze(0).unsqueeze(0).float().to(self.device)
            with torch.no_grad():
                prob_k = float(self.model.predict_proba(t_masked)[0, 0])
            scores[f"prob_drop_{int(frac*100)}pct"] = round(base_prob - prob_k, 4)

        return scores
