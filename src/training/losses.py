"""Loss functions for audio deepfake detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothedBCEWithLogitsLoss(nn.Module):
    """Binary Cross Entropy with label smoothing for anti-spoofing."""

    def __init__(self, label_smoothing: float = 0.05, pos_weight: float = 1.0):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.pos_weight = torch.tensor([pos_weight]) if pos_weight != 1.0 else None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Args:
            logits: (Batch, 1) or (Batch,)
            targets: (Batch, 1) or (Batch,) binary targets in {0, 1}
        """
        targets = targets.view_as(logits).float()
        # Smooth targets: 0 -> epsilon, 1 -> 1 - epsilon
        targets_smooth = targets * (1.0 - 2.0 * self.label_smoothing) + self.label_smoothing

        weight = self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        loss = F.binary_cross_entropy_with_logits(logits, targets_smooth, pos_weight=weight)
        return loss
