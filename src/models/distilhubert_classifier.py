"""DistilHuBERT-based Audio Deepfake Detector (Model B4).

Leverages self-supervised speech representations from ntu-spml/distilhubert
(~23.6M parameters) combined with statistical pooling (mean + std) and an
anti-spoofing classification head. Supports FP16 mixed precision and fine-tuning.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import HubertModel, AutoConfig


class StatisticalPooling(nn.Module):
    """Computes mean and standard deviation across the temporal dimension."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: Tensor of shape (Batch, Time_steps, Hidden_dim)
        Returns:
            Tensor of shape (Batch, Hidden_dim * 2)
        """
        mean = torch.mean(x, dim=1)
        var = torch.var(x, dim=1, unbiased=False)
        std = torch.sqrt(torch.clamp(var, min=self.eps))
        return torch.cat([mean, std], dim=1)


class DistilHuBERTClassifier(nn.Module):
    """Downstream Audio Deepfake Detector using pre-trained DistilHuBERT.
    
    Architecture:
      1. Raw waveform input (16 kHz mono, e.g. 48,000 samples for 3.0 seconds)
      2. ntu-spml/distilhubert backbone (768 hidden dimension)
      3. Statistical Pooling (mean + std -> 1536 dimension)
      4. Multi-layer projection and anti-spoofing classification head
    """

    def __init__(
        self,
        pretrained_model_name: str = "ntu-spml/distilhubert",
        freeze_feature_extractor: bool = True,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        self.pretrained_model_name = pretrained_model_name
        self.backbone = HubertModel.from_pretrained(pretrained_model_name)

        if freeze_feature_extractor:
            # Freeze the convolutional feature encoder to conserve VRAM and avoid overfitting
            self.backbone.feature_extractor._freeze_parameters()

        hidden_dim = self.backbone.config.hidden_size  # 768 for distilhubert
        self.pool = StatisticalPooling()

        # Classification Head (1536 -> 256 -> 64 -> 1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate / 2.0),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: Raw waveform tensor of shape (Batch, Num_samples) or (Batch, 1, Num_samples)
        Returns:
            Logits of shape (Batch, 1)
        """
        if x.dim() == 3:
            x = x.squeeze(1)

        # Forward pass through DistilHuBERT backbone
        outputs = self.backbone(x)
        features = outputs.last_hidden_state  # (Batch, Sequence_length, 768)

        # Statistical pooling across temporal frames
        pooled = self.pool(features)  # (Batch, 1536)

        # Anti-spoofing classification head
        logits = self.classifier(pooled)
        return logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns sigmoid probabilities P(Fake | x) in [0.0, 1.0]."""
        self.eval()
        logits = self.forward(x)
        return torch.sigmoid(logits)

    def count_parameters(self) -> Tuple[int, int]:
        """Returns (trainable_params, total_params)."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total
