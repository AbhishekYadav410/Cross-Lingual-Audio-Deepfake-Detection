"""Lightweight 2D-CNN architecture for audio deepfake detection (Model B).

Compact 4-layer convolutional network (~350k parameters) designed for
low-latency edge inference on 2D spectral representations (Log-Mel / LFCC).
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Convolutional building block: Conv2D -> BatchNorm -> ReLU -> MaxPool2D."""

    def __init__(self, in_channels: int, out_channels: int, pool_size: Tuple[int, int] = (2, 2)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=pool_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.relu(self.bn(self.conv(x))))


class LightweightAudioCNN(nn.Module):
    """Compact 4-layer 2D-CNN with Global Average Pooling (~350k params)."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 16,
        fc_hidden: int = 64,
        dropout_rate: float = 0.3,
    ):
        super().__init__()

        # Conv backbone
        self.block1 = ConvBlock(in_channels, base_channels)             # 1 -> 16
        self.block2 = ConvBlock(base_channels, base_channels * 2)        # 16 -> 32
        self.block3 = ConvBlock(base_channels * 2, base_channels * 4)    # 32 -> 64
        self.block4 = ConvBlock(base_channels * 4, base_channels * 8)    # 64 -> 128

        # Global Average Pooling ensures variable time frame invariance
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Compact classification head
        feature_dim = base_channels * 8  # 128
        self.fc1 = nn.Linear(feature_dim, fc_hidden)
        self.bn_fc = nn.BatchNorm1d(fc_hidden)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(fc_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (Batch, Channels, Frequency_bins, Time_frames).

        Returns:
            Logits of shape (Batch, 1).
        """
        if x.dim() == 3:
            # Add channel dimension if missing: (B, F, T) -> (B, 1, F, T)
            x = x.unsqueeze(1)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.gap(x)
        x = torch.flatten(x, 1)

        x = self.dropout(self.relu(self.bn_fc(self.fc1(x))))
        logits = self.fc2(x)
        return logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns sigmoid probabilities P(Fake | x) in [0.0, 1.0]."""
        self.eval()
        logits = self.forward(x)
        return torch.sigmoid(logits)

    def count_parameters(self) -> int:
        """Returns total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
