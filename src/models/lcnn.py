"""Light CNN (LCNN) architecture with Max-Feature-Map (MFM) activations (Model B2).

Standard anti-spoofing benchmark architecture popularized in ASVspoof challenges.
MFM acts as a competitive filter that suppresses phonetic/ambient noise while
preserving high-frequency vocoder artifacts.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class MFM(nn.Module):
    """Max-Feature-Map activation: MFM(A, B) = max(A, B)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1, out2 = torch.chunk(x, chunks=2, dim=1)
        return torch.max(out1, out2)


class MFMConv2d(nn.Module):
    """Conv2D followed by BatchNorm and MFM activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        # Convolution outputs 2 * out_channels which MFM halves to out_channels
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * 2,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.mfm = MFM()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mfm(self.bn(self.conv(x)))


class MFMLinear(nn.Module):
    """Linear layer followed by MFM activation."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features * 2)
        self.mfm = MFM()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mfm(self.fc(x))


class LCNN(nn.Module):
    """Light CNN with MFM activations (~320k parameters)."""

    def __init__(self, in_channels: int = 1, dropout_rate: float = 0.3):
        super().__init__()

        # Feature extractor backbone
        self.conv1 = MFMConv2d(in_channels, 16, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2a = MFMConv2d(16, 24, kernel_size=3, padding=1)
        self.conv2b = MFMConv2d(24, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3a = MFMConv2d(32, 32, kernel_size=3, padding=1)
        self.conv3b = MFMConv2d(32, 48, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4a = MFMConv2d(48, 48, kernel_size=3, padding=1)
        self.conv4b = MFMConv2d(48, 64, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Classification head with MFM linear compression
        self.fc1 = MFMLinear(64, 64)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: Tensor of shape (Batch, 1, Frequency_bins, Time_frames)
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2b(self.conv2a(x)))
        x = self.pool3(self.conv3b(self.conv3a(x)))
        x = self.pool4(self.conv4b(self.conv4a(x)))

        x = self.gap(x)
        x = torch.flatten(x, 1)

        x = self.dropout(self.fc1(x))
        logits = self.fc2(x)
        return logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns sigmoid probabilities P(Fake | x) in [0.0, 1.0]."""
        self.eval()
        logits = self.forward(x)
        return torch.sigmoid(logits)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
