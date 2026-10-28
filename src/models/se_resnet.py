"""Squeeze-and-Excitation ResNet (SE-ResNet) for audio deepfake detection (Model B3).

Integrates channel-attention mechanism that dynamically weights frequency
and time channels carrying vocoder-specific synthesis artifacts.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention block."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(channels, max(4, channels // reduction), bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(max(4, channels // reduction), channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        # Squeeze: Global Average Pooling across (freq, time)
        y = x.view(b, c, -1).mean(dim=2)
        # Excitation: Learn channel-wise weights
        y = self.sigmoid(self.fc2(self.relu(self.fc1(y))))
        # Scale input feature maps
        return x * y.view(b, c, 1, 1)


class SEResBlock(nn.Module):
    """Residual block equipped with Squeeze-and-Excitation channel attention."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += residual
        return self.relu(out)


class SEResNet(nn.Module):
    """Compact Squeeze-and-Excitation ResNet (~420k parameters)."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 24,
        fc_hidden: int = 64,
        dropout_rate: float = 0.3,
    ):
        super().__init__()

        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.layer1 = SEResBlock(base_channels, base_channels, stride=1)
        self.layer2 = SEResBlock(base_channels, base_channels * 2, stride=2)
        self.layer3 = SEResBlock(base_channels * 2, base_channels * 4, stride=2)
        self.layer4 = SEResBlock(base_channels * 4, base_channels * 4, stride=2)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        feat_dim = base_channels * 4  # 128
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, fc_hidden),
            nn.BatchNorm1d(fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(fc_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self.init_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.gap(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        logits = self.forward(x)
        return torch.sigmoid(logits)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
