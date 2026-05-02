from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style channel attention."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.shape
        avg = self.fc(self.avg_pool(x).view(B, C))
        mx = self.fc(self.max_pool(x).view(B, C))
        return torch.sigmoid(avg + mx).view(B, C, 1, 1)


class SpatialAttention(nn.Module):
    """Spatial attention using channel-wise pooling."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        pooled = torch.cat([avg, mx], dim=1)
        return torch.sigmoid(self.bn(self.conv(pooled)))


class DualAttention(nn.Module):
    """
    Channel + Spatial attention applied sequentially (CBAM-style).
    Also computes per-class attention entropy for interpretability.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention()

    @staticmethod
    def attention_entropy(att_map: torch.Tensor) -> torch.Tensor:
        """H = -Σ p log₂ p  over spatial dims (normalized attention map)."""
        B, _, H, W = att_map.shape
        p = att_map.view(B, -1)
        p = p / (p.sum(dim=-1, keepdim=True) + 1e-8)
        entropy = -(p * (p + 1e-8).log2()).sum(dim=-1)
        return entropy  # (B,)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        ca = self.channel_att(x)
        x = x * ca
        sa = self.spatial_att(x)
        x = x * sa
        info = {
            "channel_attention": ca,
            "spatial_attention": sa,
            "attention_entropy": self.attention_entropy(sa),
        }
        return x, info
