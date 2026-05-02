from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LCCIModule(nn.Module):
    """
    Lightweight Channel-wise Context Integration.

    LCCI(x) = ReLU(x + g(x) ⊙ (x ⊙ w))

    where g(x) is a sigmoid gate computed from a bottleneck projection
    and w is a per-channel learnable scalar weight.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 4)
        # Gate: squeeze → excite → sigmoid
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )
        # Per-channel learnable weight w
        self.w = nn.Parameter(torch.ones(1, channels, 1, 1))

    def channel_energy(self, x: torch.Tensor) -> torch.Tensor:
        """E_c = (1/HW) Σ x²  — per-channel mean squared activation."""
        return x.pow(2).mean(dim=(-2, -1))  # (B, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        g = self.gate(x).view(B, C, 1, 1)  # (B, C, 1, 1)
        context = x * self.w                # x ⊙ w
        return F.relu(x + g * context)      # ReLU(x + g(x) ⊙ (x ⊙ w))
