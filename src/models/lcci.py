from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LCCIModule(nn.Module):
    """
    Lateral Competitive Cortical Inhibition (LCCI) with HIS extension.

    HIS: positive-constrained depth scaling via Softplus ensures delta_d > 0,
    preventing sign inversion of inhibition across architectural depths.

    LCCI(x) = ReLU(x + g(x) ⊙ (x ⊙ effective_w_inh))
    effective_w_inh = softplus(depth_scale_raw) * w_inh
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 1)

        self.w_inh = nn.Parameter(torch.ones(channels))
        self.depth_scale_raw = nn.Parameter(torch.tensor(0.0))

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    @property
    def depth_scale(self) -> torch.Tensor:
        """Positive-constrained inhibition scaling: delta_d = softplus(raw)."""
        return F.softplus(self.depth_scale_raw)

    def channel_energy(self, x: torch.Tensor) -> torch.Tensor:
        """E_c = (1/HW) Σ x²  — per-channel mean squared activation."""
        return x.pow(2).mean(dim=(-2, -1))  # (B, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.gate(x).unsqueeze(-1).unsqueeze(-1)
        effective_w_inh = self.depth_scale * self.w_inh
        inhibition = g * (x * effective_w_inh.view(1, -1, 1, 1))
        return torch.relu(x + inhibition)
