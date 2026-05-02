from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableGaborLayer(nn.Module):
    """
    Bank of Gabor filters whose frequency/orientation parameters are learned
    end-to-end via reparameterization.  Output is projected back to
    `out_channels` so downstream EfficientNet sees the expected shape.
    """

    def __init__(
        self,
        in_channels: int = 3,
        n_orientations: int = 8,
        n_scales: int = 4,
        kernel_size: int = 15,
        out_channels: int = 3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.n_orientations = n_orientations
        self.n_scales = n_scales
        self.kernel_size = kernel_size
        self.n_filters = n_orientations * n_scales

        # Learnable Gabor parameters — one per filter
        self.theta = nn.Parameter(
            torch.linspace(0, math.pi, n_orientations).repeat(n_scales)
        )
        self.lam = nn.Parameter(
            torch.linspace(4.0, 16.0, n_scales).repeat_interleave(n_orientations)
        )
        self.sigma = nn.Parameter(
            torch.linspace(2.0, 8.0, n_scales).repeat_interleave(n_orientations)
        )
        self.gamma = nn.Parameter(torch.ones(self.n_filters) * 0.5)
        self.psi = nn.Parameter(torch.zeros(self.n_filters))

        pad = kernel_size // 2
        x = torch.arange(-pad, pad + 1, dtype=torch.float32)
        y = torch.arange(-pad, pad + 1, dtype=torch.float32)
        # Grid shape: (K, K, 2)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        # Register as contiguous buffers — meshgrid output shares storage,
        # which causes load_state_dict to fail when restoring checkpoints.
        self.register_buffer("xx", xx.contiguous())
        self.register_buffer("yy", yy.contiguous())

        # 1×1 projection back to out_channels
        self.proj = nn.Conv2d(self.n_filters * in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def _build_kernels(self) -> torch.Tensor:
        """Construct (n_filters, 1, K, K) Gabor kernels from current params."""
        kernels = []
        for i in range(self.n_filters):
            th = self.theta[i]
            lm = self.lam[i].clamp(min=1.0)
            sg = self.sigma[i].clamp(min=0.5)
            gm = self.gamma[i].clamp(min=0.1)
            ps = self.psi[i]

            x_theta = self.xx * th.cos() + self.yy * th.sin()
            y_theta = -self.xx * th.sin() + self.yy * th.cos()

            envelope = torch.exp(-0.5 * (x_theta**2 + gm**2 * y_theta**2) / sg**2)
            carrier = torch.cos(2 * math.pi * x_theta / lm + ps)
            kernel = envelope * carrier
            # Normalize
            kernel = kernel / (kernel.norm() + 1e-8)
            kernels.append(kernel.unsqueeze(0))

        return torch.stack(kernels, dim=0)  # (n_filters, 1, K, K)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        kernels = self._build_kernels()  # (n_filters, 1, K, K)

        # Apply each filter to each input channel independently
        # Reshape x: (B*C, 1, H, W)
        x_flat = x.reshape(B * C, 1, H, W)
        pad = self.kernel_size // 2
        out = F.conv2d(x_flat, kernels, padding=pad)  # (B*C, n_filters, H, W)
        # Reshape to (B, C*n_filters, H, W)
        out = out.reshape(B, C * self.n_filters, H, W)
        out = self.proj(out)  # (B, out_channels, H, W)
        out = self.bn(out)
        return F.relu(out)
