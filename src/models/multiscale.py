from __future__ import annotations

import torch
import torch.nn as nn

from .lcci import LCCIModule


class AggregatorBranch(nn.Module):
    """
    Single multi-scale branch: dilated depthwise-separable conv + HIS LCCI (depth 3).
    The `.lcci` attribute exposes the branch's inhibition module for HIS analysis.
    """

    def __init__(self, channels: int, dilation: int | None, reduction: int = 16):
        super().__init__()
        if dilation is None:
            self.conv = nn.Sequential(
                nn.Conv2d(channels, channels, 1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(
                    channels, channels, 3,
                    padding=dilation, dilation=dilation,
                    groups=channels, bias=False,
                ),
                nn.Conv2d(channels, channels, 1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )
        self.lcci = LCCIModule(channels, reduction=reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lcci(self.conv(x))


class MultiScaleAggregator(nn.Module):
    """
    Parallel AggregatorBranches at receptive fields RF ∈ {1,3,5,7} via dilated 3×3
    convolutions. Each branch contains an LCCIModule (HIS depth 3). Outputs are
    concatenated and fused by a 1×1 conv to preserve channel count.
    """

    _RF_TO_DILATION = {1: None, 3: 1, 5: 2, 7: 3}

    def __init__(
        self,
        channels: int,
        receptive_fields: list[int] | None = None,
        reduction: int = 16,
    ):
        super().__init__()
        if receptive_fields is None:
            receptive_fields = [1, 3, 5, 7]
        self.branches = nn.ModuleList(
            AggregatorBranch(channels, self._RF_TO_DILATION[rf], reduction=reduction)
            for rf in receptive_fields
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * len(receptive_fields), channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [branch(x) for branch in self.branches]
        return self.fuse(torch.cat(feats, dim=1))
