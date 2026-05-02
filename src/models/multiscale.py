from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleAggregator(nn.Module):
    """
    Parallel depthwise-separable branches at receptive fields RF ∈ {1,3,5,7}
    via dilated 3×3 convolutions.  Outputs are summed element-wise after
    per-branch projection to preserve channel count.
    """

    _RF_TO_DILATION = {1: None, 3: 1, 5: 2, 7: 3}

    def __init__(self, channels: int, receptive_fields: list[int] | None = None):
        super().__init__()
        if receptive_fields is None:
            receptive_fields = [1, 3, 5, 7]
        self.branches = nn.ModuleList()
        for rf in receptive_fields:
            dilation = self._RF_TO_DILATION[rf]
            if dilation is None:
                branch = nn.Sequential(
                    nn.Conv2d(channels, channels, 1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                )
            else:
                branch = nn.Sequential(
                    nn.Conv2d(
                        channels, channels, 3,
                        padding=dilation, dilation=dilation,
                        groups=channels, bias=False,
                    ),
                    nn.Conv2d(channels, channels, 1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                )
            self.branches.append(branch)

        self.fuse = nn.Sequential(
            nn.Conv2d(channels * len(receptive_fields), channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [branch(x) for branch in self.branches]
        return self.fuse(torch.cat(feats, dim=1))
