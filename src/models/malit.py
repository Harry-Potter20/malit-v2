from __future__ import annotations

from typing import Any

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import DualAttention
from .gabor import LearnableGaborLayer
from .lcci import LCCIModule
from .multiscale import MultiScaleAggregator


class MALITV2(nn.Module):
    """
    MALIT V2: Malaria Image Transformer V2

    Pipeline:
        Input (224×224 RGB)
        → LearnableGaborLayer
        → EfficientNet-B0 (partial freeze)
        → LCCIModule (channel-wise)
        → DualAttention (channel + spatial)
        → MultiScaleAggregator (RF={1,3,5,7})
        → Global Average Pool
        → MLP Classifier
    """

    def __init__(
        self,
        num_classes: int = 2,
        gabor_n_orientations: int = 8,
        gabor_n_scales: int = 4,
        gabor_kernel_size: int = 15,
        efficientnet_backbone: str = "efficientnet_b0",
        efficientnet_pretrained: bool = True,
        efficientnet_freeze_blocks: int = 4,
        lcci_reduction: int = 16,
        attention_reduction: int = 16,
        ms_rfs: list[int] | None = None,
    ):
        super().__init__()
        if ms_rfs is None:
            ms_rfs = [1, 3, 5, 7]

        # ── Learnable Gabor preprocessing ───────────────────────────────────
        self.gabor = LearnableGaborLayer(
            in_channels=3,
            n_orientations=gabor_n_orientations,
            n_scales=gabor_n_scales,
            kernel_size=gabor_kernel_size,
            out_channels=3,
        )

        # ── EfficientNet-B0 backbone ─────────────────────────────────────────
        backbone = timm.create_model(
            efficientnet_backbone,
            pretrained=efficientnet_pretrained,
            num_classes=0,
            global_pool="",
        )
        self.backbone = backbone
        self._freeze_backbone(efficientnet_freeze_blocks)
        feat_channels = self._backbone_out_channels(efficientnet_backbone)

        # ── Custom head modules ──────────────────────────────────────────────
        self.lcci = LCCIModule(feat_channels, reduction=lcci_reduction)
        self.dual_attention = DualAttention(feat_channels, reduction=attention_reduction)
        self.multiscale = MultiScaleAggregator(feat_channels, receptive_fields=ms_rfs)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(feat_channels, feat_channels // 2),
            nn.BatchNorm1d(feat_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(feat_channels // 2, num_classes),
        )

    # ────────────────────────────────────────────────────────────────────────
    # Helpers
    # ────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _backbone_out_channels(backbone_name: str) -> int:
        _map = {
            "efficientnet_b0": 1280,
            "efficientnet_b1": 1280,
            "efficientnet_b2": 1408,
            "efficientnet_b3": 1536,
        }
        return _map.get(backbone_name, 1280)

    def _freeze_backbone(self, n_blocks: int) -> None:
        """Freeze stem + first n_blocks MBConv stages of the EfficientNet backbone.

        timm's EfficientNet has a flat children() list where 'blocks' is a single
        child containing all 7 MBConv stages — iterating children() directly would
        freeze all stages at once. Use named_parameters() for stage-level control.
        """
        for name, param in self.backbone.named_parameters():
            if any(name.startswith(p) for p in ("conv_stem", "bn1", "act1")):
                param.requires_grad = False
            elif name.startswith("blocks."):
                try:
                    stage_idx = int(name.split(".")[1])
                    if stage_idx < n_blocks:
                        param.requires_grad = False
                except (IndexError, ValueError):
                    pass

    # ────────────────────────────────────────────────────────────────────────
    # Forward
    # ────────────────────────────────────────────────────────────────────────

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # Gabor preprocessing
        x = self.gabor(x)                          # (B, 3, 224, 224)

        # EfficientNet feature extraction
        feats = self.backbone(x)                   # (B, C, H, W)

        # LCCI
        feats = self.lcci(feats)

        # Dual Attention
        feats, att_info = self.dual_attention(feats)

        # MultiScale Aggregation
        feats = self.multiscale(feats)

        # Global Average Pool + classify
        pooled = self.gap(feats).flatten(1)        # (B, C)
        logits = self.classifier(pooled)

        # Channel energy (post-LCCI, pre-attention) for interpretability
        att_info["channel_energy"] = feats.pow(2).mean(dim=(-2, -1))  # (B, C)

        return logits, att_info

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """512-d embedding after GAP — used for CBR and ensemble analysis."""
        x = self.gabor(x)
        feats = self.backbone(x)
        feats = self.lcci(feats)
        feats, _ = self.dual_attention(feats)
        feats = self.multiscale(feats)
        pooled = self.gap(feats).flatten(1)  # (B, C)
        # Project to 512-d if needed
        if not hasattr(self, "_emb_proj"):
            C = pooled.shape[1]
            self._emb_proj = nn.Linear(C, 512, bias=False).to(pooled.device)
        return self._emb_proj(pooled)        # (B, 512)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward(x)
        return torch.argmax(logits, dim=1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward(x)
        return torch.softmax(logits, dim=1)
