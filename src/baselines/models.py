"""
Baseline model suite for fair comparison against MALIT V2.

All models:
  - Return (logits, {}) from forward() — compatible with Trainer
  - Expose extract_embedding(x) → (B, 512) via lazy projection
  - Accept use_gradient_checkpointing for limited-VRAM environments
  - Use the torchvision weights API (not deprecated pretrained=True)
"""
from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint_sequential
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V3_Small_Weights,
    ResNet50_Weights,
    efficientnet_b0,
    mobilenet_v3_small,
    resnet50,
)

logger = logging.getLogger(__name__)

_EMB_DIM = 512   # target embedding dimension for CBR compatibility


# ── Private attention blocks ──────────────────────────────────────────────────

class _SEBlock(nn.Module):
    """Squeeze-and-Excitation block (channel recalibration)."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.fc(self.pool(x)).view(-1, x.size(1), 1, 1)
        return x * s


class _CBAMBlock(nn.Module):
    """Convolutional Block Attention Module (channel + spatial attention)."""

    def __init__(self, channels: int, reduction: int = 16, sp_kernel: int = 7):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.ch_avg = nn.AdaptiveAvgPool2d(1)
        self.ch_max = nn.AdaptiveMaxPool2d(1)
        self.ch_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(mid, channels, bias=False),
        )
        pad = sp_kernel // 2
        self.sp_conv = nn.Conv2d(2, 1, sp_kernel, padding=pad, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.ch_mlp(self.ch_avg(x))
        max_out = self.ch_mlp(self.ch_max(x))
        ch_att = torch.sigmoid(avg_out + max_out).view(-1, x.size(1), 1, 1)
        x = x * ch_att
        avg_sp = x.mean(dim=1, keepdim=True)
        max_sp, _ = x.max(dim=1, keepdim=True)
        sp_att = torch.sigmoid(self.sp_conv(torch.cat([avg_sp, max_sp], dim=1)))
        return x * sp_att


def _replace_inplace_relu(module: nn.Module) -> None:
    """Replace all ReLU(inplace=True) with ReLU(inplace=False) recursively.

    Required before wrapping ResNet with checkpoint_sequential(use_reentrant=False),
    which cannot handle in-place ops during activation recomputation.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU) and child.inplace:
            setattr(module, name, nn.ReLU(inplace=False))
        else:
            _replace_inplace_relu(child)


# ── EfficientNet-B0 base class (shared by three variants) ────────────────────

class _EfficientNetBase(nn.Module):
    _FEAT_CHANNELS: int = 1280

    def __init__(
        self,
        num_classes: int,
        pretrained: bool,
        use_gradient_checkpointing: bool,
        dropout: float = 0.2,
    ):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = efficientnet_b0(weights=weights)
        self.features: nn.Sequential = backbone.features
        self.use_gc = use_gradient_checkpointing
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self._FEAT_CHANNELS, num_classes)
        self._emb_proj: nn.Linear | None = None

    def _run_features(self, x: torch.Tensor) -> torch.Tensor:
        """Run backbone feature extractor, with optional gradient checkpointing."""
        if self.use_gc and self.training:
            return checkpoint_sequential(self.features, segments=4, input=x, use_reentrant=False)
        return self.features(x)

    @torch.no_grad()
    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        pooled = self.pool(feat).flatten(1)
        if self._emb_proj is None:
            self._emb_proj = nn.Linear(
                self._FEAT_CHANNELS, _EMB_DIM, bias=False, device=pooled.device
            )
        return self._emb_proj(pooled)


# ── Concrete baseline models ──────────────────────────────────────────────────

class EfficientNetBaseline(_EfficientNetBase):
    """EfficientNet-B0 without any custom attention — pure backbone baseline."""

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__(num_classes, pretrained, use_gradient_checkpointing)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        feat = self._run_features(x)
        pooled = self.pool(feat).flatten(1)
        return self.classifier(self.dropout(pooled)), {}


class EfficientNetSEBaseline(_EfficientNetBase):
    """EfficientNet-B0 with Squeeze-and-Excitation attention after the backbone."""

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__(num_classes, pretrained, use_gradient_checkpointing)
        self.se = _SEBlock(self._FEAT_CHANNELS, reduction=16)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        feat = self._run_features(x)
        feat = self.se(feat)
        pooled = self.pool(feat).flatten(1)
        return self.classifier(self.dropout(pooled)), {}

    @torch.no_grad()
    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.se(self.features(x))
        pooled = self.pool(feat).flatten(1)
        if self._emb_proj is None:
            self._emb_proj = nn.Linear(
                self._FEAT_CHANNELS, _EMB_DIM, bias=False, device=pooled.device
            )
        return self._emb_proj(pooled)


class EfficientNetCBAMBaseline(_EfficientNetBase):
    """EfficientNet-B0 with CBAM (channel + spatial) attention after the backbone."""

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__(num_classes, pretrained, use_gradient_checkpointing)
        self.cbam = _CBAMBlock(self._FEAT_CHANNELS, reduction=16)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        feat = self._run_features(x)
        feat = self.cbam(feat)
        pooled = self.pool(feat).flatten(1)
        return self.classifier(self.dropout(pooled)), {}

    @torch.no_grad()
    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.cbam(self.features(x))
        pooled = self.pool(feat).flatten(1)
        if self._emb_proj is None:
            self._emb_proj = nn.Linear(
                self._FEAT_CHANNELS, _EMB_DIM, bias=False, device=pooled.device
            )
        return self._emb_proj(pooled)


class ResNet50Baseline(nn.Module):
    """ResNet-50 fine-tuned baseline."""

    _FEAT_CHANNELS: int = 2048

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        net = resnet50(weights=weights)
        # Inplace ReLUs must be replaced for checkpoint_sequential(use_reentrant=False)
        _replace_inplace_relu(net)
        self.features = nn.Sequential(
            net.conv1, net.bn1, net.relu, net.maxpool,
            net.layer1, net.layer2, net.layer3, net.layer4,
        )
        self.use_gc = use_gradient_checkpointing
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(self._FEAT_CHANNELS, num_classes)
        self._emb_proj: nn.Linear | None = None

    def _run_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_gc and self.training:
            return checkpoint_sequential(self.features, segments=4, input=x, use_reentrant=False)
        return self.features(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        feat = self._run_features(x)
        pooled = self.pool(feat).flatten(1)
        return self.classifier(self.dropout(pooled)), {}

    @torch.no_grad()
    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(self.features(x)).flatten(1)
        if self._emb_proj is None:
            self._emb_proj = nn.Linear(
                self._FEAT_CHANNELS, _EMB_DIM, bias=False, device=pooled.device
            )
        return self._emb_proj(pooled)


class MobileNetV3Baseline(nn.Module):
    """MobileNetV3-Small fine-tuned baseline (lightweight, low-VRAM)."""

    _FEAT_CHANNELS: int = 576

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        net = mobilenet_v3_small(weights=weights)
        self.features: nn.Sequential = net.features
        self.use_gc = use_gradient_checkpointing
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(self._FEAT_CHANNELS, num_classes)
        self._emb_proj: nn.Linear | None = None

    def _run_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_gc and self.training:
            return checkpoint_sequential(self.features, segments=4, input=x, use_reentrant=False)
        return self.features(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        feat = self._run_features(x)
        pooled = self.pool(feat).flatten(1)
        return self.classifier(self.dropout(pooled)), {}

    @torch.no_grad()
    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(self.features(x)).flatten(1)
        if self._emb_proj is None:
            self._emb_proj = nn.Linear(
                self._FEAT_CHANNELS, _EMB_DIM, bias=False, device=pooled.device
            )
        return self._emb_proj(pooled)


# ── Factory ───────────────────────────────────────────────────────────────────

_REGISTRY: dict[str, type] = {
    "efficientnet": EfficientNetBaseline,
    "resnet50":     ResNet50Baseline,
    "mobilenetv3":  MobileNetV3Baseline,
    "senet":        EfficientNetSEBaseline,
    "cbam":         EfficientNetCBAMBaseline,
}

ALL_BASELINE_NAMES: list[str] = list(_REGISTRY.keys())


def build_baseline(
    name: str,
    num_classes: int = 2,
    pretrained: bool = True,
    use_gradient_checkpointing: bool = False,
) -> nn.Module:
    """Instantiate a baseline model by name.

    Parameters
    ----------
    name : one of "efficientnet", "resnet50", "mobilenetv3", "senet", "cbam"
    pretrained : use ImageNet-pretrained weights (set False for tests)
    use_gradient_checkpointing : recompute activations during backward to save VRAM
    """
    if name not in _REGISTRY:
        raise ValueError(f"Unknown baseline '{name}'. Choose from {list(_REGISTRY)}")
    return _REGISTRY[name](
        num_classes=num_classes,
        pretrained=pretrained,
        use_gradient_checkpointing=use_gradient_checkpointing,
    )
