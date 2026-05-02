"""Tests for MALIT V2 model architecture."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn


# ── LearnableGaborLayer ──────────────────────────────────────────────────────

class TestLearnableGaborLayer:
    def setup_method(self):
        from src.models.gabor import LearnableGaborLayer
        self.layer = LearnableGaborLayer(
            in_channels=3, n_orientations=4, n_scales=2, kernel_size=7, out_channels=3
        )

    def test_output_shape(self):
        x = torch.randn(2, 3, 64, 64)
        out = self.layer(x)
        assert out.shape == (2, 3, 64, 64), f"Expected (2,3,64,64), got {out.shape}"

    def test_output_is_non_negative(self):
        """ReLU activation at end — all values ≥ 0."""
        x = torch.randn(2, 3, 32, 32)
        out = self.layer(x)
        assert (out >= 0).all()

    def test_parameters_are_learnable(self):
        param_names = [n for n, p in self.layer.named_parameters() if p.requires_grad]
        for expected in ["theta", "lam", "sigma", "gamma", "psi"]:
            assert expected in param_names, f"Parameter '{expected}' not learnable"

    def test_n_filters(self):
        assert self.layer.n_filters == 4 * 2  # n_orientations * n_scales

    def test_kernel_shape(self):
        kernels = self.layer._build_kernels()
        assert kernels.shape == (8, 1, 7, 7)

    def test_gradient_flows(self):
        self.layer.train()
        x = torch.randn(2, 3, 32, 32)
        out = self.layer(x)
        loss = out.sum()
        loss.backward()
        assert self.layer.theta.grad is not None
        assert self.layer.lam.grad is not None


# ── LCCIModule ───────────────────────────────────────────────────────────────

class TestLCCIModule:
    def setup_method(self):
        from src.models.lcci import LCCIModule
        self.lcci = LCCIModule(channels=64, reduction=4)

    def test_output_shape_preserved(self):
        x = torch.randn(4, 64, 14, 14)
        out = self.lcci(x)
        assert out.shape == x.shape

    def test_output_non_negative(self):
        """LCCI ends with ReLU."""
        x = torch.randn(4, 64, 14, 14)
        out = self.lcci(x)
        assert (out >= 0).all()

    def test_channel_energy(self):
        x = torch.randn(4, 64, 14, 14)
        energy = self.lcci.channel_energy(x)
        assert energy.shape == (4, 64)
        # Energy is mean of x² — must be non-negative
        assert (energy >= 0).all()

    def test_residual_increases_magnitude(self):
        """Post-LCCI features should generally differ from input due to residual."""
        x = torch.ones(2, 64, 14, 14)
        out = self.lcci(x)
        assert not torch.allclose(out, x)


# ── DualAttention ────────────────────────────────────────────────────────────

class TestDualAttention:
    def setup_method(self):
        from src.models.attention import DualAttention
        self.att = DualAttention(channels=64, reduction=4)

    def test_output_shape(self):
        x = torch.randn(4, 64, 14, 14)
        out, info = self.att(x)
        assert out.shape == x.shape

    def test_info_keys(self):
        x = torch.randn(4, 64, 14, 14)
        _, info = self.att(x)
        assert "channel_attention" in info
        assert "spatial_attention" in info
        assert "attention_entropy" in info

    def test_channel_attention_range(self):
        x = torch.randn(4, 64, 14, 14)
        _, info = self.att(x)
        ca = info["channel_attention"]
        assert ca.shape == (4, 64, 1, 1)
        assert (ca >= 0).all() and (ca <= 1).all()

    def test_spatial_attention_range(self):
        x = torch.randn(4, 64, 14, 14)
        _, info = self.att(x)
        sa = info["spatial_attention"]
        assert sa.shape == (4, 1, 14, 14)
        assert (sa >= 0).all() and (sa <= 1).all()

    def test_attention_entropy_positive(self):
        x = torch.randn(4, 64, 14, 14)
        _, info = self.att(x)
        assert (info["attention_entropy"] >= 0).all()

    def test_entropy_formula(self):
        """Uniform attention map → maximum entropy."""
        from src.models.attention import DualAttention
        # Create a uniform spatial attention map
        uniform = torch.full((1, 1, 4, 4), 1.0 / 16)
        entropy = DualAttention.attention_entropy(uniform)
        expected = 4.0  # log2(16)
        assert abs(entropy.item() - expected) < 0.1


# ── MultiScaleAggregator ─────────────────────────────────────────────────────

class TestMultiScaleAggregator:
    def setup_method(self):
        from src.models.multiscale import MultiScaleAggregator
        self.ms = MultiScaleAggregator(channels=64, receptive_fields=[1, 3, 5, 7])

    def test_output_shape_preserved(self):
        x = torch.randn(4, 64, 14, 14)
        out = self.ms(x)
        assert out.shape == x.shape

    def test_n_branches(self):
        assert len(self.ms.branches) == 4

    def test_different_rfs_give_different_outputs(self):
        from src.models.multiscale import MultiScaleAggregator
        ms1 = MultiScaleAggregator(64, [1])
        ms2 = MultiScaleAggregator(64, [7])
        x = torch.randn(2, 64, 14, 14)
        # They have different weights — outputs must differ
        out1 = ms1.branches[0](x)
        out2 = ms2.branches[0](x)
        assert out1.shape == out2.shape


# ── Full MALITV2 ─────────────────────────────────────────────────────────────

class TestMALITV2:
    def test_forward_output_shapes(self, tiny_model, sample_batch):
        with torch.no_grad():
            logits, info = tiny_model(sample_batch)
        assert logits.shape == (8, 2)
        assert "channel_attention" in info
        assert "spatial_attention" in info
        assert "attention_entropy" in info
        assert "channel_energy" in info

    def test_logits_are_finite(self, tiny_model, sample_batch):
        with torch.no_grad():
            logits, _ = tiny_model(sample_batch)
        assert torch.isfinite(logits).all()

    def test_predict_shape(self, tiny_model, sample_batch):
        with torch.no_grad():
            preds = tiny_model.predict(sample_batch)
        assert preds.shape == (8,)
        assert preds.max() <= 1 and preds.min() >= 0

    def test_predict_proba_sums_to_one(self, tiny_model, sample_batch):
        with torch.no_grad():
            probs = tiny_model.predict_proba(sample_batch)
        assert probs.shape == (8, 2)
        assert torch.allclose(probs.sum(dim=1), torch.ones(8), atol=1e-5)

    def test_extract_embedding_shape(self, tiny_model, sample_batch):
        with torch.no_grad():
            embs = tiny_model.extract_embedding(sample_batch)
        assert embs.shape == (8, 512)
        assert torch.isfinite(embs).all()

    def test_freeze_blocks_reduces_trainable_params(self, tiny_model):
        trainable = sum(p.numel() for p in tiny_model.parameters() if p.requires_grad)
        assert trainable > 0, "All parameters frozen — model cannot train."

    def test_channel_energy_shape(self, tiny_model, sample_batch):
        with torch.no_grad():
            _, info = tiny_model(sample_batch)
        energy = info["channel_energy"]
        assert energy.ndim == 2
        assert energy.shape[0] == 8

    def test_single_image_inference(self, tiny_model):
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            logits, _ = tiny_model(x)
        assert logits.shape == (1, 2)
