"""Shared pytest fixtures for all MALIT V2 tests."""

from __future__ import annotations

import warnings
from pathlib import Path
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset


def pytest_configure(config: pytest.Config) -> None:
    # MPS does not support pin_memory; suppress this known-safe warning
    warnings.filterwarnings(
        "ignore",
        message=".*pin_memory.*MPS.*",
        category=UserWarning,
    )


# ── Tiny synthetic model ────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def tiny_model():
    """MALITV2 with minimal settings — no pretrained weights, CPU-only."""
    from src.models.malit import MALITV2
    return MALITV2(
        num_classes=2,
        gabor_n_orientations=2,
        gabor_n_scales=2,
        gabor_kernel_size=7,
        efficientnet_backbone="efficientnet_b0",
        efficientnet_pretrained=False,
        efficientnet_freeze_blocks=0,
        lcci_reduction=4,
        attention_reduction=4,
        ms_rfs=[1, 3],
    ).eval()


# ── Synthetic image tensors ─────────────────────────────────────────────────

@pytest.fixture(scope="session")
def sample_batch():
    """8 random 3×224×224 images."""
    torch.manual_seed(0)
    return torch.randn(8, 3, 224, 224)


@pytest.fixture(scope="session")
def sample_labels():
    return torch.tensor([0, 1, 0, 1, 1, 0, 1, 0])


@pytest.fixture(scope="session")
def sample_loader(sample_batch, sample_labels):
    ds = TensorDataset(sample_batch, sample_labels)
    return DataLoader(ds, batch_size=4, shuffle=False)


# ── Synthetic predictions ───────────────────────────────────────────────────

@pytest.fixture
def binary_labels():
    np.random.seed(42)
    return list(np.random.randint(0, 2, 100))


@pytest.fixture
def perfect_preds(binary_labels):
    return list(binary_labels)


@pytest.fixture
def random_preds():
    np.random.seed(99)
    return list(np.random.randint(0, 2, 100))


@pytest.fixture
def seed_preds_dict(binary_labels):
    np.random.seed(0)
    return {
        42: [int(x) for x in np.random.randint(0, 2, 100)],
        123: [int(x) for x in np.random.randint(0, 2, 100)],
        456: [int(x) for x in np.random.randint(0, 2, 100)],
    }
