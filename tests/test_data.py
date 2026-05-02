"""Tests for data deduplication, dataset, and pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image


def _create_fake_dataset(root: Path, n_per_class: int = 10) -> None:
    """Create a tiny fake malaria dataset on disk."""
    for class_name in ["Uninfected", "Parasitized"]:
        d = root / class_name
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            img = Image.fromarray(
                np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            )
            img.save(d / f"cell_{i:04d}.png")


# ── MalariaDataset ──────────────────────────────────────────────────────────

class TestMalariaDataset:
    def test_from_directory_loads_all(self):
        from src.data.dataset import MalariaDataset
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _create_fake_dataset(root, n_per_class=5)
            ds = MalariaDataset.from_directory(root)
        assert len(ds) == 10

    def test_label_balance(self):
        from src.data.dataset import MalariaDataset
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _create_fake_dataset(root, n_per_class=5)
            ds = MalariaDataset.from_directory(root)
        labels = ds.labels
        assert labels.count(0) == 5
        assert labels.count(1) == 5

    def test_getitem_returns_tensor(self):
        from src.data.dataset import MalariaDataset, build_transforms
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _create_fake_dataset(root, n_per_class=3)
            ds = MalariaDataset.from_directory(root, transform=build_transforms(64, is_train=False))
            # Access inside the context so paths are still valid
            img, label = ds[0]
        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, 64, 64)
        assert label in [0, 1]

    def test_subset(self):
        from src.data.dataset import MalariaDataset
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _create_fake_dataset(root, n_per_class=5)
            ds = MalariaDataset.from_directory(root)
        sub = ds.subset([0, 1, 2])
        assert len(sub) == 3

    def test_class_names(self):
        from src.data.dataset import MalariaDataset
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _create_fake_dataset(root, n_per_class=2)
            ds = MalariaDataset.from_directory(root)
        assert "Uninfected" in ds.class_names
        assert "Parasitized" in ds.class_names


# ── Deduplication ───────────────────────────────────────────────────────────

class TestDeduplication:
    def test_no_duplicates_keeps_all(self):
        from src.data.deduplication import deduplicate
        np.random.seed(7)
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for i in range(5):
                p = Path(tmp) / f"img_{i}.png"
                # Highly distinct images: checkerboard patterns at different scales
                arr = np.zeros((64, 64, 3), dtype=np.uint8)
                step = 4 * (i + 1)
                for r in range(64):
                    for c in range(64):
                        if (r // step + c // step) % 2 == 0:
                            arr[r, c] = [255, 0, 0]
                        else:
                            arr[r, c] = [0, 0, i * 50]
                Image.fromarray(arr).save(p)
                paths.append(p)
            kept, removed = deduplicate(paths, hash_size=8, hamming_threshold=4)
        assert len(kept) == 5
        assert len(removed) == 0

    def test_exact_duplicates_removed(self):
        from src.data.deduplication import deduplicate
        with tempfile.TemporaryDirectory() as tmp:
            arr = np.full((32, 32, 3), 128, dtype=np.uint8)
            paths = []
            for i in range(3):
                p = Path(tmp) / f"dup_{i}.png"
                Image.fromarray(arr).save(p)
                paths.append(p)
            # Add one unique image
            unique = Path(tmp) / "unique.png"
            Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(unique)
            paths.append(unique)
            kept, removed = deduplicate(paths, hash_size=8, hamming_threshold=4)
        assert len(kept) <= 3  # at most 2 of the 3 duplicates + unique
        assert len(kept) + len(removed) == 4


# ── DataPipeline ─────────────────────────────────────────────────────────────

class TestDataPipeline:
    def _make_pipeline(self, root: Path, n: int = 20):
        from src.data.pipeline import DataPipeline
        return DataPipeline(
            root=root,
            image_size=64,
            train_frac=0.70,
            val_frac=0.15,
            split_seed=42,
            batch_size=4,
            num_workers=0,
            deduplicate_data=False,
        )

    def test_prepare_creates_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _create_fake_dataset(root, n_per_class=20)
            pipe = self._make_pipeline(root).prepare()
        si = pipe.split_indices
        assert "train" in si and "val" in si and "test" in si
        total = len(si["train"]) + len(si["val"]) + len(si["test"])
        assert total == 40

    def test_no_overlap_between_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _create_fake_dataset(root, n_per_class=20)
            pipe = self._make_pipeline(root).prepare()
        si = pipe.split_indices
        train_s = set(si["train"])
        val_s = set(si["val"])
        test_s = set(si["test"])
        assert len(train_s & val_s) == 0, "Train/val overlap!"
        assert len(train_s & test_s) == 0, "Train/test overlap!"
        assert len(val_s & test_s) == 0, "Val/test overlap!"

    def test_split_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _create_fake_dataset(root, n_per_class=20)
            si1 = self._make_pipeline(root).prepare().split_indices
            si2 = self._make_pipeline(root).prepare().split_indices
        assert si1["train"] == si2["train"]
        assert si1["test"] == si2["test"]

    def test_loaders_yield_correct_shapes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _create_fake_dataset(root, n_per_class=20)
            pipe = self._make_pipeline(root).prepare()
            train_l, val_l, test_l = pipe.get_loaders()
            imgs, labels = next(iter(train_l))
        assert imgs.shape == (4, 3, 64, 64)
        assert labels.shape == (4,)

    def test_stratified_class_balance(self):
        """Both train/val/test splits should contain both classes."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _create_fake_dataset(root, n_per_class=20)
            pipe = self._make_pipeline(root).prepare()
        for split_name, ds in pipe.datasets.items():
            labels = ds.labels
            assert 0 in labels, f"Class 0 missing from {split_name}"
            assert 1 in labels, f"Class 1 missing from {split_name}"
