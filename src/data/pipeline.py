from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .dataset import MalariaDataset, build_transforms
from .deduplication import deduplicate

logger = logging.getLogger(__name__)


class DataPipeline:
    """
    End-to-end data preparation: deduplication → stratified split → DataLoaders.

    Splits are fixed by seed=42 so they are identical across all training seeds.
    This prevents data leakage across experiments.
    """

    def __init__(
        self,
        root: str | Path,
        image_size: int = 224,
        train_frac: float = 0.70,
        val_frac: float = 0.15,
        split_seed: int = 42,
        hash_size: int = 16,
        hamming_threshold: int = 4,
        batch_size: int = 32,
        num_workers: int = 4,
        deduplicate_data: bool = True,
    ):
        self.root = Path(root)
        self.image_size = image_size
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = round(1.0 - train_frac - val_frac, 6)
        self.split_seed = split_seed
        self.hash_size = hash_size
        self.hamming_threshold = hamming_threshold
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.deduplicate_data = deduplicate_data

        self._train_ds: MalariaDataset | None = None
        self._val_ds: MalariaDataset | None = None
        self._test_ds: MalariaDataset | None = None
        self._split_indices: dict[str, list[int]] | None = None

    # ── Public API ──────────────────────────────────────────────────────────

    def prepare(self) -> "DataPipeline":
        """Load, deduplicate, and split. Call once before get_loaders()."""
        full_ds = MalariaDataset.from_directory(self.root)
        logger.info("Loaded %d images from %s", len(full_ds), self.root)

        paths, labels = full_ds.image_paths, full_ds.labels

        if self.deduplicate_data:
            kept, removed = deduplicate(paths, self.hash_size, self.hamming_threshold)
            kept_set = set(kept)
            indices = [i for i, p in enumerate(paths) if p in kept_set]
            paths = [paths[i] for i in indices]
            labels = [labels[i] for i in indices]
            logger.info("After dedup: %d images (%d removed)", len(paths), len(removed))

        labels_arr = np.array(labels)
        all_idx = np.arange(len(paths))

        # Fixed split: train+val+test, stratified, seed=42 (CRITICAL)
        train_val_idx, test_idx = train_test_split(
            all_idx,
            test_size=self.test_frac,
            stratify=labels_arr,
            random_state=self.split_seed,
        )
        relative_val = self.val_frac / (self.train_frac + self.val_frac)
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=relative_val,
            stratify=labels_arr[train_val_idx],
            random_state=self.split_seed,
        )

        self._split_indices = {
            "train": train_idx.tolist(),
            "val": val_idx.tolist(),
            "test": test_idx.tolist(),
        }

        train_tf = build_transforms(self.image_size, is_train=True)
        eval_tf = build_transforms(self.image_size, is_train=False)

        base_ds = MalariaDataset(paths, labels, transform=None)
        self._train_ds = base_ds.subset(train_idx.tolist(), train_tf)
        self._val_ds = base_ds.subset(val_idx.tolist(), eval_tf)
        self._test_ds = base_ds.subset(test_idx.tolist(), eval_tf)

        logger.info(
            "Split sizes — train: %d  val: %d  test: %d",
            len(self._train_ds), len(self._val_ds), len(self._test_ds),
        )
        return self

    def get_loaders(
        self, device: "torch.device | None" = None
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        import torch
        assert self._train_ds is not None, "Call prepare() first."
        # pin_memory is only supported on CUDA; MPS and CPU must not use it
        if device is not None:
            pin_memory = device.type == "cuda"
        else:
            pin_memory = torch.cuda.is_available()
        train_loader = DataLoader(
            self._train_ds, batch_size=self.batch_size,
            shuffle=True, num_workers=self.num_workers, pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            self._val_ds, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers, pin_memory=pin_memory,
        )
        test_loader = DataLoader(
            self._test_ds, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers, pin_memory=pin_memory,
        )
        return train_loader, val_loader, test_loader

    @property
    def split_indices(self) -> dict[str, list[int]]:
        assert self._split_indices is not None, "Call prepare() first."
        return self._split_indices

    @property
    def datasets(self) -> dict[str, MalariaDataset]:
        assert self._train_ds is not None, "Call prepare() first."
        return {"train": self._train_ds, "val": self._val_ds, "test": self._test_ds}
