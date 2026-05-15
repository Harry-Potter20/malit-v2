#!/usr/bin/env python3
"""
MALIT-H Chest X-Ray Validation Pipeline

SECURITY: This file MUST NOT contain any API keys or credentials.

Trains MALITV2 on the Chest X-Ray Pneumonia dataset (Kaggle: chest-xray-pneumonia,
Paul Mooney, 5863 images, binary NORMAL/PNEUMONIA) and validates that the DAIS depth
gradient (δ₁ > δ₂) emerges on a completely different imaging modality.

Usage (Kaggle):
    !PYTHONPATH=. python scripts/run_xray_pipeline.py

Environment variables:
    XRAY_DATASET_PATH — path to chest_xray root (auto-detected on Kaggle if unset)
    OUTPUT_DIR        — base results dir (default: /kaggle/working/results on Kaggle, else results)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image as PILImage
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("xray_pipeline")

SEEDS = [42, 123, 456]
EPOCHS = 30
EARLY_STOP_PATIENCE = 3
EARLY_STOP_MIN_EPOCHS = 8   # don't stop before epoch 8 — pretrained epoch-1 can be misleadingly high
T_MAX = EARLY_STOP_MIN_EPOCHS + 2 * EARLY_STOP_PATIENCE  # 14 — LR fully decays within actual window
IMAGE_SIZE = 224
LR = 3e-4
WEIGHT_DECAY = 1e-4


# ── Dataset utilities ────────────────────────────────────────────────────────


def _find_xray_dataset() -> Path:
    """Auto-detect chest_xray dataset root from env var or Kaggle input paths."""
    env = os.getenv("XRAY_DATASET_PATH")
    if env and Path(env).exists():
        return Path(env)

    candidates = [
        Path("/kaggle/input/chest-xray-pneumonia/chest_xray"),
        Path("/kaggle/input/chest-xray-images-pneumonia/chest_xray"),
        Path("/kaggle/input/chestxray2013/chest_xray"),
        Path("data/chest_xray"),
    ]
    for c in candidates:
        if (c / "train").exists():
            logger.info("Auto-detected X-ray dataset at: %s", c)
            return c

    # Last-resort: search /kaggle/input for a directory named 'PNEUMONIA'
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        hits = [h for h in kaggle_input.rglob("PNEUMONIA") if h.is_dir()]
        if hits:
            root = hits[0].parent.parent  # chest_xray/
            logger.info("Located PNEUMONIA folder — using root: %s", root)
            return root

    raise FileNotFoundError(
        "Chest X-Ray dataset not found. "
        "Set env var XRAY_DATASET_PATH or attach the 'chest-xray-pneumonia' dataset."
    )


def _build_transforms(is_train: bool) -> transforms.Compose:
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    # Grayscale first — normalise any scanner colour tinting before augmentation,
    # consistent with val/test. ColorJitter then operates purely on brightness/contrast.
    if is_train:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


class _SubsetWithTransform(torch.utils.data.Dataset):
    """Wraps a list of (path, label) pairs with a transform."""

    def __init__(self, samples: list[tuple[str, int]], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = PILImage.open(path).convert("RGB")
        return self.transform(img), label


def _build_loaders(dataset_root: Path, batch_size: int):
    """
    Build train/val/test DataLoaders for the chest X-ray dataset.

    Kaggle's provided val split has only 8 images per class — far too small.
    We pool train/ + val/ together (5216 images) and make an 80/20 stratified
    split ourselves. The test/ folder (624 images) is used as-is.
    """
    # Collect all train+val samples using ImageFolder (handles NORMAL/PNEUMONIA dirs)
    raw_trainval = datasets.ImageFolder(str(dataset_root / "train"))
    raw_val      = datasets.ImageFolder(str(dataset_root / "val"))
    raw_test     = datasets.ImageFolder(str(dataset_root / "test"))

    # Guard: all three splits must agree on class→index mapping before pooling.
    # If they differed, NORMAL/PNEUMONIA labels would be silently swapped on pooled subset.
    assert raw_trainval.class_to_idx == raw_val.class_to_idx == raw_test.class_to_idx, (
        f"class_to_idx mismatch across splits: "
        f"train={raw_trainval.class_to_idx} val={raw_val.class_to_idx} test={raw_test.class_to_idx}"
    )

    pooled = raw_trainval.samples + raw_val.samples   # list of (path, label)
    all_labels = [s[1] for s in pooled]

    test_samples = list(raw_test.samples)

    # class_to_idx: NORMAL=0, PNEUMONIA=1 (ImageFolder sorts alphabetically)
    class_names = raw_trainval.classes  # ['NORMAL', 'PNEUMONIA']
    logger.info(
        "Classes: %s  |  Train+val pool: %d  |  Test: %d",
        class_names, len(pooled), len(test_samples),
    )

    counts = np.bincount(all_labels)
    for name, cnt in zip(class_names, counts):
        logger.info("  %s: %d (%.1f%%)", name, cnt, 100 * cnt / len(pooled))

    # Stratified 80/20 split
    train_idx, val_idx = train_test_split(
        range(len(pooled)), test_size=0.20, random_state=42, stratify=all_labels
    )
    train_samples = [pooled[i] for i in train_idx]
    val_samples   = [pooled[i] for i in val_idx]

    logger.info("Train: %d  Val: %d  Test: %d", len(train_samples), len(val_samples), len(test_samples))

    # Weighted sampler to correct 74% PNEUMONIA imbalance in train
    train_labels = [s[1] for s in train_samples]
    class_weights = 1.0 / (np.bincount(train_labels).astype(float))
    sample_weights = [class_weights[l] for l in train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    on_gpu = torch.cuda.is_available()
    # persistent_workers intentionally omitted: workers are re-spawned per epoch so
    # each seed's set_seed() call seeds their RNG correctly via torch.initial_seed().
    kw = dict(num_workers=2, pin_memory=on_gpu)
    train_ds = _SubsetWithTransform(train_samples, _build_transforms(True))
    val_ds   = _SubsetWithTransform(val_samples,   _build_transforms(False))
    test_ds  = _SubsetWithTransform(test_samples,  _build_transforms(False))

    return (
        DataLoader(train_ds, batch_size=batch_size, sampler=sampler, **kw),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False,   **kw),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False,   **kw),
    )


# ── Utilities ────────────────────────────────────────────────────────────────


def _nan_safe(v: float) -> float | None:
    """Return None instead of NaN so json.dump produces valid JSON."""
    import math
    return None if (isinstance(v, float) and math.isnan(v)) else v


# ── Model ────────────────────────────────────────────────────────────────────


def _build_model(device: torch.device) -> nn.Module:
    from src.models.malit import MALITV2
    model = MALITV2(num_classes=2)
    return model.to(device)


# ── Per-seed training ─────────────────────────────────────────────────────────


def _train_seed(
    seed: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    out_dir: Path,
) -> dict:
    from src.utils.seed import set_seed
    from src.training.trainer import Trainer
    from src.interpretability.his_analysis import extract_his_stats

    set_seed(seed)
    model = _build_model(device)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        epochs=EPOCHS,
        T_max=T_MAX,
        eta_min=1e-6,
        amp=True,
        early_stopping_patience=EARLY_STOP_PATIENCE,
        early_stopping_min_epochs=EARLY_STOP_MIN_EPOCHS,
        grad_clip=1.0,
        device=device,
        saver=None,
        model_name=f"xray_seed{seed}",
        seed=seed,
        lambda_cal=0.1,
        cal_warmup_epochs=3,
        f1_average="macro",   # macro F1 for both early stopping and reporting
    )

    logger.info("══ Seed %d ══", seed)
    trainer.fit()

    # Evaluate on test set — test_metrics already contains f1 (macro), auroc, ece, brier
    test_metrics, preds, labels, _ = trainer.evaluate(test_loader)
    macro_f1 = test_metrics.get("f1", float("nan"))
    auroc    = test_metrics.get("auroc", float("nan"))

    # HIS stats
    his_stats = extract_his_stats(model)
    his_out = {
        "delta_1":          round(his_stats["delta_1"], 6),
        "delta_2":          round(his_stats["delta_2"], 6),
        "delta_3_mean":     round(his_stats["delta_3_mean"], 6),
        "delta_1_gt_delta_2": his_stats["delta_1"] > his_stats["delta_2"],
        "gap":              round(his_stats["delta_1"] - his_stats["delta_2"], 6),
    }

    logger.info(
        "HIS: delta_1=%.4f delta_2=%.4f delta_1>delta_2=%s gap=%.4f",
        his_out["delta_1"], his_out["delta_2"],
        his_out["delta_1_gt_delta_2"], his_out["gap"],
    )
    logger.info(
        "Seed %d | F1=%.4f (macro)  AUROC=%.4f",
        seed, macro_f1, auroc,
    )

    # Save weights
    seed_dir = out_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), seed_dir / "weights.pt")

    # Save per-seed metrics
    seed_result = {
        "seed":     seed,
        "f1":       _nan_safe(round(macro_f1, 6)),
        "auroc":    _nan_safe(round(auroc, 6)),
        "accuracy": _nan_safe(round(test_metrics.get("accuracy", float("nan")), 6)),
        "ece":      _nan_safe(round(test_metrics.get("ece", float("nan")), 6)),
        "brier":    _nan_safe(round(test_metrics.get("brier", float("nan")), 6)),
        "his":      his_out,
    }
    with open(seed_dir / "metrics.json", "w") as f:
        json.dump(seed_result, f, indent=2)

    del model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return seed_result


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    t0 = time.time()

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"━━━ MALIT-H Chest X-Ray Validation ━━━")
        print(f"GPU: {name} ({mem_gb:.1f} GB)")
    else:
        device = torch.device("cpu")
        print("━━━ MALIT-H Chest X-Ray Validation ━━━")
        print("GPU: None (CPU mode — will be slow)")

    # Paths
    dataset_root = _find_xray_dataset()
    output_dir = Path(os.getenv("OUTPUT_DIR",
                                "/kaggle/working/results" if Path("/kaggle").exists() else "results"))
    out_dir = output_dir / "xray"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {dataset_root}")

    # Auto batch size
    if torch.cuda.is_available():
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        batch_size = 64 if mem_gb >= 16 else 32 if mem_gb >= 8 else 16
    else:
        batch_size = 16

    # Data
    train_loader, val_loader, test_loader = _build_loaders(dataset_root, batch_size)

    # Run seeds
    all_results = []
    for seed in SEEDS:
        result = _train_seed(seed, train_loader, val_loader, test_loader, device, out_dir)
        all_results.append(result)

    # Aggregate
    f1_vals    = [r["f1"]    for r in all_results]
    auroc_vals = [r["auroc"] for r in all_results]
    d1_vals    = [r["his"]["delta_1"] for r in all_results]
    d2_vals    = [r["his"]["delta_2"] for r in all_results]

    mean_f1    = float(np.mean(f1_vals))
    std_f1     = float(np.std(f1_vals,    ddof=1) if len(f1_vals) >= 2 else 0.0)
    mean_auroc = float(np.mean(auroc_vals))
    std_auroc  = float(np.std(auroc_vals, ddof=1) if len(auroc_vals) >= 2 else 0.0)
    mean_d1    = float(np.mean(d1_vals))
    mean_d2    = float(np.mean(d2_vals))
    d1_gt_d2_all = all(r["his"]["delta_1_gt_delta_2"] for r in all_results)

    summary = {
        "mean_f1":    round(mean_f1,    6),
        "std_f1":     round(std_f1,     6),
        "mean_auroc": round(mean_auroc, 6),
        "std_auroc":  round(std_auroc,  6),
        "his_summary": {
            "delta_1_values":            [round(v, 6) for v in d1_vals],
            "delta_2_values":            [round(v, 6) for v in d2_vals],
            "mean_delta_1":              round(mean_d1, 6),
            "mean_delta_2":              round(mean_d2, 6),
            "delta_1_gt_delta_2_all_seeds": d1_gt_d2_all,
            "seeds_with_d1_gt_d2": sum(r["his"]["delta_1_gt_delta_2"] for r in all_results),
        },
        "per_seed": all_results,
        "seeds": SEEDS,
        "dataset": str(dataset_root),
    }

    with open(out_dir / "xray_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    elapsed = time.time() - t0

    print(f"\n━━━ X-RAY SUMMARY ━━━")
    print(f"F1:    {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"AUROC: {mean_auroc:.4f} ± {std_auroc:.4f}")
    print(f"DAIS:  mean_delta_1={mean_d1:.4f}  mean_delta_2={mean_d2:.4f}  "
          f"delta_1>delta_2 in all seeds: {d1_gt_d2_all}")
    print(f"Results saved to {out_dir / 'xray_summary.json'}")
    print(f"━━━ Done in {elapsed:.1f}s ━━━")


if __name__ == "__main__":
    main()
