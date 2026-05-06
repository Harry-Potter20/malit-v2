#!/usr/bin/env python3
"""
Ablation study for MALIT V2.

Trains 4 component-ablated variants (no_gabor, no_lcci, no_attention,
no_multiscale) and compares each against the full model using McNemar +
TOST.  Requires run_full_pipeline.py to have run first (loads MALIT
predictions from results/statistics/seed_*_metrics.json).

Usage:
    python scripts/run_ablation.py
    python scripts/run_ablation.py --quick     # 1 epoch, 10% data
    python scripts/run_ablation.py --config configs/malit_v2.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ablation")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MALIT V2 ablation study")
    p.add_argument("--config", default="configs/malit_v2.yaml")
    p.add_argument("--quick", action="store_true",
                   help="Quick mode: 1 epoch, 10%% data, 1 seed")
    return p.parse_args()


def load_full_malit_results(results_base: Path) -> dict | None:
    """Load full MALIT V2 predictions from disk."""
    stats_dir = results_base / "statistics"
    seed_files = sorted(stats_dir.glob("seed_*_metrics.json")) if stats_dir.exists() else []
    if not seed_files:
        logger.error("No MALIT results found in %s — run run_full_pipeline.py first.", stats_dir)
        return None

    seed_preds: dict = {}
    seed_metrics: dict = {}
    for f in seed_files:
        try:
            seed = int(f.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        data = json.loads(f.read_text())
        if not data.get("predictions"):
            logger.error("No predictions in %s — re-run run_full_pipeline.py.", f.name)
            return None
        seed_preds[seed] = data["predictions"]
        seed_metrics[seed] = data.get("metrics", {})

    logger.info("Full MALIT results loaded for seeds: %s", sorted(seed_preds.keys()))
    return {"seed_preds": seed_preds, "seed_metrics": seed_metrics}


def build_quick_loaders(dataset_path, image_size, batch_size):
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader
    from src.data.dataset import MalariaDataset, build_transforms

    full_ds = MalariaDataset.from_directory(Path(dataset_path))
    n = max(40, len(full_ds) // 10)
    rng = np.random.RandomState(42)
    idx = rng.choice(len(full_ds), n, replace=False).tolist()
    labels = [full_ds.labels[i] for i in idx]
    paths  = [full_ds.image_paths[i] for i in idx]
    small  = MalariaDataset(paths, labels)
    all_idx = list(range(len(small)))
    tr, rest = train_test_split(all_idx, test_size=0.30, random_state=42, stratify=labels)
    va, te = train_test_split(rest, test_size=0.50, random_state=42,
                               stratify=[labels[i] for i in rest])
    kw = dict(batch_size=batch_size, num_workers=0)
    return (
        DataLoader(small.subset(tr, build_transforms(image_size, True)),  shuffle=True,  **kw),
        DataLoader(small.subset(va, build_transforms(image_size, False)), shuffle=False, **kw),
        DataLoader(small.subset(te, build_transforms(image_size, False)), shuffle=False, **kw),
    )


def main() -> None:
    load_dotenv()
    args = parse_args()
    t0 = time.time()

    from src.utils.gpu import get_device, get_dataset_path, get_output_dir, auto_batch_size
    from src.utils.config import load_config
    from src.utils.save import ArtifactSaver, _make_serializable
    from src.ablation.runner import AblationRunner

    cfg = load_config(args.config)
    device = get_device()
    dataset_path = get_dataset_path()
    output_dir = get_output_dir()
    cfg.ensure_dirs(output_dir)
    saver = ArtifactSaver(output_dir)
    results_base = Path(output_dir) / "results"

    if args.quick:
        cfg.training.epochs = 1
        cfg.training.early_stopping.patience = 999
        seeds = [42]
    else:
        seeds = cfg.training.seeds_tier2  # [42, 123]

    batch_size = auto_batch_size(device, base=cfg.training.batch_size)
    logger.info("Device=%s  batch_size=%d  seeds=%s", device, batch_size, seeds)

    # ── Data (same split as MALIT) ─────────────────────────────────────────────
    logger.info("━━━ [1/3] Data Pipeline ━━━")
    if args.quick:
        train_l, val_l, test_l = build_quick_loaders(
            dataset_path, cfg.model.image_size, batch_size
        )
    else:
        from src.data.pipeline import DataPipeline
        pipeline = DataPipeline(
            root=dataset_path,
            image_size=cfg.model.image_size,
            train_frac=cfg.data.splits.train,
            val_frac=cfg.data.splits.val,
            split_seed=cfg.data.seed,
            hash_size=cfg.data.deduplication.hash_size,
            hamming_threshold=cfg.data.deduplication.hamming_threshold,
            batch_size=batch_size,
            num_workers=cfg.training.num_workers,
        ).prepare()
        train_l, val_l, test_l = pipeline.get_loaders()

    # ── Load full MALIT results ────────────────────────────────────────────────
    full_results = load_full_malit_results(results_base)
    if full_results is None:
        logger.error("Cannot run ablation without full MALIT results. Aborting.")
        sys.exit(1)

    full_f1_vals = [m.get("f1", float("nan"))
                    for m in full_results["seed_metrics"].values()]
    full_mean_f1 = float(np.nanmean(full_f1_vals))
    logger.info("Full MALIT V2  F1=%.4f (mean over %d seeds)",
                full_mean_f1, len(full_f1_vals))

    # ── Ablation training ──────────────────────────────────────────────────────
    logger.info("━━━ [2/3] Ablation Training (4 variants × %d seeds) ━━━", len(seeds))
    ablation = AblationRunner(
        cfg=cfg,
        train_loader=train_l,
        val_loader=val_l,
        test_loader=test_l,
        saver=saver,
        device=device,
        seeds=seeds,
    )
    ablation.run()

    # ── Comparison + reporting ─────────────────────────────────────────────────
    logger.info("━━━ [3/3] Comparison & Reporting ━━━")
    comparison = ablation.compare_against_full(
        full_results["seed_preds"], full_results["seed_metrics"]
    )

    ablation_dir = results_base / "ablation"
    ablation.save_ablation_table(comparison, full_mean_f1, ablation_dir)

    # Save full comparison JSON
    out_json = ablation_dir / "ablation_comparison.json"
    with open(out_json, "w") as f:
        json.dump(_make_serializable(comparison), f, indent=2)
    logger.info("Full ablation results → %s", out_json)

    # Summary
    logger.info("━━━ Ablation Summary ━━━")
    logger.info("  %-20s  %6s  %8s  %s", "variant", "F1", "ΔF1", "verdict")
    logger.info("  %-20s  %6.4f  %8s  %s", "full (MALIT V2)", full_mean_f1, "—", "reference")
    for name, res in comparison.items():
        if "error" not in res:
            logger.info("  %-20s  %6.4f  %+8.4f  %s",
                        name, res["variant_mean_f1"], -res["delta_f1"], res["verdict"])

    logger.info("━━━ Ablation complete in %.1fs ━━━", time.time() - t0)


if __name__ == "__main__":
    main()
