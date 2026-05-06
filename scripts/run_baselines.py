#!/usr/bin/env python3
"""
Baseline training and comparison pipeline for MALIT V2.

Usage:
    python scripts/run_baselines.py                      # all 5 baselines, 2 seeds
    python scripts/run_baselines.py --quick              # 1 seed, 1 epoch, 10% data
    python scripts/run_baselines.py --models efficientnet,resnet50
    python scripts/run_baselines.py --grad-accum-steps 4  # for 4 GB GPU

Environment variables:
    DATASET_PATH   — path to cell_images root (default: data/cell_images)
    OUTPUT_DIR     — base results directory (default: .)
"""
from __future__ import annotations

import argparse
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
logger = logging.getLogger("baselines")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MALIT V2 baseline comparison pipeline")
    p.add_argument("--config", default="configs/malit_v2.yaml")
    p.add_argument("--quick", action="store_true",
                   help="Quick mode: 1 seed, 1 epoch, 10%% data")
    p.add_argument("--models", default="efficientnet,resnet50,mobilenetv3,senet,cbam",
                   help="Comma-separated baseline names")
    p.add_argument("--grad-accum-steps", type=int, default=1,
                   help="Gradient accumulation steps (use 4+ for <8 GB VRAM)")
    p.add_argument("--gradient-checkpointing", action="store_true",
                   help="Enable gradient checkpointing (saves activation memory)")
    p.add_argument("--pretrained", action="store_true", default=True,
                   help="Use ImageNet-pretrained weights (default: True)")
    p.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    return p.parse_args()


def build_quick_loaders(dataset_path: str, image_size: int, batch_size: int):
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader
    from src.data.dataset import MalariaDataset, build_transforms

    full_ds = MalariaDataset.from_directory(Path(dataset_path))
    n = max(40, len(full_ds) // 10)
    rng = np.random.RandomState(42)
    idx = rng.choice(len(full_ds), n, replace=False).tolist()
    labels = [full_ds.labels[i] for i in idx]
    paths  = [full_ds.image_paths[i] for i in idx]
    small_ds = MalariaDataset(paths, labels)

    all_idx = list(range(len(small_ds)))
    train_idx, rest_idx = train_test_split(
        all_idx, test_size=0.30, random_state=42, stratify=labels
    )
    val_idx, test_idx = train_test_split(
        rest_idx, test_size=0.50, random_state=42,
        stratify=[labels[i] for i in rest_idx],
    )
    kw = dict(batch_size=batch_size, num_workers=0)
    return (
        DataLoader(small_ds.subset(train_idx, build_transforms(image_size, True)),  shuffle=True,  **kw),
        DataLoader(small_ds.subset(val_idx,   build_transforms(image_size, False)), shuffle=False, **kw),
        DataLoader(small_ds.subset(test_idx,  build_transforms(image_size, False)), shuffle=False, **kw),
    )


def load_malit_results(results_base: Path) -> dict | None:
    """Auto-discover and load all available MALIT per-seed metrics."""
    stats_dir = results_base / "statistics"
    if not stats_dir.exists():
        logger.warning("MALIT statistics dir not found: %s", stats_dir)
        return None

    seed_files = sorted(stats_dir.glob("seed_*_metrics.json"))
    if not seed_files:
        logger.warning("No seed_*_metrics.json found in %s", stats_dir)
        return None

    seed_metrics: dict = {}
    seed_preds: dict = {}

    for f in seed_files:
        try:
            seed = int(f.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        import json
        data = json.loads(f.read_text())
        if not data.get("predictions"):
            logger.warning(
                "No predictions in %s — McNemar will be skipped. "
                "Re-run run_full_pipeline.py to regenerate.", f.name
            )
            return None
        seed_metrics[seed] = data.get("metrics", {})
        seed_preds[seed] = data["predictions"]

    if not seed_preds:
        return None

    logger.info("Loaded MALIT results for seeds: %s", sorted(seed_preds.keys()))
    return {"seed_metrics": seed_metrics, "seed_preds": seed_preds}


def main() -> None:
    load_dotenv()
    args = parse_args()
    t0 = time.time()

    from src.utils.gpu import get_device, get_dataset_path, get_output_dir, auto_batch_size
    from src.utils.config import load_config
    from src.utils.save import ArtifactSaver
    from src.baselines.runner import BaselineRunner
    from src.comparison.protocol import ComparisonProtocol
    from src.comparison.reporting import (
        build_performance_table, build_statistical_table,
        save_paper_tables, plot_roc_curves, plot_calibration,
        generate_comparison_text,
    )

    cfg = load_config(args.config)
    device = get_device()
    dataset_path = get_dataset_path()
    output_dir = get_output_dir()
    cfg.ensure_dirs(output_dir)
    saver = ArtifactSaver(output_dir)
    results_base = Path(output_dir) / "results"

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]

    if args.quick:
        logger.info("⚡ QUICK MODE: 1 seed, 1 epoch, 10%% data")
        seeds = [42]
        epochs = 1
        patience = 999
    else:
        seeds = [42, 123]
        epochs = cfg.training.epochs
        patience = cfg.training.early_stopping.patience

    batch_size = auto_batch_size(device, base=cfg.training.batch_size)
    logger.info("Device=%s  batch_size=%d  grad_accum=%d  grad_ckpt=%s",
                device, batch_size, args.grad_accum_steps, args.gradient_checkpointing)

    # ── Stage 1: Data (SAME split as MALIT — split_seed=42 is critical) ───────
    logger.info("━━━ [1/4] Data Pipeline (split_seed=42) ━━━")
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
            split_seed=cfg.data.seed,       # MUST equal 42
            hash_size=cfg.data.deduplication.hash_size,
            hamming_threshold=cfg.data.deduplication.hamming_threshold,
            batch_size=batch_size,
            num_workers=cfg.training.num_workers,
        ).prepare()
        train_l, val_l, test_l = pipeline.get_loaders(device=device)
    n_test = len(test_l.dataset)
    logger.info("Test set size: %d samples", n_test)

    # ── Stage 2: Train baselines ───────────────────────────────────────────────
    logger.info("━━━ [2/4] Baseline Training (%s) ━━━", model_names)
    runner = BaselineRunner(
        train_loader=train_l,
        val_loader=val_l,
        test_loader=test_l,
        saver=saver,
        device=device,
        lr=cfg.training.optimizer.lr,
        weight_decay=cfg.training.optimizer.weight_decay,
        epochs=epochs,
        amp=cfg.training.amp,
        patience=patience,
        grad_clip=cfg.training.grad_clip,
        grad_accum_steps=args.grad_accum_steps,
        use_gradient_checkpointing=args.gradient_checkpointing,
        seeds=seeds,
        pretrained=args.pretrained,
    )
    runner.run(model_names)

    # ── Stage 3: Statistical comparison ───────────────────────────────────────
    logger.info("━━━ [3/4] Statistical Comparison ━━━")
    malit_results = load_malit_results(results_base)
    if malit_results is None:
        logger.warning(
            "MALIT results not found in %s — skipping comparison. "
            "Run scripts/run_full_pipeline.py first.",
            results_base / "statistics",
        )
        comparison_results = {}
    else:
        first_model = model_names[0]
        labels = runner.results[first_model]["seed_labels"] or []
        protocol = ComparisonProtocol(
            malit_results=malit_results,
            baseline_results=runner.results,
            labels=labels,
        )
        comparison_results = protocol.compare_all()
        saver.save_metrics(comparison_results, "comparison_results")
        text = generate_comparison_text(comparison_results)
        saver.save_reproducibility(text, "comparison_statements.json")
        logger.info(text["comparison_statement"])

    # ── Stage 4: Generate tables and figures ───────────────────────────────────
    logger.info("━━━ [4/4] Reporting ━━━")
    tables_dir = results_base / "tables"

    all_results_for_table = dict(runner.results)
    if malit_results:
        all_results_for_table["malit_v2"] = malit_results

    perf_table = build_performance_table(all_results_for_table)
    tables: dict = {"performance_table": perf_table}

    if comparison_results:
        stat_table = build_statistical_table(comparison_results)
        tables["statistical_table"] = stat_table

    saved = save_paper_tables(tables, tables_dir)
    for name, path in saved.items():
        logger.info("Saved table: %s", path)

    figs_dir = results_base / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)
    plot_roc_curves(runner.results, figs_dir / "roc_curves.png")
    plot_calibration(runner.results, figs_dir / "calibration.png")

    logger.info("━━━ Baselines complete in %.1fs ━━━", time.time() - t0)


if __name__ == "__main__":
    main()
