#!/usr/bin/env python3
"""
Unified MALIT V2 full pipeline — single entry point for local and Kaggle execution.

Usage:
    python scripts/run_full_pipeline.py              # full run, all 3 seeds
    python scripts/run_full_pipeline.py --quick      # 1 seed, 10% data, 1 epoch
    python scripts/run_full_pipeline.py --config configs/malit_v2.yaml

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
logger = logging.getLogger("pipeline")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MALIT V2 full pipeline")
    p.add_argument("--config", default="configs/malit_v2.yaml")
    p.add_argument("--quick", action="store_true",
                   help="Quick test mode: 1 seed, 10%% data, 1 epoch")
    return p.parse_args()


def validate_stage(stage: str, paths: list[Path]) -> None:
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise RuntimeError(
            f"Stage '{stage}' failed — expected outputs missing: "
            + ", ".join(str(p) for p in missing)
        )
    logger.info("[✓] Stage '%s' — all outputs present", stage)


def build_quick_loaders(dataset_path: str, image_size: int, batch_size: int):
    """Build tiny DataLoaders for --quick mode (10% of data, no dedup)."""
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader
    from src.data.dataset import MalariaDataset, build_transforms

    full_ds = MalariaDataset.from_directory(Path(dataset_path))
    n = max(40, len(full_ds) // 10)
    rng = np.random.RandomState(42)
    idx = rng.choice(len(full_ds), n, replace=False).tolist()
    labels = [full_ds.labels[i] for i in idx]
    paths = [full_ds.image_paths[i] for i in idx]
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


def main() -> None:
    load_dotenv()
    args = parse_args()
    t0 = time.time()

    from src.utils.gpu import get_device, get_dataset_path, get_output_dir, auto_batch_size, log_gpu_memory
    from src.utils.config import load_config
    from src.utils.save import ArtifactSaver

    cfg = load_config(args.config)
    device = get_device()
    dataset_path = get_dataset_path()
    output_dir = get_output_dir()
    cfg.ensure_dirs(output_dir)
    saver = ArtifactSaver(output_dir)
    results_base = Path(output_dir) / "results"

    if args.quick:
        logger.info("⚡ QUICK MODE: 1 seed, 1 epoch, 10%% data")
        cfg.training.seeds_tier1 = [42]
        cfg.training.active_tier = 3
        cfg.training.epochs = 1
        cfg.training.early_stopping.patience = 999

    batch_size = auto_batch_size(device, base=cfg.training.batch_size)
    logger.info("Device=%s  batch_size=%d", device, batch_size)

    # ── Stage 1: Data ─────────────────────────────────────────────────────────
    logger.info("━━━ [1/6] Data Pipeline ━━━")
    if args.quick:
        train_l, val_l, test_l = build_quick_loaders(dataset_path, cfg.model.image_size, batch_size)
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
        saver.save_reproducibility(pipeline.split_indices, "split_indices.json")

    # ── Stage 2: Training ─────────────────────────────────────────────────────
    logger.info("━━━ [2/6] Multi-Seed Training (seeds=%s) ━━━", cfg.training.active_seeds)
    from src.training.runner import MultiSeedRunner

    runner = MultiSeedRunner(
        cfg=cfg, train_loader=train_l, val_loader=val_l, test_loader=test_l,
        saver=saver, device=device,
    ).run()
    log_gpu_memory("post-train")

    validate_stage("training", [
        results_base / "statistics" / f"seed_{s}_metrics.json"
        for s in cfg.training.active_seeds
    ])

    # ── Stage 3: Uncertainty ──────────────────────────────────────────────────
    logger.info("━━━ [3/6] Uncertainty Estimation ━━━")
    from src.explainability.bayesian_uncertainty import BayesianUncertainty
    from src.explainability.ensemble_confidence import EnsembleConfidence

    first_model = next(iter(runner.seed_models.values()))
    T = 5 if args.quick else 30
    mc_results = BayesianUncertainty(T=T, device=device).run_on_loader(first_model, test_l)
    saver.save_metrics(BayesianUncertainty.uncertainty_summary(mc_results), "mc_dropout_summary")

    ens_results = EnsembleConfidence().run(
        runner.all_seed_probs, runner.all_seed_preds, np.array(runner.seed_labels)
    )
    saver.save_metrics(
        {k: v for k, v in ens_results.items() if not isinstance(v, np.ndarray)},
        "ensemble_summary",
    )
    log_gpu_memory("post-uncertainty")

    # ── Stage 4: CBR ──────────────────────────────────────────────────────────
    logger.info("━━━ [4/6] Case-Based Reasoning ━━━")
    from src.explainability.case_based_reasoning import CaseBasedReasoning, EmbeddingExtractor

    extractor = EmbeddingExtractor(first_model, device)
    train_embs, train_labels = extractor.extract(train_l)
    test_embs, test_labels = extractor.extract(test_l)

    cbr = CaseBasedReasoning(k=5)
    cbr.build_index(train_embs, train_labels)
    cbr_results = cbr.run_on_embeddings(test_embs, test_labels)
    saver.save_metrics(
        {"mean_consistency": cbr_results["mean_consistency"],
         "std_consistency": cbr_results["std_consistency"]},
        "cbr_summary",
    )

    # ── Stage 5: Statistics ───────────────────────────────────────────────────
    logger.info("━━━ [5/6] Statistical Testing ━━━")
    from src.statistics.tests import StatisticsReport

    if len(runner.seed_preds) >= 2:
        report = StatisticsReport(
            seed_preds=runner.seed_preds,
            seed_metrics=runner.seed_metrics,
            labels=runner.seed_labels,
            margin=cfg.statistics.tost_margin,
            alpha=cfg.statistics.tost_alpha,
        ).full_report()
        saver.save_metrics(report, "full_statistics_report")
        logger.info("F1: %.4f ± %.4f  TOST-equiv: %s",
                    report["summary"]["mean_f1"], report["summary"]["std_f1"],
                    report["tost"]["equivalent"])

    # ── Stage 6: Reproducibility ──────────────────────────────────────────────
    logger.info("━━━ [6/6] Reproducibility Artifacts ━━━")
    saver.save_reproducibility(
        {"seeds": cfg.training.active_seeds, "split_seed": cfg.data.seed}, "seed_manifest.json"
    )
    saver.save_reproducibility(
        {"lr": cfg.training.optimizer.lr, "epochs": cfg.training.epochs,
         "batch_size": batch_size, "amp": cfg.training.amp}, "hyperparameters.json"
    )
    saver.save_reproducibility(
        {"models": [f"malit_v2_seed{s}" for s in cfg.training.active_seeds]}, "model_registry.json"
    )

    validate_stage("reproducibility", [
        results_base / "reproducibility" / f
        for f in ["seed_manifest.json", "hyperparameters.json", "model_registry.json"]
    ])

    logger.info("━━━ Pipeline complete in %.1fs ━━━", time.time() - t0)


if __name__ == "__main__":
    main()
