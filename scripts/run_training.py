#!/usr/bin/env python3
"""Run multi-seed MALIT V2 training experiments."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from src.data.pipeline import DataPipeline
from src.training.runner import MultiSeedRunner
from src.utils.config import load_config
from src.utils.save import ArtifactSaver

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    load_dotenv()
    cfg = load_config("configs/malit_v2.yaml")
    cfg.ensure_dirs()

    saver = ArtifactSaver(".")

    logger.info("Preparing data pipeline…")
    pipeline = DataPipeline(
        root="data/cell_images",
        image_size=cfg.model.image_size,
        train_frac=cfg.data.splits.train,
        val_frac=cfg.data.splits.val,
        split_seed=cfg.data.seed,
        hash_size=cfg.data.deduplication.hash_size,
        hamming_threshold=cfg.data.deduplication.hamming_threshold,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
    ).prepare()

    train_loader, val_loader, test_loader = pipeline.get_loaders()

    # Save split indices for reproducibility
    saver.save_reproducibility(pipeline.split_indices, "split_indices.json")

    logger.info("Starting MultiSeedRunner  (seeds=%s)…", cfg.training.active_seeds)
    runner = MultiSeedRunner(
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        saver=saver,
    ).run()

    logger.info("Training complete. Results in results/")


if __name__ == "__main__":
    main()
