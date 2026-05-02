#!/usr/bin/env python3
"""Run McNemar and TOST statistical tests on multi-seed results."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from src.statistics.tests import StatisticsReport
from src.utils.config import load_config
from src.utils.save import ArtifactSaver

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    load_dotenv()
    cfg = load_config("configs/malit_v2.yaml")
    cfg.ensure_dirs()
    saver = ArtifactSaver(".")

    stats_dir = Path("results/statistics")
    seed_preds, seed_metrics, labels = {}, {}, None

    for seed in cfg.training.active_seeds:
        metrics_path = stats_dir / f"seed_{seed}_metrics.json"
        if not metrics_path.exists():
            logger.warning("No metrics file for seed %d — skipping.", seed)
            continue
        with open(metrics_path) as f:
            data = json.load(f)
        seed_metrics[seed] = data.get("metrics", {})

        # Predictions from stored logits (if available)
        from pathlib import Path as _P
        preds_file = _P(f"results/statistics/seed_{seed}_preds.json")
        if preds_file.exists():
            with open(preds_file) as f:
                pd_data = json.load(f)
            seed_preds[seed] = pd_data["preds"]
            labels = pd_data["labels"]

    if len(seed_metrics) < 2:
        logger.warning(
            "Need ≥2 seeds for statistical tests. Found: %d", len(seed_metrics)
        )
        if len(seed_metrics) == 0:
            raise SystemExit(1)

    if seed_preds and labels:
        report = StatisticsReport(
            seed_preds=seed_preds,
            seed_metrics=seed_metrics,
            labels=labels,
            margin=cfg.statistics.tost_margin,
            alpha=cfg.statistics.tost_alpha,
        ).full_report()
        saver.save_metrics(report, "full_statistics_report")

        import pandas as pd
        if report["mcnemar"]:
            df = pd.DataFrame(report["mcnemar"])
            saver.save_table(df, "mcnemar_table")

        logger.info("Statistics report saved.")
        logger.info("Mean F1: %.4f ± %.4f", report["summary"]["mean_f1"], report["summary"]["std_f1"])
        logger.info("TOST equivalent: %s", report["tost"]["equivalent"])
    else:
        logger.info("No predictions found. Saving metric summary only.")
        saver.save_metrics({"per_seed": seed_metrics}, "metrics_summary")


if __name__ == "__main__":
    main()
