#!/usr/bin/env python3
"""Run MC Dropout and Ensemble confidence estimation."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from src.data.pipeline import DataPipeline
from src.explainability.bayesian_uncertainty import BayesianUncertainty
from src.explainability.ensemble_confidence import EnsembleConfidence
from src.models.malit import MALITV2
from src.utils.config import load_config
from src.utils.save import ArtifactSaver

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_seed_models(cfg, saver: ArtifactSaver, device: torch.device) -> dict[int, MALITV2]:
    models = {}
    for seed in cfg.training.active_seeds:
        model_path = next(Path("results/models").glob(f"*seed{seed}*.pt"), None)
        if model_path is None:
            logger.warning("No checkpoint found for seed %d — skipping.", seed)
            continue
        mc = cfg.model
        model = MALITV2(
            num_classes=mc.num_classes,
            gabor_n_orientations=mc.gabor.n_orientations,
            gabor_n_scales=mc.gabor.n_scales,
            gabor_kernel_size=mc.gabor.kernel_size,
            efficientnet_backbone=mc.efficientnet.backbone,
            efficientnet_pretrained=False,
            efficientnet_freeze_blocks=0,
            lcci_reduction=mc.lcci_reduction,
        )
        ckpt = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        model.to(device)
        models[seed] = model
        logger.info("Loaded model for seed %d from %s", seed, model_path)
    return models


def main() -> None:
    load_dotenv()
    cfg = load_config("configs/malit_v2.yaml")
    cfg.ensure_dirs()
    saver = ArtifactSaver(".")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pipeline = DataPipeline(
        root="data/cell_images",
        image_size=cfg.model.image_size,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
    ).prepare()
    _, _, test_loader = pipeline.get_loaders()

    seed_models = load_seed_models(cfg, saver, device)
    if not seed_models:
        logger.error("No trained models found. Run run_training.py first.")
        raise SystemExit(1)

    # ── MC Dropout ─────────────────────────────────────────────────────────
    logger.info("Running MC Dropout (T=30)…")
    first_model = next(iter(seed_models.values()))
    mc = BayesianUncertainty(T=30, device=device)
    mc_results = mc.run_on_loader(first_model, test_loader)
    summary = BayesianUncertainty.uncertainty_summary(mc_results)

    mc_dir = Path("results/uncertainty/mc_dropout")
    mc_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(mc_results["mean_probs"]).to_csv(mc_dir / "mean_probs.csv", index=False)
    pd.DataFrame(mc_results["variance"]).to_csv(mc_dir / "variance.csv", index=False)
    pd.Series(mc_results["predictive_entropy"]).to_csv(mc_dir / "predictive_entropy.csv", index=False)
    saver.save_metrics(summary, "mc_dropout_summary")
    logger.info("MC Dropout summary: %s", summary)

    # ── Ensemble Confidence ─────────────────────────────────────────────────
    logger.info("Computing ensemble confidence…")
    seed_probs = {}
    seed_preds = {}
    labels_list = []
    for seed, model in seed_models.items():
        model.eval()
        probs_list, preds_list, labels_list = [], [], []
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs = imgs.to(device)
                logits, _ = model(imgs)
                p = torch.softmax(logits, dim=1).cpu().numpy()
                probs_list.append(p)
                preds_list.extend(logits.argmax(1).cpu().tolist())
                labels_list.extend(labels.tolist())
        seed_probs[seed] = np.concatenate(probs_list, axis=0)
        seed_preds[seed] = np.array(preds_list)

    all_probs = np.stack(list(seed_probs.values()), axis=0)
    all_preds = np.stack(list(seed_preds.values()), axis=0)
    labels_arr = np.array(labels_list)

    ens = EnsembleConfidence()
    ens_results = ens.run(all_probs, all_preds, labels_arr)

    ens_dir = Path("results/uncertainty/ensemble")
    ens_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ens_results["mean_probs"]).to_csv(ens_dir / "mean_probs.csv", index=False)
    ci_df = pd.DataFrame({
        "ci_low_pos": ens_results["ci_low"][:, 1],
        "ci_high_pos": ens_results["ci_high"][:, 1],
        "ci_width_pos": ens_results["ci_width"][:, 1],
    })
    ci_df.to_csv(ens_dir / "ci_bounds.csv", index=False)
    pd.Series(ens_results["agreement"]).to_csv(ens_dir / "agreement.csv", index=False)

    summary_ens = {k: v for k, v in ens_results.items() if not isinstance(v, np.ndarray)}
    saver.save_metrics(summary_ens, "ensemble_summary")
    logger.info("Ensemble mean CI width: %.4f  mean agreement: %.4f",
                ens_results["mean_ci_width"], ens_results["mean_agreement"])
    logger.info("Uncertainty estimation complete.")


if __name__ == "__main__":
    main()
