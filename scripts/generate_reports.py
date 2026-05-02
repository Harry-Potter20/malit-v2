#!/usr/bin/env python3
"""Generate clinical reports, explainability plots, and reproducibility artifacts."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from src.data.pipeline import DataPipeline
from src.explainability.clinical_confidence import ClinicalConfidenceEngine
from src.interpretability.attention_viz import AttentionVisualizer
from src.interpretability.gabor_viz import GaborVisualizer
from src.interpretability.gradcam import GradCAMVisualizer
from src.interpretability.lcci_viz import LCCIVisualizer
from src.models.malit import MALITV2
from src.utils.config import load_config
from src.utils.save import ArtifactSaver

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    load_dotenv()
    cfg = load_config("configs/malit_v2.yaml")
    cfg.ensure_dirs()
    saver = ArtifactSaver(".")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pipeline = DataPipeline(
        root="data/cell_images",
        image_size=cfg.model.image_size,
        batch_size=min(8, cfg.training.batch_size),
        num_workers=2,
    ).prepare()
    _, val_loader, test_loader = pipeline.get_loaders()

    mc = cfg.model
    model = MALITV2(
        num_classes=mc.num_classes,
        gabor_n_orientations=mc.gabor.n_orientations,
        gabor_n_scales=mc.gabor.n_scales,
        gabor_kernel_size=mc.gabor.kernel_size,
        efficientnet_backbone=mc.efficientnet.backbone,
        efficientnet_pretrained=False,
        efficientnet_freeze_blocks=0,
    )
    model_path = next(Path("results/models").glob("*seed42*.pt"), None)
    if model_path:
        ckpt = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state"])
    model.to(device)

    # ── Temperature calibration ─────────────────────────────────────────────
    logger.info("Fitting temperature calibration on validation set…")
    engine = ClinicalConfidenceEngine(num_channels=1280)
    val_logits, val_labels = [], []
    model.eval()
    with torch.no_grad():
        for imgs, labels in val_loader:
            logits, _ = model(imgs.to(device))
            val_logits.append(logits.cpu().numpy())
            val_labels.extend(labels.tolist())
    import numpy as np
    val_logits_arr = np.concatenate(val_logits, axis=0)
    temp = engine.fit_calibration(val_logits_arr, np.array(val_labels))
    saver.save_reproducibility({"temperature": temp}, "calibration.json")

    # ── GradCAM ─────────────────────────────────────────────────────────────
    logger.info("Generating GradCAM…")
    gcam = GradCAMVisualizer(model)
    sample_batch, _ = next(iter(test_loader))
    gcam.save_batch(sample_batch[:8].to(device), "results/gradcam", "gradcam")

    # ── Attention maps ──────────────────────────────────────────────────────
    logger.info("Generating attention maps…")
    att_viz = AttentionVisualizer(model, device)
    att_viz.plot_spatial_attention(sample_batch[:8], "results/attention_maps")
    att_data = att_viz.collect_attention(test_loader, max_batches=20)
    if att_data["attention_entropy"]:
        att_viz.plot_entropy_distribution(att_data["attention_entropy"], "results/plots")

    # ── LCCI channel suppression ─────────────────────────────────────────────
    logger.info("Generating LCCI visualization…")
    lcci_viz = LCCIVisualizer(model, device)
    energies = lcci_viz.collect_channel_energies(test_loader, max_batches=20)
    lcci_viz.plot_channel_suppression(energies, "results/plots")
    dr = lcci_viz.dominance_ratio(energies["post_lcci"])
    logger.info("Dominance ratio (post-LCCI): %.4f", dr)

    # ── Gabor filter bank ──────────────────────────────────────────────────
    logger.info("Generating Gabor filter visualizations…")
    gbr_viz = GaborVisualizer(model.gabor)
    gbr_viz.plot_filter_bank("results/plots")
    gbr_viz.plot_parameter_distribution("results/plots")
    gbr_viz.plot_decomposition(sample_batch[0], "results/plots")

    # ── Reproducibility artifacts ────────────────────────────────────────────
    logger.info("Saving reproducibility artifacts…")
    saver.save_reproducibility(
        {
            "seeds": cfg.training.active_seeds,
            "split_seed": cfg.data.seed,
            "hash_size": cfg.data.deduplication.hash_size,
            "hamming_threshold": cfg.data.deduplication.hamming_threshold,
        },
        "seed_manifest.json",
    )
    saver.save_reproducibility(
        {
            "lr": cfg.training.optimizer.lr,
            "weight_decay": cfg.training.optimizer.weight_decay,
            "epochs": cfg.training.epochs,
            "batch_size": cfg.training.batch_size,
            "amp": cfg.training.amp,
            "grad_clip": cfg.training.grad_clip,
            "early_stopping_patience": cfg.training.early_stopping.patience,
            "scheduler": "cosine_annealing",
            "T_max": cfg.training.scheduler.T_max,
        },
        "hyperparameters.json",
    )
    saver.save_reproducibility(
        {"models": [f"malit_v2_seed{s}" for s in cfg.training.active_seeds]},
        "model_registry.json",
    )
    logger.info("All reports and artifacts generated.")


if __name__ == "__main__":
    main()
