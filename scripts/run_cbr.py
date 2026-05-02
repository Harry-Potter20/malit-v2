#!/usr/bin/env python3
"""Build FAISS CBR index and run case-based reasoning."""

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
from src.explainability.case_based_reasoning import CaseBasedReasoning, EmbeddingExtractor
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
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
    ).prepare()
    train_loader, _, test_loader = pipeline.get_loaders()

    # Load best model (seed 42)
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
    model_path = next(Path("results/models").glob("*seed42*.pt"), None)
    if model_path:
        ckpt = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state"])
    model.to(device)

    extractor = EmbeddingExtractor(model, device)

    logger.info("Extracting train embeddings…")
    train_ds = pipeline.datasets["train"]
    train_embs, train_labels = extractor.extract(train_loader)
    train_paths = [str(p) for p in train_ds.image_paths]

    logger.info("Extracting test embeddings…")
    test_embs, test_labels = extractor.extract(test_loader)

    cbr_dir = Path("results/cbr")
    cbr_dir.mkdir(parents=True, exist_ok=True)
    np.save(cbr_dir / "train_embeddings.npy", train_embs)
    np.save(cbr_dir / "test_embeddings.npy", test_embs)
    np.save(cbr_dir / "train_labels.npy", train_labels)
    np.save(cbr_dir / "test_labels.npy", test_labels)

    logger.info("Building CBR index…")
    cbr = CaseBasedReasoning(k=5)
    cbr.build_index(train_embs, train_labels, train_paths)

    try:
        import faiss
        faiss.write_index(cbr.index, str(cbr_dir / "index.faiss"))
        logger.info("FAISS index saved.")
    except Exception:
        logger.warning("Could not save FAISS index — using dict format instead.")

    logger.info("Running CBR retrieval on test set…")
    results = cbr.run_on_embeddings(test_embs, test_labels)

    with open(cbr_dir / "neighbors.json", "w") as f:
        json.dump(results["neighbors"][:200], f, indent=2)  # save first 200

    saver.save_metrics(
        {"mean_consistency": results["mean_consistency"], "std_consistency": results["std_consistency"]},
        "cbr_summary",
    )
    logger.info(
        "CBR mean consistency: %.4f ± %.4f",
        results["mean_consistency"], results["std_consistency"],
    )


if __name__ == "__main__":
    main()
