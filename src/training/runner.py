from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.malit import MALITV2
from src.utils.config import Config
from src.utils.prediction_logging import PredictionLogger
from src.utils.save import ArtifactSaver
from src.utils.seed import set_seed
from .trainer import Trainer

logger = logging.getLogger(__name__)


class MultiSeedRunner:
    """
    Runs training and evaluation across multiple seeds.

    Stores per-seed predictions, probabilities, logits, and metrics.
    All seeds use identical data splits (split_seed=42) to ensure
    statistical comparisons are valid.
    """

    def __init__(
        self,
        cfg: Config,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        saver: ArtifactSaver,
        device: str | torch.device | None = None,
    ):
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.saver = saver
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Results containers — populated by run()
        self.seed_preds: dict[int, list[int]] = {}
        self.seed_probs: dict[int, np.ndarray] = {}
        self.seed_logits: dict[int, np.ndarray] = {}
        self.seed_labels: list[int] | None = None
        self.seed_metrics: dict[int, dict[str, Any]] = {}
        self.seed_models: dict[int, MALITV2] = {}

    def _build_model(self) -> MALITV2:
        mc = self.cfg.model
        return MALITV2(
            num_classes=mc.num_classes,
            gabor_n_orientations=mc.gabor.n_orientations,
            gabor_n_scales=mc.gabor.n_scales,
            gabor_kernel_size=mc.gabor.kernel_size,
            efficientnet_backbone=mc.efficientnet.backbone,
            efficientnet_pretrained=mc.efficientnet.pretrained,
            efficientnet_freeze_blocks=mc.efficientnet.freeze_blocks,
            lcci_reduction=mc.lcci_reduction,
            attention_reduction=mc.attention_channel_reduction,
            ms_rfs=mc.multiscale_rfs,
        )

    def run(self) -> "MultiSeedRunner":
        seeds = self.cfg.training.active_seeds
        logger.info("MultiSeedRunner: %d seeds %s", len(seeds), seeds)

        for seed in seeds:
            logger.info("── Seed %d ──────────────────────────────────────", seed)
            set_seed(seed)

            model = self._build_model()
            tc = self.cfg.training

            trainer = Trainer(
                model=model,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                lr=tc.optimizer.lr,
                weight_decay=tc.optimizer.weight_decay,
                epochs=tc.epochs,
                T_max=tc.scheduler.T_max,
                eta_min=tc.scheduler.eta_min,
                amp=tc.amp,
                early_stopping_patience=tc.early_stopping.patience,
                grad_clip=tc.grad_clip,
                device=self.device,
                saver=self.saver,
                model_name="malit_v2",
                seed=seed,
                lambda_cal=tc.lambda_cal,
                cal_warmup_epochs=tc.cal_warmup_epochs,
            )

            trainer.fit()
            metrics, preds, labels, history = trainer.evaluate(self.test_loader)

            # Collect logits and probs from a second pass (model is restored to best)
            probs_list, logits_list = self._collect_outputs(model)

            self.seed_preds[seed] = preds
            self.seed_probs[seed] = np.array(probs_list)
            self.seed_logits[seed] = np.array(logits_list)
            self.seed_labels = labels
            self.seed_metrics[seed] = metrics
            self.seed_models[seed] = model

            self.saver.save_metrics(
                {"seed": seed, "metrics": metrics, "history": history,
                 "predictions": preds, "labels": labels},
                f"seed_{seed}_metrics",
            )

            self._save_sample_predictions(seed, probs_list, preds, labels)
            logger.info("Seed %d  F1=%.4f  Acc=%.4f", seed, metrics.get("f1", 0), metrics.get("accuracy", 0))

        self._save_aggregate()
        return self

    @torch.no_grad()
    def _collect_outputs(self, model: MALITV2) -> tuple[list, list]:
        model.eval()
        model.to(self.device)
        probs_list, logits_list = [], []
        for imgs, _ in self.test_loader:
            imgs = imgs.to(self.device)
            logits, _ = model(imgs)
            probs = torch.softmax(logits, dim=1)
            probs_list.extend(probs.cpu().tolist())
            logits_list.extend(logits.cpu().tolist())
        return probs_list, logits_list

    def _save_sample_predictions(
        self, seed: int, probs_list: list, preds: list, labels: list
    ) -> None:
        try:
            probs = np.array(probs_list)
            confidence = probs.max(axis=1)
            entropy = -(probs * np.log(probs + 1e-8)).sum(axis=1)
            log_obj = PredictionLogger()
            for i, (pred, label) in enumerate(zip(preds, labels)):
                log_obj.add(
                    image_id=i,
                    true_label=int(label),
                    predicted_label=int(pred),
                    confidence=float(confidence[i]),
                    entropy=float(entropy[i]),
                    correct=int(pred == label),
                )
            out = self.saver.base / "results" / "statistics"
            log_obj.save(out / f"sample_predictions_seed{seed}.csv")
        except Exception as exc:
            logger.warning("Per-sample prediction logging failed (non-fatal): %s", exc)

    def _save_aggregate(self) -> None:
        agg: dict[str, Any] = {}
        for metric_name in ["f1", "accuracy", "auroc", "sensitivity", "specificity"]:
            vals = [m.get(metric_name, float("nan")) for m in self.seed_metrics.values()]
            agg[metric_name] = {
                "mean": float(np.nanmean(vals)),
                "std": float(np.nanstd(vals)),
                "per_seed": {int(s): float(v) for s, v in zip(self.seed_metrics.keys(), vals)},
            }
        self.saver.save_metrics(agg, "aggregate_metrics")
        logger.info("Aggregate F1: %.4f ± %.4f", agg["f1"]["mean"], agg["f1"]["std"])

    @property
    def all_seed_probs(self) -> np.ndarray:
        """Shape: (n_seeds, n_samples, n_classes)"""
        seeds = list(self.seed_probs.keys())
        return np.stack([self.seed_probs[s] for s in seeds], axis=0)

    @property
    def all_seed_preds(self) -> np.ndarray:
        """Shape: (n_seeds, n_samples)"""
        seeds = list(self.seed_preds.keys())
        return np.stack([self.seed_preds[s] for s in seeds], axis=0)
