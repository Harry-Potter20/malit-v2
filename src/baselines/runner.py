"""
BaselineRunner: trains all baseline models under identical conditions to MALIT V2.

Enforces:
  - Same data splits (caller must pass loaders from the same DataPipeline)
  - Same optimizer, scheduler, AMP, early stopping
  - Seeds [42, 123] (Tier 2)
  - Results saved to results/baselines/{model_name}/
  - GPU memory cleanup between model runs
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.baselines.models import ALL_BASELINE_NAMES, build_baseline
from src.training.trainer import Trainer
from src.utils.save import ArtifactSaver, _make_serializable
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)


class BaselineRunner:
    """
    Trains all (or a specified subset of) baseline models with identical
    hyperparameters and data splits, then stores per-seed results.

    Parameters
    ----------
    grad_accum_steps : int
        Gradient accumulation steps. Use > 1 when physical batch size is
        limited by VRAM (e.g., 4 steps × batch_size=8 ≈ effective batch 32).
    use_gradient_checkpointing : bool
        Recompute activations during backward pass to save activation memory.
        Recommended when VRAM < 8 GB.
    """

    def __init__(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        saver: ArtifactSaver,
        device: str | torch.device | None = None,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        epochs: int = 50,
        amp: bool = True,
        patience: int = 3,
        grad_clip: float = 1.0,
        grad_accum_steps: int = 1,
        use_gradient_checkpointing: bool = False,
        seeds: list[int] | None = None,
        num_classes: int = 2,
        pretrained: bool = True,
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.saver = saver
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.amp = amp
        self.patience = patience
        self.grad_clip = grad_clip
        self.grad_accum_steps = grad_accum_steps
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.seeds = seeds if seeds is not None else [42, 123]
        self.num_classes = num_classes
        self.pretrained = pretrained
        self._results: dict[str, dict[str, Any]] = {}

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(
        self,
        model_names: list[str] | None = None,
    ) -> "BaselineRunner":
        """Train all specified baseline models across all seeds.

        Parameters
        ----------
        model_names : list of model names to train. Defaults to ALL_BASELINE_NAMES.
        """
        names = model_names if model_names is not None else ALL_BASELINE_NAMES
        logger.info("BaselineRunner: %d models × %d seeds", len(names), len(self.seeds))

        for model_name in names:
            logger.info("══ Baseline: %s ══", model_name)
            self._train_one_model(model_name)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self._save_hparams()
        return self

    @property
    def results(self) -> dict[str, dict[str, Any]]:
        return self._results

    # ── Private helpers ────────────────────────────────────────────────────────

    def _train_one_model(self, model_name: str) -> None:
        seed_preds: dict[int, list[int]] = {}
        seed_metrics: dict[int, dict[str, Any]] = {}
        seed_probs: dict[int, np.ndarray] = {}
        seed_labels: list[int] | None = None

        out_dir = Path(self.saver.base) / "results" / "baselines" / model_name
        out_dir.mkdir(parents=True, exist_ok=True)

        for seed in self.seeds:
            logger.info("  Seed %d", seed)
            set_seed(seed)

            model = build_baseline(
                model_name,
                num_classes=self.num_classes,
                pretrained=self.pretrained,
                use_gradient_checkpointing=self.use_gradient_checkpointing,
            )

            trainer = Trainer(
                model=model,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                lr=self.lr,
                weight_decay=self.weight_decay,
                epochs=self.epochs,
                T_max=self.epochs,
                amp=self.amp,
                early_stopping_patience=self.patience,
                grad_clip=self.grad_clip,
                grad_accum_steps=self.grad_accum_steps,
                device=self.device,
                saver=self.saver,
                model_name=model_name,
                seed=seed,
            )

            trainer.fit()
            metrics, preds, labels, _ = trainer.evaluate(self.test_loader)
            probs = self._collect_probs(model)

            seed_preds[seed] = preds
            seed_metrics[seed] = metrics
            seed_labels = labels
            seed_probs[seed] = np.array(probs)

            # Save per-seed metrics
            per_seed_path = out_dir / f"seed_{seed}_metrics.json"
            with open(per_seed_path, "w") as f:
                json.dump(_make_serializable({"seed": seed, "metrics": metrics}), f, indent=2)

            logger.info(
                "  %s seed=%d  F1=%.4f  Acc=%.4f",
                model_name, seed,
                metrics.get("f1", 0.0), metrics.get("accuracy", 0.0),
            )

            # Free model memory between seeds
            del model, trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self._results[model_name] = {
            "seed_preds":   seed_preds,
            "seed_metrics": seed_metrics,
            "seed_labels":  seed_labels,
            "seed_probs":   seed_probs,
        }
        self._save_aggregate(model_name, seed_metrics, out_dir)

    def _save_hparams(self) -> None:
        out_dir = Path(self.saver.base) / "results" / "baselines"
        out_dir.mkdir(parents=True, exist_ok=True)
        hparams = {
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "epochs": self.epochs,
            "amp": self.amp,
            "patience": self.patience,
            "grad_clip": self.grad_clip,
            "grad_accum_steps": self.grad_accum_steps,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "seeds": self.seeds,
            "num_classes": self.num_classes,
            "pretrained": self.pretrained,
            "note": "Identical hyperparameters used for all baselines (Option A).",
        }
        with open(out_dir / "hparams.json", "w") as f:
            json.dump(hparams, f, indent=2)
        logger.info("Saved baseline hyperparameters to %s", out_dir / "hparams.json")

    @torch.no_grad()
    def _collect_probs(self, model: torch.nn.Module) -> list[list[float]]:
        """Collect softmax probabilities over the test set."""
        model.eval()
        model.to(self.device)
        probs_list = []
        for imgs, _ in self.test_loader:
            imgs = imgs.to(self.device)
            logits, _ = model(imgs)
            probs = torch.softmax(logits, dim=1)
            probs_list.extend(probs.cpu().tolist())
        return probs_list

    def _save_aggregate(
        self,
        model_name: str,
        seed_metrics: dict[int, dict[str, Any]],
        out_dir: Path,
    ) -> None:
        agg: dict[str, Any] = {}
        for metric_name in ["f1", "accuracy", "auroc", "sensitivity", "specificity"]:
            vals = [m.get(metric_name, float("nan")) for m in seed_metrics.values()]
            agg[metric_name] = {
                "mean": float(np.nanmean(vals)),
                "std": float(np.nanstd(vals, ddof=1) if len(vals) >= 2 else 0.0),
                "per_seed": {int(s): float(v) for s, v in zip(seed_metrics.keys(), vals)},
            }
        with open(out_dir / "aggregate_metrics.json", "w") as f:
            json.dump(_make_serializable(agg), f, indent=2)
        logger.info(
            "%s aggregate  F1=%.4f ± %.4f",
            model_name, agg["f1"]["mean"], agg["f1"]["std"],
        )
