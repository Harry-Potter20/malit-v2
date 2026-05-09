from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation.metrics import EvaluationMetrics
from src.training.cal_loss import CalibrationAwareLoss
from src.utils.save import ArtifactSaver

logger = logging.getLogger(__name__)


class EarlyStopping:
    def __init__(self, patience: int = 3, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.best: float = float("-inf") if mode == "max" else float("inf")
        self.counter: int = 0
        self.best_state: dict | None = None

    def __call__(self, value: float, model: nn.Module) -> bool:
        improved = (self.mode == "max" and value > self.best) or (
            self.mode == "min" and value < self.best
        )
        if improved:
            self.best = value
            self.counter = 0
            self.best_state = {k: v.cpu().contiguous().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


class Trainer:
    """
    Full training loop for MALIT V2.

    Features: AdamW, cosine annealing, AMP, early stopping, gradient clipping.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        epochs: int = 50,
        T_max: int = 50,
        eta_min: float = 1e-6,
        amp: bool = True,
        early_stopping_patience: int = 3,
        grad_clip: float = 1.0,
        device: str | torch.device | None = None,
        saver: ArtifactSaver | None = None,
        model_name: str = "malit_v2",
        seed: int = 42,
        grad_accum_steps: int = 1,
        lambda_cal: float = 0.1,
        cal_warmup_epochs: int = 3,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.amp = amp and self.device.type == "cuda"
        self.grad_clip = grad_clip
        self.saver = saver
        self.model_name = model_name
        self.seed = seed

        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=T_max, eta_min=eta_min
        )
        self.grad_accum_steps = max(1, grad_accum_steps)
        self.scaler = GradScaler("cuda", enabled=self.amp)
        self.criterion = CalibrationAwareLoss(lambda_cal=lambda_cal)
        self.cal_warmup_epochs = cal_warmup_epochs
        self.early_stopping = EarlyStopping(patience=early_stopping_patience, mode="max")
        self.history: list[dict[str, Any]] = []
        self.calibration_history: list[dict[str, Any]] = []
        self._track_his = hasattr(model, "gabor_lcci") and hasattr(model, "backbone_lcci")

    # ── Training ────────────────────────────────────────────────────────────

    def train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        total_loss, ece_loss_total, all_preds, all_labels = 0.0, 0.0, [], []
        self.optimizer.zero_grad(set_to_none=True)
        apply_cal = epoch >= self.cal_warmup_epochs

        n_batches = len(self.train_loader)
        for batch_idx, (imgs, labels) in enumerate(
            tqdm(self.train_loader, desc="Train", leave=False)
        ):
            imgs, labels = imgs.to(self.device), labels.to(self.device)

            with autocast("cuda", enabled=self.amp):
                logits, _ = self.model(imgs)
                loss, loss_comps = self.criterion(logits, labels, apply_calibration=apply_cal)

            # Divide loss so gradient magnitudes are independent of accum_steps
            self.scaler.scale(loss / self.grad_accum_steps).backward()

            total_loss += loss.item() * len(labels)
            ece_loss_total += loss_comps["ece"] * len(labels)
            all_preds.extend(logits.detach().argmax(1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            is_last_batch = (batch_idx + 1) == n_batches
            if (batch_idx + 1) % self.grad_accum_steps == 0 or is_last_batch:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

        n = len(self.train_loader.dataset)
        metrics = EvaluationMetrics.from_predictions(all_labels, all_preds)
        metrics["loss"] = total_loss / n
        metrics["ece_loss"] = ece_loss_total / n
        return metrics

    @torch.no_grad()
    def eval_epoch(self, loader: DataLoader) -> tuple[dict[str, float], list, list]:
        self.model.eval()
        total_loss, all_preds, all_labels, all_probs = 0.0, [], [], []

        for imgs, labels in tqdm(loader, desc="Eval", leave=False):
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            with autocast("cuda", enabled=self.amp):
                logits, _ = self.model(imgs)
                loss, _ = self.criterion(logits, labels, apply_calibration=False)

            probs = torch.softmax(logits, dim=1)
            total_loss += loss.item() * len(labels)
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

        metrics = EvaluationMetrics.from_predictions(all_labels, all_preds, all_probs)
        metrics["loss"] = total_loss / len(loader.dataset)
        metrics.update(self._compute_cal_metrics(all_probs, all_labels))
        return metrics, all_preds, all_labels

    @staticmethod
    def _compute_cal_metrics(
        all_probs: list, all_labels: list, n_bins: int = 10
    ) -> dict[str, float]:
        probs = np.array(all_probs)       # (N, C)
        labels = np.array(all_labels)     # (N,)
        confidence = probs.max(axis=1)
        preds = probs.argmax(axis=1)
        correct = (preds == labels).astype(float)
        N = len(labels)

        ece = 0.0
        for b in range(n_bins):
            lo, hi = b / n_bins, (b + 1) / n_bins
            mask = (confidence > lo) & (confidence <= hi)
            if mask.sum() > 0:
                ece += mask.sum() / N * abs(correct[mask].mean() - confidence[mask].mean())

        target_oh = np.zeros_like(probs)
        target_oh[np.arange(N), labels] = 1.0
        brier = float(np.mean(np.sum((probs - target_oh) ** 2, axis=1)))

        p_true = np.clip(probs[np.arange(N), labels], 1e-8, 1.0)
        nll = float(-np.mean(np.log(p_true)))

        return {
            "ece": float(ece),
            "brier": float(brier),
            "nll": float(nll),
            "confidence_mean": float(confidence.mean()),
            "confidence_var": float(confidence.var()),
        }

    # ── Main fit loop ────────────────────────────────────────────────────────

    def fit(self) -> dict[str, Any]:
        logger.info("Training on %s  (AMP=%s)", self.device, self.amp)
        best_metrics = {}

        his_tracker = None
        if self._track_his:
            from src.interpretability.his_tracking import HISTracker
            from src.interpretability.his_analysis import extract_his_stats
            his_tracker = HISTracker()

        for epoch in range(1, self.epochs + 1):
            train_m = self.train_epoch(epoch)
            val_m, val_preds, val_labels = self.eval_epoch(self.val_loader)
            self.scheduler.step()

            row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_m.items()},
                   **{f"val_{k}": v for k, v in val_m.items()}}
            self.history.append(row)

            cal_row = {
                "epoch": epoch,
                "train_ece_loss": train_m.get("ece_loss", 0.0),
                "val_ece": val_m.get("ece", 0.0),
                "val_brier": val_m.get("brier", 0.0),
                "val_nll": val_m.get("nll", 0.0),
                "val_confidence_mean": val_m.get("confidence_mean", 0.0),
                "val_confidence_var": val_m.get("confidence_var", 0.0),
            }
            self.calibration_history.append(cal_row)

            if his_tracker is not None:
                his_stats = extract_his_stats(self.model)
                his_tracker.update(epoch, his_stats)

            val_f1 = val_m.get("f1", 0.0)
            logger.info(
                "Epoch %d/%d  train_loss=%.4f  val_f1=%.4f  val_ece=%.4f",
                epoch, self.epochs, train_m["loss"], val_f1, val_m.get("ece", 0.0),
            )

            stop = self.early_stopping(val_f1, self.model)
            if val_f1 >= self.early_stopping.best and self.saver:
                self.saver.save_model(self.model, self.model_name, self.seed, epoch)
                best_metrics = val_m

            if stop:
                logger.info("Early stopping at epoch %d", epoch)
                break

        self.early_stopping.restore_best(self.model)

        if self.saver and self.calibration_history:
            self.saver.save_metrics(
                {f"epoch_{r['epoch']}": {k: v for k, v in r.items() if k != "epoch"}
                 for r in self.calibration_history},
                f"calibration_curves_seed{self.seed}",
            )

        if self.saver and his_tracker and his_tracker.history:
            self.saver.save_metrics(his_tracker.history, f"his_dynamics_seed{self.seed}")

        return best_metrics

    # ── Evaluation on held-out test set ─────────────────────────────────────

    def evaluate(self, test_loader: DataLoader) -> tuple[dict[str, float], list, list, list]:
        metrics, preds, labels = self.eval_epoch(test_loader)
        return metrics, preds, labels, self.history
