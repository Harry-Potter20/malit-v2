from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation.metrics import EvaluationMetrics
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
        self.criterion = nn.CrossEntropyLoss()
        self.early_stopping = EarlyStopping(patience=early_stopping_patience, mode="max")
        self.history: list[dict[str, Any]] = []

    # ── Training ────────────────────────────────────────────────────────────

    def train_epoch(self) -> dict[str, float]:
        self.model.train()
        total_loss, all_preds, all_labels = 0.0, [], []
        self.optimizer.zero_grad(set_to_none=True)

        n_batches = len(self.train_loader)
        for batch_idx, (imgs, labels) in enumerate(
            tqdm(self.train_loader, desc="Train", leave=False)
        ):
            imgs, labels = imgs.to(self.device), labels.to(self.device)

            with autocast("cuda", enabled=self.amp):
                logits, _ = self.model(imgs)
                loss = self.criterion(logits, labels)

            # Divide loss so gradient magnitudes are independent of accum_steps
            self.scaler.scale(loss / self.grad_accum_steps).backward()

            total_loss += loss.item() * len(labels)
            all_preds.extend(logits.detach().argmax(1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            is_last_batch = (batch_idx + 1) == n_batches
            if (batch_idx + 1) % self.grad_accum_steps == 0 or is_last_batch:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

        metrics = EvaluationMetrics.from_predictions(all_labels, all_preds)
        metrics["loss"] = total_loss / len(self.train_loader.dataset)
        return metrics

    @torch.no_grad()
    def eval_epoch(self, loader: DataLoader) -> tuple[dict[str, float], list, list]:
        self.model.eval()
        total_loss, all_preds, all_labels, all_probs = 0.0, [], [], []

        for imgs, labels in tqdm(loader, desc="Eval", leave=False):
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            with autocast("cuda", enabled=self.amp):
                logits, _ = self.model(imgs)
                loss = self.criterion(logits, labels)

            probs = torch.softmax(logits, dim=1)
            total_loss += loss.item() * len(labels)
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

        metrics = EvaluationMetrics.from_predictions(all_labels, all_preds, all_probs)
        metrics["loss"] = total_loss / len(loader.dataset)
        return metrics, all_preds, all_labels

    # ── Main fit loop ────────────────────────────────────────────────────────

    def fit(self) -> dict[str, Any]:
        logger.info("Training on %s  (AMP=%s)", self.device, self.amp)
        best_metrics = {}

        for epoch in range(1, self.epochs + 1):
            train_m = self.train_epoch()
            val_m, val_preds, val_labels = self.eval_epoch(self.val_loader)
            self.scheduler.step()

            row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_m.items()},
                   **{f"val_{k}": v for k, v in val_m.items()}}
            self.history.append(row)

            val_f1 = val_m.get("f1", 0.0)
            logger.info(
                "Epoch %d/%d  train_loss=%.4f  val_f1=%.4f  val_acc=%.4f",
                epoch, self.epochs, train_m["loss"], val_f1, val_m.get("accuracy", 0.0),
            )

            stop = self.early_stopping(val_f1, self.model)
            if val_f1 >= self.early_stopping.best and self.saver:
                self.saver.save_model(self.model, self.model_name, self.seed, epoch)
                best_metrics = val_m

            if stop:
                logger.info("Early stopping at epoch %d", epoch)
                break

        self.early_stopping.restore_best(self.model)
        return best_metrics

    # ── Evaluation on held-out test set ─────────────────────────────────────

    def evaluate(self, test_loader: DataLoader) -> tuple[dict[str, float], list, list, list]:
        metrics, preds, labels = self.eval_epoch(test_loader)
        return metrics, preds, labels, self.history
