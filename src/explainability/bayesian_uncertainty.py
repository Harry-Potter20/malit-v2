from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


def enable_mc_dropout(model: nn.Module) -> None:
    """Keep Dropout layers in train mode during inference for MC sampling."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


class BayesianUncertainty:
    """
    MC Dropout uncertainty estimation.

    E[p(y|x)] ≈ (1/T) Σ p(y|x, w_t)
    Var[p(y|x)] ≈ (1/T) Σ (p_t − mean)²

    Failure detection:
    - Zero variance across all samples → dropout not active
    - Identical outputs across passes → implementation bug
    """

    def __init__(self, T: int = 30, device: str | torch.device | None = None):
        self.T = T
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

    def mc_dropout_predict(
        self,
        model: nn.Module,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run T stochastic forward passes on a single batch tensor.

        Returns
        -------
        mean : (B, C) mean probabilities
        var  : (B, C) variance of probabilities
        """
        model.eval()
        enable_mc_dropout(model)
        x = x.to(self.device)

        all_probs = []
        with torch.no_grad():
            for _ in range(self.T):
                logits, _ = model(x)
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs.cpu())

        stacked = torch.stack(all_probs, dim=0)  # (T, B, C)
        mean = stacked.mean(dim=0)               # (B, C)
        var = stacked.var(dim=0)                 # (B, C)
        return mean, var

    def run_on_loader(
        self, model: nn.Module, loader: DataLoader
    ) -> dict[str, np.ndarray]:
        means, variances, entropies, labels_all = [], [], [], []

        for imgs, labels in tqdm(loader, desc="MC Dropout"):
            mean, var = self.mc_dropout_predict(model, imgs)
            entropy = -(mean * (mean + 1e-8).log()).sum(dim=1)
            means.append(mean.numpy())
            variances.append(var.numpy())
            entropies.append(entropy.numpy())
            labels_all.extend(labels.tolist())

        mean_probs = np.concatenate(means, axis=0)
        var_probs = np.concatenate(variances, axis=0)
        pred_entropy = np.concatenate(entropies, axis=0)

        self._validate(var_probs, mean_probs)

        return {
            "mean_probs": mean_probs,
            "variance": var_probs,
            "predictive_entropy": pred_entropy,
            "labels": np.array(labels_all),
            "T": self.T,
        }

    @staticmethod
    def _validate(var: np.ndarray, mean: np.ndarray) -> None:
        if var.max() < 1e-10:
            logger.error("FAILURE: MC Dropout variance near zero — Dropout may not be active.")
        if mean.std(axis=0).max() < 1e-10:
            logger.warning("WARNING: Low diversity across MC passes — check model structure.")

    @staticmethod
    def uncertainty_summary(results: dict[str, np.ndarray]) -> dict[str, Any]:
        var = results["variance"]
        entropy = results["predictive_entropy"]
        labels = results["labels"]
        preds = results["mean_probs"].argmax(axis=1)
        errors = (preds != labels).astype(float)

        # Validate: higher variance → higher error
        pos_var_corr = float(np.corrcoef(var[:, 1], errors)[0, 1]) if len(errors) > 1 else 0.0
        ent_err_corr = float(np.corrcoef(entropy, errors)[0, 1]) if len(errors) > 1 else 0.0

        return {
            "mean_variance": float(var.mean()),
            "mean_entropy": float(entropy.mean()),
            "variance_error_correlation": pos_var_corr,
            "entropy_error_correlation": ent_err_corr,
            "n_samples": len(labels),
            "T": int(results["T"]),
        }
