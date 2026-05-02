from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class EnsembleConfidence:
    """
    Multi-seed ensemble confidence estimation.

    Uses predictions from all seeds to compute:
    - Mean probability
    - Per-seed variance and std
    - 95% confidence intervals
    - Seed agreement score

    Failure detection:
    - CI width = 0 → seeds not diverse
    - Agreement always 1 → possible data leakage
    """

    @staticmethod
    def ensemble_stats(
        probs: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """
        Parameters
        ----------
        probs : (n_seeds, n_samples, n_classes)

        Returns
        -------
        dict with mean, std, ci_low, ci_high, ci_width (all shape: n_samples)
        """
        n_seeds = probs.shape[0]
        mean = probs.mean(axis=0)                     # (n_samples, n_classes)
        std = probs.std(axis=0, ddof=1) if n_seeds > 1 else np.zeros_like(mean)
        se = std / np.sqrt(n_seeds)
        ci_low = mean - 1.96 * se
        ci_high = mean + 1.96 * se
        ci_width = ci_high - ci_low

        return {
            "mean": mean,
            "std": std,
            "ci_low": np.clip(ci_low, 0, 1),
            "ci_high": np.clip(ci_high, 0, 1),
            "ci_width": ci_width,
        }

    @staticmethod
    def agreement_score(preds: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        preds : (n_seeds, n_samples) integer predictions

        Returns
        -------
        agreement : (n_samples,) fraction of seeds predicting the majority class
        """
        n_seeds, n_samples = preds.shape
        agreement = np.zeros(n_samples)
        for i in range(n_samples):
            sample_preds = preds[:, i]
            majority = np.bincount(sample_preds).argmax()
            agreement[i] = (sample_preds == majority).sum() / n_seeds
        return agreement

    def run(
        self,
        all_probs: np.ndarray,
        all_preds: np.ndarray,
        labels: np.ndarray,
    ) -> dict[str, Any]:
        """
        Parameters
        ----------
        all_probs : (n_seeds, n_samples, n_classes)
        all_preds : (n_seeds, n_samples)
        labels    : (n_samples,)
        """
        stats_dict = self.ensemble_stats(all_probs)
        agreement = self.agreement_score(all_preds)
        ensemble_preds = stats_dict["mean"].argmax(axis=1)
        errors = (ensemble_preds != labels).astype(float)

        # Validate: wider CI → higher error
        ci_width_pos = stats_dict["ci_width"][:, 1]  # CI width for positive class
        ci_err_corr = float(np.corrcoef(ci_width_pos, errors)[0, 1]) if len(errors) > 1 else 0.0
        agr_err_corr = float(np.corrcoef(1 - agreement, errors)[0, 1]) if len(errors) > 1 else 0.0

        # Failure checks
        if stats_dict["ci_width"].max() < 1e-8:
            logger.error("FAILURE: CI width near zero — seeds may not be diverse.")
        if agreement.min() > 0.999 and len(all_preds) > 1:
            logger.warning("WARNING: Agreement always 1 — check for data leakage.")

        return {
            "mean_probs": stats_dict["mean"],
            "std_probs": stats_dict["std"],
            "ci_low": stats_dict["ci_low"],
            "ci_high": stats_dict["ci_high"],
            "ci_width": stats_dict["ci_width"],
            "agreement": agreement,
            "ensemble_preds": ensemble_preds,
            "ci_error_correlation": ci_err_corr,
            "agreement_error_correlation": agr_err_corr,
            "mean_ci_width": float(ci_width_pos.mean()),
            "mean_agreement": float(agreement.mean()),
        }
