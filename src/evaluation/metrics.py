from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)


class EvaluationMetrics:
    """Computes all evaluation metrics for binary/multi-class classification."""

    @staticmethod
    def from_predictions(
        labels: list[int],
        preds: list[int],
        probs: list[list[float]] | None = None,
        average: str = "binary",
    ) -> dict[str, Any]:
        y_true = np.array(labels)
        y_pred = np.array(preds)

        metrics: dict[str, Any] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
            "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        }

        if probs is not None:
            p = np.array(probs)
            if p.ndim == 2 and p.shape[1] >= 2:
                # Brier score: mean squared error against one-hot targets
                y_onehot = np.eye(p.shape[1])[y_true]
                metrics["brier_score"] = float(np.mean(np.sum((p - y_onehot) ** 2, axis=1)))
                # Negative Log-Likelihood
                correct_log_p = np.log(p[np.arange(len(y_true)), y_true] + 1e-8)
                metrics["nll"] = float(-np.mean(correct_log_p))
                # For binary: AUROC and AUPRC from positive-class probability
                if p.shape[1] == 2:
                    pos_probs = p[:, 1]
                    try:
                        metrics["auroc"] = float(roc_auc_score(y_true, pos_probs))
                        metrics["auprc"] = float(average_precision_score(y_true, pos_probs))
                    except ValueError:
                        pass
            metrics["ece"] = EvaluationMetrics.expected_calibration_error(y_true, p)

        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics["sensitivity"] = float(tp / (tp + fn + 1e-8))
            metrics["specificity"] = float(tn / (tn + fp + 1e-8))
            metrics["confusion_matrix"] = cm.tolist()

        return metrics

    @staticmethod
    def expected_calibration_error(
        y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15
    ) -> float:
        """Expected Calibration Error (ECE)."""
        if probs.ndim == 2:
            confidences = probs.max(axis=1)
            predictions = probs.argmax(axis=1)
        else:
            confidences = probs
            predictions = (probs > 0.5).astype(int)

        accuracies = (predictions == y_true).astype(float)
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        n = len(y_true)

        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (confidences >= lo) & (confidences < hi)
            if mask.sum() == 0:
                continue
            bin_acc = accuracies[mask].mean()
            bin_conf = confidences[mask].mean()
            ece += mask.sum() / n * abs(bin_acc - bin_conf)

        return float(ece)

    @staticmethod
    def classification_report_str(labels: list[int], preds: list[int]) -> str:
        return classification_report(
            labels, preds, target_names=["Uninfected", "Parasitized"]
        )
