"""Runtime validation utilities: detect NaNs, bad CIs, and failure conditions."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_HIGH_CONF_THRESHOLD = 0.90
_HIGH_ENT_THRESHOLD = 0.60
_LOW_AGR_THRESHOLD = 0.50


def validate_metrics(metrics: dict[str, Any]) -> list[str]:
    """Return list of issues (NaN or Inf) found in a flat metrics dict."""
    issues = []
    for key, val in metrics.items():
        if isinstance(val, (int, float)):
            if np.isnan(val):
                issues.append(f"NaN in metric '{key}'")
            elif np.isinf(val):
                issues.append(f"Inf in metric '{key}'")
    if issues:
        for msg in issues:
            logger.error("METRIC VALIDATION: %s", msg)
    return issues


def validate_confidence_intervals(
    ci_low: np.ndarray,
    ci_high: np.ndarray,
    label: str = "CI",
) -> list[str]:
    """Return issues when CI bounds are inverted (ci_low > ci_high)."""
    issues = []
    n_bad = int((ci_low > ci_high).sum())
    if n_bad > 0:
        issues.append(f"{label}: {n_bad} sample(s) have ci_low > ci_high")
        logger.error("CI VALIDATION: %s", issues[-1])
    return issues


def validate_variance(var: np.ndarray, label: str = "variance") -> list[str]:
    """Return issues if variance array contains negative values."""
    issues = []
    if np.any(var < -1e-10):
        issues.append(f"{label}: negative values found (min={float(var.min()):.2e})")
        logger.error("VARIANCE VALIDATION: %s", issues[-1])
    return issues


def check_failure_conditions(
    probs: np.ndarray,
    labels: np.ndarray,
    variance: np.ndarray | None = None,
    ci_width: np.ndarray | None = None,
    agreement: np.ndarray | None = None,
    entropy: np.ndarray | None = None,
    n_seeds: int = 1,
) -> dict[str, list[str]]:
    """
    Detect critical and warning failure conditions.

    Critical
    --------
    - High-confidence incorrect predictions (>5% of samples)
    - Zero ensemble variance for multi-seed runs
    - Zero CI width for multi-seed runs

    Warning
    -------
    - High entropy + high confidence (softmax miscalibration)
    - Low seed agreement + high confidence

    Returns
    -------
    {"critical": [...], "warning": [...]}
    """
    preds = probs.argmax(axis=1)
    errors = preds != labels
    confidences = probs.max(axis=1)

    critical: list[str] = []
    warnings: list[str] = []

    # Critical: high confidence + incorrect
    high_conf_errors = errors & (confidences > _HIGH_CONF_THRESHOLD)
    error_frac = high_conf_errors.sum() / max(len(labels), 1)
    if error_frac > 0.05:
        critical.append(
            f"High-confidence errors: {int(high_conf_errors.sum())}/{len(labels)} "
            f"samples with conf>{_HIGH_CONF_THRESHOLD:.0%} are wrong"
        )

    # Critical: zero variance in multi-seed ensemble
    if variance is not None and n_seeds > 1 and float(variance.max()) < 1e-8:
        critical.append("Zero ensemble variance — seeds may not be diverse")

    # Critical: zero CI width for multi-seed run
    if ci_width is not None and n_seeds > 1 and float(ci_width.max()) < 1e-8:
        critical.append("CI width near zero for multi-seed run — check seed diversity")

    # Warning: high entropy + high confidence (softmax miscalibration)
    if entropy is not None:
        n_classes = probs.shape[1]
        h_norm = entropy / (np.log2(n_classes) + 1e-12)
        suspect = (h_norm > _HIGH_ENT_THRESHOLD) & (confidences > _HIGH_CONF_THRESHOLD)
        if int(suspect.sum()) > 0:
            warnings.append(
                f"High entropy + high confidence: {int(suspect.sum())} sample(s) "
                "(possible softmax miscalibration)"
            )

    # Warning: low seed agreement + high confidence
    if agreement is not None:
        suspect_agr = (agreement < _LOW_AGR_THRESHOLD) & (confidences > _HIGH_CONF_THRESHOLD)
        if int(suspect_agr.sum()) > 0:
            warnings.append(
                f"Low seed agreement + high confidence: {int(suspect_agr.sum())} sample(s)"
            )

    for msg in critical:
        logger.error("CRITICAL: %s", msg)
    for msg in warnings:
        logger.warning("WARNING: %s", msg)

    return {"critical": critical, "warning": warnings}
