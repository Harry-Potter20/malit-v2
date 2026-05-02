"""
Paper-ready reporting: performance tables, statistical tables, and figures.

All plot functions degrade gracefully when matplotlib/sklearn are unavailable.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.statistics.tests import _safe_ci, _safe_std

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend for CI / Kaggle
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import auc, roc_curve
    _PLOT_AVAILABLE = True
except ImportError:
    _PLOT_AVAILABLE = False


# ── Performance table ─────────────────────────────────────────────────────────

def build_performance_table(all_results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """
    Build a DataFrame with one row per model and mean ± CI columns.

    Parameters
    ----------
    all_results : {model_name: {"seed_metrics": {seed: metric_dict}}}
    """
    rows = []
    for model_name, data in all_results.items():
        seed_metrics = list(data.get("seed_metrics", {}).values())
        if not seed_metrics:
            continue

        def stat(key: str) -> tuple[float, float, float]:
            vals = [m.get(key, float("nan")) for m in seed_metrics]
            vals = [v for v in vals if not np.isnan(v)]
            if not vals:
                return float("nan"), float("nan"), float("nan")
            mean = float(np.mean(vals))
            std = _safe_std(np.array(vals))
            _, ci_hi = _safe_ci(mean, std, len(vals))
            ci_half = ci_hi - mean
            return mean, std, ci_half

        f1_m, f1_s, f1_ci = stat("f1")
        sens_m, _, _ = stat("sensitivity")
        spec_m, _, _ = stat("specificity")
        auc_m, _, _ = stat("auroc")
        ece_m, _, _ = stat("ece")
        brier_m, _, _ = stat("brier_score")
        nll_m, _, _ = stat("nll")

        rows.append({
            "Model":        model_name,
            "F1":           f"{f1_m:.4f}" if not np.isnan(f1_m) else "—",
            "F1_CI":        f"±{f1_ci:.4f}" if not np.isnan(f1_ci) else "—",
            "F1 (mean±CI)": (
                f"{f1_m:.4f} ± {f1_ci:.4f}"
                if not np.isnan(f1_m) else "—"
            ),
            "Sensitivity":  f"{sens_m:.4f}" if not np.isnan(sens_m) else "—",
            "Specificity":  f"{spec_m:.4f}" if not np.isnan(spec_m) else "—",
            "AUC":          f"{auc_m:.4f}"  if not np.isnan(auc_m)  else "—",
            "ECE":          f"{ece_m:.4f}"  if not np.isnan(ece_m)  else "—",
            "Brier":        f"{brier_m:.4f}" if not np.isnan(brier_m) else "—",
            "NLL":          f"{nll_m:.4f}"  if not np.isnan(nll_m)  else "—",
            # Raw floats for programmatic use
            "_f1_mean":     f1_m,
            "_f1_std":      f1_s,
            "_n_seeds":     len(seed_metrics),
        })

    return pd.DataFrame(rows)


# ── Statistical comparison table ──────────────────────────────────────────────

def build_statistical_table(
    comparison_results: dict[str, dict[str, Any]]
) -> pd.DataFrame:
    """
    Build a DataFrame with one row per baseline comparison.

    Parameters
    ----------
    comparison_results : output of ComparisonProtocol.compare_all()
    """
    rows = []
    for model_name, cmp in comparison_results.items():
        mcnemar_p = cmp.get("mcnemar", {}).get("p_value", float("nan"))
        tost = cmp.get("tost", {})
        majority_sig = cmp.get("multi_seed_mcnemar", {}).get(
            "aggregated", {}
        ).get("majority_significant", False)
        tost_display = (
            "Indeterminate (low variance)"
            if tost.get("indeterminate") else
            str(tost.get("equivalent", False))
        )
        rows.append({
            "Baseline":             model_name,
            "ΔF1":                  f"{cmp.get('delta_f1', float('nan')):+.4f}",
            "McNemar_p":            (
                f"{mcnemar_p:.4f}" if not np.isnan(mcnemar_p) else "—"
            ),
            "McNemar_majority_sig": str(majority_sig),
            "TOST_result":          tost_display,
            "Effect_size":          (
                f"{cmp['effect_size']:.3f}"
                if cmp.get("effect_size") is not None else "—"
            ),
            "Conclusion":           cmp.get("verdict", "Inconclusive"),
            "Failure_flags":        ", ".join(cmp.get("failure_flags", [])) or "none",
            # Raw floats
            "_delta_f1":            cmp.get("delta_f1", float("nan")),
            "_mcnemar_p":           mcnemar_p,
        })
    return pd.DataFrame(rows)


# ── Save tables ───────────────────────────────────────────────────────────────

def save_paper_tables(
    tables: dict[str, pd.DataFrame],
    output_dir: str | Path,
) -> dict[str, Path]:
    """
    Save each DataFrame as CSV and LaTeX.

    Returns mapping of {table_name: csv_path}.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}

    for name, df in tables.items():
        # Drop internal "_*" helper columns before saving
        display_df = df[[c for c in df.columns if not c.startswith("_")]].copy()

        csv_path = output_dir / f"{name}.csv"
        display_df.to_csv(csv_path, index=False)
        saved[name] = csv_path

        tex_path = output_dir / f"{name}.tex"
        try:
            tex_str = display_df.to_latex(index=False, float_format="%.4f")
            tex_path.write_text(tex_str)
        except Exception as exc:
            logger.warning("LaTeX export failed for '%s': %s", name, exc)

    return saved


# ── Figures ───────────────────────────────────────────────────────────────────

def plot_confidence_histogram(
    confidences: list[float] | np.ndarray,
    errors: list[int] | np.ndarray,
    output_path: str | Path,
) -> Any | None:
    """
    Histogram of confidence scores split by correct vs incorrect predictions.

    Shows whether the model assigns higher confidence to correct predictions.
    Saved to results/plots/confidence_histogram.png (or output_path).
    Returns matplotlib Figure or None if matplotlib unavailable.
    """
    if not _PLOT_AVAILABLE:
        logger.warning("matplotlib not available — skipping confidence histogram")
        return None

    try:
        conf = np.array(confidences, dtype=float)
        err_mask = np.array(errors, dtype=bool)
        correct = conf[~err_mask]
        incorrect = conf[err_mask]

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(correct, bins=20, alpha=0.6, color="steelblue",
                label=f"Correct (n={len(correct)})")
        ax.hist(incorrect, bins=20, alpha=0.6, color="tomato",
                label=f"Incorrect (n={len(incorrect)})")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Frequency")
        ax.set_title("Confidence Distribution: Correct vs Incorrect Predictions")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        return fig
    except Exception as exc:
        logger.warning("Confidence histogram failed: %s", exc)
        return None


def plot_roc_curves(
    all_results: dict[str, dict[str, Any]],
    output_path: str | Path,
) -> Any | None:
    """
    Plot ROC curves for all models on one figure (seed-42 probabilities).

    Returns matplotlib Figure or None if matplotlib unavailable.
    """
    if not _PLOT_AVAILABLE:
        logger.warning("matplotlib not available — skipping ROC plot")
        return None

    try:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")

        for model_name, data in all_results.items():
            probs_dict = data.get("seed_probs", {})
            labels = data.get("seed_labels")
            if not probs_dict or labels is None:
                continue
            # Use seed 42 if available, otherwise first seed
            seed = 42 if 42 in probs_dict else next(iter(probs_dict))
            probs = np.array(probs_dict[seed])
            if probs.ndim == 2:
                pos_probs = probs[:, 1]
            else:
                pos_probs = probs

            try:
                fpr, tpr, _ = roc_curve(labels, pos_probs)
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, lw=1.5, label=f"{model_name} (AUC={roc_auc:.3f})")
            except Exception:
                continue

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        return fig
    except Exception as exc:
        logger.warning("ROC plot failed: %s", exc)
        return None


def plot_calibration(
    all_results: dict[str, dict[str, Any]],
    output_path: str | Path,
) -> Any | None:
    """
    Reliability diagram (calibration curve) for all models.

    Returns matplotlib Figure or None if matplotlib unavailable.
    """
    if not _PLOT_AVAILABLE:
        logger.warning("matplotlib not available — skipping calibration plot")
        return None

    try:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")

        for model_name, data in all_results.items():
            probs_dict = data.get("seed_probs", {})
            labels = data.get("seed_labels")
            if not probs_dict or labels is None:
                continue
            seed = 42 if 42 in probs_dict else next(iter(probs_dict))
            probs = np.array(probs_dict[seed])
            pos_probs = probs[:, 1] if probs.ndim == 2 else probs

            try:
                prob_true, prob_pred = calibration_curve(
                    labels, pos_probs, n_bins=10
                )
                ax.plot(prob_pred, prob_true, "s-", lw=1.5,
                        markersize=4, label=model_name)
            except Exception:
                continue

        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title("Calibration Curves")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        return fig
    except Exception as exc:
        logger.warning("Calibration plot failed: %s", exc)
        return None


def plot_ci_width_vs_error(
    ensemble_results: dict[str, Any],
    output_path: str | Path,
) -> Any | None:
    """
    Scatter plot: CI width vs per-sample error (0/1).

    Parameters
    ----------
    ensemble_results : output of EnsembleConfidence.run()
    """
    if not _PLOT_AVAILABLE:
        logger.warning("matplotlib not available — skipping CI-vs-error plot")
        return None

    try:
        ci_width = np.array(ensemble_results.get("ci_width", []))
        preds = np.array(ensemble_results.get("ensemble_preds", []))
        labels = np.array(ensemble_results.get("labels", []))

        if not len(ci_width) or not len(labels):
            return None

        errors = (preds != labels).astype(float)
        ci_pos = ci_width[:, 1] if ci_width.ndim == 2 else ci_width

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(ci_pos[errors == 0], np.zeros(int((errors == 0).sum())),
                   alpha=0.3, s=8, label="Correct", color="steelblue")
        ax.scatter(ci_pos[errors == 1], np.ones(int((errors == 1).sum())),
                   alpha=0.3, s=8, label="Error", color="tomato")
        ax.set_xlabel("CI Width")
        ax.set_ylabel("Error (0=correct, 1=wrong)")
        ax.set_title("CI Width vs Prediction Error")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        return fig
    except Exception as exc:
        logger.warning("CI-width-vs-error plot failed: %s", exc)
        return None


# ── Confidence vs error analysis ─────────────────────────────────────────────

def compute_confidence_bins(
    confidences: list[float] | np.ndarray,
    errors: list[int] | np.ndarray,
    bins: int = 10,
) -> pd.DataFrame:
    """
    Bin predictions by confidence and compute error rate per bin.

    Returns a DataFrame with columns: bin_low, bin_high, n, error_rate.
    Higher confidence bins should show lower error rates.
    """
    conf = np.array(confidences, dtype=float)
    err = np.array(errors, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf >= lo) & (conf < hi) if i < bins - 1 else (conf >= lo) & (conf <= hi)
        n = int(mask.sum())
        error_rate = float(err[mask].mean()) if n > 0 else float("nan")
        rows.append({"bin_low": round(lo, 2), "bin_high": round(hi, 2),
                     "n": n, "error_rate": error_rate})
    return pd.DataFrame(rows)


def check_confidence_monotonicity(
    conf_df: pd.DataFrame,
    tolerance: float = 0.02,
    max_violations: int = 1,
) -> dict[str, Any]:
    """
    Verify that error rate decreases as confidence increases, with noise tolerance.

    A violation is only counted when error_rate rises by more than `tolerance`.
    Up to `max_violations` violations are allowed before flagging as non-monotonic.
    Minor violations are logged as warnings rather than hard failures.

    Returns dict with 'monotonic' bool, 'n_violations', 'n_bins_checked', and note.
    """
    valid_rates = conf_df[conf_df["n"] > 0]["error_rate"].dropna().values
    if len(valid_rates) < 2:
        return {
            "monotonic": None,
            "n_violations": 0,
            "n_bins_checked": len(valid_rates),
            "tolerance": tolerance,
            "note": "Insufficient bins with data for monotonicity check",
        }

    n_violations = int(sum(
        valid_rates[i] > valid_rates[i - 1] + tolerance
        for i in range(1, len(valid_rates))
    ))
    monotonic = n_violations <= max_violations
    note = (
        "Monotonic: higher confidence → lower error"
        if n_violations == 0 else
        f"{n_violations} violation(s) within tolerance ({tolerance}) — treating as monotonic"
        if monotonic else
        f"{n_violations} violation(s) exceed tolerance — non-monotonic confidence-error relationship"
    )
    if n_violations > 0:
        log_fn = logger.warning if monotonic else logger.error
        log_fn("Confidence monotonicity: %s", note)
    return {
        "monotonic": monotonic,
        "n_violations": n_violations,
        "n_bins_checked": len(valid_rates),
        "tolerance": tolerance,
        "note": note,
    }


# ── MALIT non-regression check ────────────────────────────────────────────────

def check_malit_non_regression(
    malit_f1: float,
    baseline_results: dict[str, dict[str, Any]],
    tolerance: float = 0.01,
) -> dict[str, Any]:
    """
    Soft non-regression check: warn when MALIT is slightly below best baseline,
    flag as failed when it exceeds the tolerance gap.

    Severity levels:
      "pass"    — malit_f1 >= best_baseline_f1
      "warning" — best_baseline_f1 - tolerance <= malit_f1 < best_baseline_f1
      "fail"    — malit_f1 < best_baseline_f1 - tolerance
    """
    best_name: str | None = None
    best_f1 = float("-inf")

    for name, data in baseline_results.items():
        vals = [
            m.get("f1", float("nan"))
            for m in data.get("seed_metrics", {}).values()
        ]
        vals = [v for v in vals if not np.isnan(v)]
        if not vals:
            continue
        mean_f1 = float(np.mean(vals))
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_name = name

    if best_name is None:
        return {"passed": None, "severity": "unknown", "note": "No baseline F1 scores available"}

    threshold = best_f1 - tolerance
    if malit_f1 >= best_f1:
        severity = "pass"
        passed = True
        note = "MALIT meets non-regression requirement"
    elif malit_f1 >= threshold:
        severity = "warning"
        passed = True
        note = (
            f"MALIT F1 ({malit_f1:.4f}) slightly below best baseline ({best_f1:.4f}) "
            f"but within tolerance ({tolerance})"
        )
        logger.warning("Non-regression: %s", note)
    else:
        severity = "fail"
        passed = False
        note = (
            f"MALIT F1 ({malit_f1:.4f}) significantly below best baseline "
            f"({best_f1:.4f}) by more than tolerance ({tolerance})"
        )
        logger.error("Non-regression FAILED: %s", note)

    return {
        "passed": passed,
        "severity": severity,
        "malit_f1": malit_f1,
        "best_baseline_name": best_name,
        "best_baseline_f1": best_f1,
        "tolerance": tolerance,
        "threshold": threshold,
        "note": note,
    }


# ── Text generation ───────────────────────────────────────────────────────────

def generate_comparison_text(
    comparison_results: dict[str, dict[str, Any]],
) -> dict[str, str]:
    """
    Generate paper-ready comparison statements from protocol results.

    Returns dict with "comparison_statement" and "statistical_statement" keys.
    """
    superiority_lines = []
    equivalence_lines = []
    inconclusive_lines = []

    for model_name, cmp in comparison_results.items():
        verdict = cmp.get("verdict", "Inconclusive")
        delta = cmp.get("delta_f1", float("nan"))
        p_val = cmp.get("mcnemar", {}).get("p_value", float("nan"))
        delta_str = f"{delta:+.4f}" if not np.isnan(delta) else "N/A"

        if verdict == "Superior":
            p_str = f"p={p_val:.4f}" if not np.isnan(p_val) else "p=N/A"
            superiority_lines.append(
                f"{model_name} (ΔF1={delta_str}, McNemar {p_str})"
            )
        elif verdict == "Equivalent":
            equivalence_lines.append(f"{model_name} (ΔF1={delta_str})")
        else:
            inconclusive_lines.append(model_name)

    # Collect indeterminate TOST cases for explicit reporting
    indeterminate_lines = []
    for model_name, cmp in comparison_results.items():
        if cmp.get("tost", {}).get("indeterminate"):
            indeterminate_lines.append(model_name)

    parts = []
    if superiority_lines:
        parts.append(
            "MALIT V2 demonstrated superior performance over: "
            + "; ".join(superiority_lines) + "."
        )
    if equivalence_lines:
        parts.append(
            "MALIT V2 was statistically equivalent to: "
            + "; ".join(equivalence_lines) + " (TOST margin ±0.5% F1)."
        )
    if indeterminate_lines:
        parts.append(
            "TOST equivalence test was indeterminate (low cross-seed variance) for: "
            + ", ".join(indeterminate_lines)
            + "; no equivalence conclusion drawn."
        )
    if inconclusive_lines:
        parts.append(
            "Results were inconclusive for: "
            + ", ".join(inconclusive_lines) + "."
        )
    comparison_statement = " ".join(parts) if parts else "No comparisons available."

    # Statistical statement
    n_superior = len(superiority_lines)
    n_total = len(comparison_results)
    statistical_statement = (
        f"Statistical significance was assessed using the McNemar test "
        f"(α=0.05, majority vote across seeds; mean p-values reported for reference only) "
        f"for pairwise prediction comparison and the TOST equivalence test "
        f"(margin=±0.5% F1) for practical equivalence. "
        f"MALIT V2 outperformed {n_superior}/{n_total} baseline(s) at p<0.05 "
        f"(majority vote). "
        f"Significance determined via majority vote across seeds; "
        f"mean p-values reported for reference only."
    )

    return {
        "comparison_statement": comparison_statement,
        "statistical_statement": statistical_statement,
    }
