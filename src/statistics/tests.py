from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar

logger = logging.getLogger(__name__)


# ── Numerical helpers ─────────────────────────────────────────────────────────

def _safe_std(arr: np.ndarray) -> float:
    """Population std with ddof=1; returns 0.0 for arrays shorter than 2."""
    if len(arr) < 2:
        return 0.0
    result = float(np.std(arr, ddof=1))
    return result if np.isfinite(result) else 0.0


def _safe_ci(mean: float, std: float, n: int) -> tuple[float, float]:
    """95% CI; collapses to (mean, mean) when std=0 or n<2 (single-seed run)."""
    if n < 2 or std == 0:
        return mean, mean
    margin = 1.96 * std / np.sqrt(n)
    return mean - margin, mean + margin


# ── McNemar Test ─────────────────────────────────────────────────────────────

class McNemarTest:
    """
    McNemar test for comparing two classifiers on the same test set.

    Null hypothesis: the two classifiers have the same error rate.
    """

    @staticmethod
    def contingency_table(
        preds_a: list[int], preds_b: list[int], labels: list[int]
    ) -> np.ndarray:
        """
        Returns 2×2 table:
            [[both correct, a correct only],
             [b correct only, both wrong]]
        """
        a_correct = np.array(preds_a) == np.array(labels)
        b_correct = np.array(preds_b) == np.array(labels)
        n00 = int((~a_correct & ~b_correct).sum())  # both wrong
        n01 = int((~a_correct & b_correct).sum())   # only b correct
        n10 = int((a_correct & ~b_correct).sum())   # only a correct
        n11 = int((a_correct & b_correct).sum())    # both correct
        return np.array([[n11, n10], [n01, n00]])

    @staticmethod
    def test(
        preds_a: list[int], preds_b: list[int], labels: list[int]
    ) -> dict[str, Any]:
        table = McNemarTest.contingency_table(preds_a, preds_b, labels)
        n01, n10 = int(table[1, 0]), int(table[0, 1])
        # Degenerate case: both classifiers agree on every sample — no test possible
        if n01 + n10 == 0:
            return {
                "statistic": 0.0,
                "p_value": 1.0,
                "significant": False,
                "contingency_table": table.tolist(),
            }
        result = sm_mcnemar(table, exact=False, correction=True)
        return {
            "statistic": float(result.statistic),
            "p_value": float(result.pvalue),
            "significant": bool(result.pvalue < 0.05),
            "contingency_table": table.tolist(),
        }

    @staticmethod
    def multi_seed_mcnemar(
        preds_a: dict[int, list[int]],
        preds_b: dict[int, list[int]],
        labels: list[int],
    ) -> dict[str, Any]:
        """
        Run McNemar per shared seed and aggregate via mean p-value + majority vote.
        """
        shared_seeds = sorted(set(preds_a.keys()) & set(preds_b.keys()))
        if not shared_seeds:
            return {
                "per_seed": {},
                "aggregated": {
                    "mean_p_value": float("nan"),
                    "majority_significant": False,
                    "n_significant": 0,
                    "n_seeds": 0,
                },
                "failure": "No shared seeds",
            }

        per_seed: dict[int, dict[str, Any]] = {}
        p_values: list[float] = []
        n_significant = 0
        n_labels = len(labels)

        for seed in shared_seeds:
            pa, pb = preds_a[seed], preds_b[seed]
            if len(pa) != n_labels or len(pb) != n_labels:
                per_seed[seed] = {
                    "statistic": float("nan"), "p_value": float("nan"),
                    "significant": False,
                    "note": f"Skipped — length mismatch (a={len(pa)}, b={len(pb)}, labels={n_labels})",
                }
                continue
            result = McNemarTest.test(pa, pb, labels)
            per_seed[seed] = result
            p_values.append(result["p_value"])
            if result["significant"]:
                n_significant += 1

        n_valid = len(p_values)
        mean_p = float(np.mean(p_values)) if p_values else float("nan")
        majority_significant = n_significant > n_valid / 2 if n_valid > 0 else False

        return {
            "per_seed": per_seed,
            "aggregated": {
                "mean_p_value": mean_p,
                "majority_significant": majority_significant,
                "n_significant": n_significant,
                "n_seeds": n_valid,
            },
        }

    @staticmethod
    def per_seed_mcnemar_table(
        seed_preds: dict[int, list[int]],
        labels: list[int],
    ) -> pd.DataFrame:
        """Compare all seed pairs and return a DataFrame of results."""
        seeds = list(seed_preds.keys())
        rows = []
        for i, s1 in enumerate(seeds):
            for s2 in seeds[i + 1 :]:
                res = McNemarTest.test(seed_preds[s1], seed_preds[s2], labels)
                rows.append({
                    "seed_a": s1,
                    "seed_b": s2,
                    "statistic": res["statistic"],
                    "p_value": res["p_value"],
                    "significant": res["significant"],
                })
        return pd.DataFrame(rows)


# ── TOST Equivalence Test ─────────────────────────────────────────────────────

class TOSTEquivalence:
    """
    Two One-Sided Tests (TOST) for equivalence.

    Tests whether the difference in F1 between two models is within ±margin.
    """

    @staticmethod
    def test(
        scores_a: list[float],
        scores_b: list[float],
        margin: float = 0.005,
        alpha: float = 0.05,
    ) -> dict[str, Any]:
        """
        Tests H0: |μ_a - μ_b| > margin   (not equivalent)
        vs   Ha: |μ_a - μ_b| ≤ margin   (equivalent)

        Rejects H0 (concludes equivalence) if both one-sided p-values < alpha.
        """
        a = np.array(scores_a, dtype=float)
        b = np.array(scores_b, dtype=float)
        diff = a - b

        n = len(diff)
        mean_diff = float(diff.mean())
        se = _safe_std(diff) / np.sqrt(n) if n >= 2 else 0.0

        # Zero variance: SE cannot be computed — statistical test not applicable.
        if se < 1e-12:
            logger.info(
                "TOST: zero variance (se≈0) — mean_diff=%.4f, margin=%.4f → indeterminate.",
                mean_diff, margin,
            )
            return {
                "mean_diff": mean_diff,
                "se": 0.0,
                "margin": margin,
                "t_low": None,
                "t_high": None,
                "p_low": None,
                "p_high": None,
                "equivalent": False,
                "indeterminate": True,
                "alpha": alpha,
                "note": "Indeterminate (zero variance): statistical test not applicable.",
            }

        # Lower one-sided: H0: μ_diff ≤ -margin
        t_low = (mean_diff + margin) / se
        p_low = float(stats.t.sf(t_low, df=n - 1))

        # Upper one-sided: H0: μ_diff ≥ +margin
        t_high = (mean_diff - margin) / se
        p_high = float(stats.t.cdf(t_high, df=n - 1))

        equivalent = (p_low < alpha) and (p_high < alpha)
        return {
            "mean_diff": mean_diff,
            "se": se,
            "margin": margin,
            "t_low": t_low,
            "t_high": t_high,
            "p_low": p_low,
            "p_high": p_high,
            "equivalent": equivalent,
            "alpha": alpha,
        }

    @staticmethod
    def tost_equivalence(
        seed_f1_scores: dict[int, float],
        baseline_f1: float | None = None,
        margin: float = 0.005,
        alpha: float = 0.05,
    ) -> dict[str, Any]:
        """
        Test equivalence of all seed F1 scores against each other or a baseline.
        """
        scores = list(seed_f1_scores.values())
        if baseline_f1 is not None:
            baseline = [baseline_f1] * len(scores)
            result = TOSTEquivalence.test(scores, baseline, margin, alpha)
        else:
            mean_score = np.mean(scores)
            baseline = [mean_score] * len(scores)
            result = TOSTEquivalence.test(scores, baseline, margin, alpha)
        result["seed_f1_scores"] = seed_f1_scores
        return result


# ── Full Report ───────────────────────────────────────────────────────────────

class StatisticsReport:
    """Aggregates McNemar + TOST results into a full reproducibility report."""

    def __init__(
        self,
        seed_preds: dict[int, list[int]],
        seed_metrics: dict[int, dict[str, Any]],
        labels: list[int],
        margin: float = 0.005,
        alpha: float = 0.05,
    ):
        self.seed_preds = seed_preds
        self.seed_metrics = seed_metrics
        self.labels = labels
        self.margin = margin
        self.alpha = alpha

    def full_report(self) -> dict[str, Any]:
        seed_f1 = {s: m.get("f1", 0.0) for s, m in self.seed_metrics.items()}

        mcnemar_df = McNemarTest.per_seed_mcnemar_table(self.seed_preds, self.labels)
        tost_result = TOSTEquivalence.tost_equivalence(seed_f1, margin=self.margin, alpha=self.alpha)

        f1_vals = list(seed_f1.values())
        f1_arr = np.array(f1_vals)
        mean_f1 = float(np.mean(f1_arr))
        std_f1 = _safe_std(f1_arr)
        ci_low, ci_high = _safe_ci(mean_f1, std_f1, len(f1_vals))
        report = {
            "summary": {
                "n_seeds": len(self.seed_preds),
                "seeds": list(self.seed_preds.keys()),
                "mean_f1": mean_f1,
                "std_f1": std_f1,
                "min_f1": float(np.min(f1_arr)),
                "max_f1": float(np.max(f1_arr)),
                "ci_low_f1": ci_low,
                "ci_high_f1": ci_high,
            },
            "mcnemar": mcnemar_df.to_dict(orient="records"),
            "tost": tost_result,
            "per_seed_metrics": self.seed_metrics,
        }
        return report
