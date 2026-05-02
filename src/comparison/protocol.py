"""
Statistical comparison protocol: MALIT V2 vs each baseline.

For each baseline:
  - McNemar test (per-seed + aggregated majority vote)
  - TOST equivalence (shared-seed F1 scores)
  - ΔF1 and ΔSensitivity effect sizes
  - Verdict: Superior / Equivalent / Inconclusive
  - Error overlap analysis
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.statistics.tests import McNemarTest, TOSTEquivalence
from src.utils.save import _make_serializable

logger = logging.getLogger(__name__)


class ComparisonProtocol:
    """
    Compare MALIT V2 against a set of baseline models.

    Parameters
    ----------
    malit_results : dict with keys:
        seed_preds   : dict[int, list[int]]
        seed_metrics : dict[int, dict[str, Any]]
        seed_probs   : dict[int, array]
        (seed_labels optional here — use the labels argument instead)
    baseline_results : dict[model_name, same structure]
    labels : ground-truth test labels shared by all models
    mcnemar_seed : which seed to use for McNemar (both MALIT and baseline must have it)
    """

    def __init__(
        self,
        malit_results: dict[str, Any],
        baseline_results: dict[str, dict[str, Any]],
        labels: list[int],
        mcnemar_seed: int = 42,
        tost_margin: float = 0.005,
        tost_alpha: float = 0.05,
    ):
        self.malit_results = malit_results
        self.baseline_results = baseline_results
        self.labels = labels
        self.mcnemar_seed = mcnemar_seed
        self.tost_margin = tost_margin
        self.tost_alpha = tost_alpha

    def compare_all(self) -> dict[str, dict[str, Any]]:
        """Run full comparison protocol against every baseline."""
        comparisons: dict[str, dict[str, Any]] = {}
        for model_name, bl_data in self.baseline_results.items():
            logger.info("Comparing MALIT vs %s", model_name)
            comparisons[model_name] = self._compare_one(model_name, bl_data)
        return comparisons

    # ── Private ────────────────────────────────────────────────────────────────

    def _compare_one(
        self, model_name: str, baseline_data: dict[str, Any]
    ) -> dict[str, Any]:
        failure_flags: list[str] = []

        # ── 1. Validate sample counts ─────────────────────────────────────────
        malit_preds_ref = self.malit_results["seed_preds"].get(self.mcnemar_seed, [])
        bl_preds_ref = baseline_data["seed_preds"].get(self.mcnemar_seed, [])

        if len(malit_preds_ref) != len(self.labels):
            failure_flags.append("malit_sample_count_mismatch")
            logger.error(
                "MALIT sample count mismatch: got %d, expected %d",
                len(malit_preds_ref), len(self.labels),
            )
        if len(bl_preds_ref) != len(self.labels):
            failure_flags.append(f"{model_name}_sample_count_mismatch")
            logger.error(
                "%s sample count mismatch: got %d, expected %d",
                model_name, len(bl_preds_ref), len(self.labels),
            )

        # ── 2. McNemar test (single seed-42 + multi-seed) ────────────────────
        if malit_preds_ref and bl_preds_ref and len(malit_preds_ref) == len(bl_preds_ref):
            mcnemar_result = McNemarTest.test(malit_preds_ref, bl_preds_ref, self.labels)
        else:
            failure_flags.append("mcnemar_skipped_due_to_mismatch")
            mcnemar_result = {
                "statistic": float("nan"),
                "p_value": float("nan"),
                "significant": False,
                "contingency_table": [],
                "note": "Skipped — sample count mismatch",
            }

        multi_seed_mcnemar = McNemarTest.multi_seed_mcnemar(
            self.malit_results["seed_preds"],
            baseline_data["seed_preds"],
            self.labels,
        )
        if multi_seed_mcnemar.get("failure"):
            failure_flags.append("multi_seed_mcnemar_no_shared_seeds")

        # ── 3. Delta metrics ──────────────────────────────────────────────────
        malit_f1_mean = float(np.nanmean([
            m.get("f1", float("nan"))
            for m in self.malit_results["seed_metrics"].values()
        ]))
        bl_f1_mean = float(np.nanmean([
            m.get("f1", float("nan"))
            for m in baseline_data["seed_metrics"].values()
        ]))
        delta_f1 = malit_f1_mean - bl_f1_mean

        malit_sens_mean = float(np.nanmean([
            m.get("sensitivity", float("nan"))
            for m in self.malit_results["seed_metrics"].values()
        ]))
        bl_sens_mean = float(np.nanmean([
            m.get("sensitivity", float("nan"))
            for m in baseline_data["seed_metrics"].values()
        ]))
        delta_sensitivity = malit_sens_mean - bl_sens_mean

        if np.isnan(delta_f1):
            failure_flags.append("nan_delta_f1")

        # ── 4. TOST on shared seeds ───────────────────────────────────────────
        shared_seeds = sorted(
            set(self.malit_results["seed_metrics"].keys())
            & set(baseline_data["seed_metrics"].keys())
        )
        if not shared_seeds:
            failure_flags.append("no_shared_seeds")
            tost_result: dict[str, Any] = {"equivalent": False, "note": "No shared seeds"}
        else:
            malit_f1s = [
                self.malit_results["seed_metrics"][s].get("f1", float("nan"))
                for s in shared_seeds
            ]
            bl_f1s = [
                baseline_data["seed_metrics"][s].get("f1", float("nan"))
                for s in shared_seeds
            ]
            if any(np.isnan(v) for v in malit_f1s + bl_f1s):
                failure_flags.append("nan_in_tost_scores")
                tost_result = {"equivalent": False, "note": "NaN F1 scores in TOST input"}
            else:
                tost_result = TOSTEquivalence.test(
                    malit_f1s, bl_f1s,
                    margin=self.tost_margin, alpha=self.tost_alpha,
                )

        # ── 5. Effect size (Cohen's d-like) ──────────────────────────────────
        all_f1s = (
            [m.get("f1", float("nan")) for m in self.malit_results["seed_metrics"].values()]
            + [m.get("f1", float("nan")) for m in baseline_data["seed_metrics"].values()]
        )
        pooled_std = (
            float(np.nanstd(all_f1s, ddof=1)) if len(all_f1s) >= 2 else 0.0
        )
        effect_size = delta_f1 / pooled_std if pooled_std > 1e-12 else float("nan")

        # ── 6. Error overlap ──────────────────────────────────────────────────
        error_overlap = self._error_overlap(malit_preds_ref, bl_preds_ref)

        # ── 7. Verdict ────────────────────────────────────────────────────────
        mcnemar_p = mcnemar_result.get("p_value", float("nan"))
        majority_sig = multi_seed_mcnemar.get("aggregated", {}).get(
            "majority_significant", False
        )
        tost_equivalent = (
            tost_result.get("equivalent", False)
            and not tost_result.get("indeterminate", False)
        )
        if failure_flags and "multi_seed_mcnemar_no_shared_seeds" not in failure_flags:
            verdict = "Inconclusive"
        elif majority_sig and delta_f1 > 0:
            verdict = "Superior"
        elif (
            not majority_sig
            and not np.isnan(mcnemar_p)
            and mcnemar_p < 0.05
            and delta_f1 > 0
        ):
            verdict = "Superior"
        elif tost_equivalent:
            verdict = "Equivalent"
        else:
            verdict = "Inconclusive"

        if failure_flags:
            for flag in failure_flags:
                logger.warning("COMPARISON FLAG [%s vs %s]: %s", "MALIT", model_name, flag)

        return {
            "mcnemar":             mcnemar_result,
            "multi_seed_mcnemar":  multi_seed_mcnemar,
            "delta_f1":            float(delta_f1),
            "delta_sensitivity":   float(delta_sensitivity),
            "malit_f1_mean":       float(malit_f1_mean),
            "baseline_f1_mean":    float(bl_f1_mean),
            "tost":                tost_result,
            "effect_size":         float(effect_size) if not np.isnan(effect_size) else None,
            "verdict":             verdict,
            "failure_flags":       failure_flags,
            "error_overlap":       error_overlap,
            "shared_seeds":        shared_seeds,
        }

    def save_reproducibility(
        self,
        results: dict[str, dict[str, Any]],
        output_dir: str | Path,
    ) -> None:
        """Save per-seed McNemar, aggregated McNemar, and TOST results to JSON."""
        out = Path(output_dir) / "results" / "statistics"
        out.mkdir(parents=True, exist_ok=True)

        mcnemar_per_seed: dict[str, Any] = {}
        mcnemar_aggregated: dict[str, Any] = {}
        tost_results: dict[str, Any] = {}

        for model_name, cmp in results.items():
            ms = cmp.get("multi_seed_mcnemar", {})
            mcnemar_per_seed[model_name] = ms.get("per_seed", {})
            mcnemar_aggregated[model_name] = ms.get("aggregated", {})
            tost_results[model_name] = cmp.get("tost", {})

        mcnemar_summary: dict[str, Any] = {}
        for model_name, agg in mcnemar_aggregated.items():
            mcnemar_summary[model_name] = {
                **agg,
                "significance_decision": "Significant" if agg.get("majority_significant") else "Not Significant",
                "decision_method": "majority_vote",
                "note": "Significance determined via majority vote across seeds; mean p-values reported for reference only.",
            }

        for fname, data in [
            ("mcnemar_per_seed.json", mcnemar_per_seed),
            ("mcnemar_aggregated.json", mcnemar_aggregated),
            ("mcnemar_summary.json", mcnemar_summary),
            ("tost_results.json", tost_results),
        ]:
            with open(out / fname, "w") as f:
                json.dump(_make_serializable(data), f, indent=2)

        logger.info("Saved reproducibility files to %s", out)

    def _error_overlap(
        self, malit_preds: list[int], bl_preds: list[int]
    ) -> dict[str, int]:
        """Count the four error-agreement quadrants between MALIT and baseline."""
        n = len(self.labels)
        if (not malit_preds or not bl_preds
                or len(malit_preds) != n or len(bl_preds) != n):
            return {"both_correct": 0, "malit_correct_bl_wrong": 0,
                    "malit_wrong_bl_correct": 0, "both_wrong": 0}
        labels = np.array(self.labels)
        m = np.array(malit_preds) == labels
        b = np.array(bl_preds) == labels
        return {
            "both_correct":           int((m & b).sum()),
            "malit_correct_bl_wrong": int((m & ~b).sum()),
            "malit_wrong_bl_correct": int((~m & b).sum()),
            "both_wrong":             int((~m & ~b).sum()),
        }
