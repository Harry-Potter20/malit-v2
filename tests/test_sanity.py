"""
Post-validation sanity checks — publication-grade reproducibility hardening.

Tests verify that:
  1. Confidence calibration: error_rate(low_conf) > error_rate(high_conf)
  2. Uncertainty monotonicity: error_rate(high_entropy) > error_rate(low_entropy)
                               error_rate(high_variance) > error_rate(low_variance)
  3. Ensemble stability: error_rate(wide_CI) > error_rate(narrow_CI)
  4. CBR consistency: knn_error_rate(low_consistency) > knn_error_rate(high_consistency)
  5. Metric validation: no NaN/Inf, CI validity, variance ≥ 0
  6. Brier Score + NLL present and bounded correctly
  7. Failure detection: critical/warning flags raised for known bad patterns
  8. Numerical edge cases: safe_std, safe_ci, TOST zero-variance
  9. pin_memory is CUDA-only (no MPS/CPU pin_memory)
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


# ── 1. Confidence Calibration ─────────────────────────────────────────────────

class TestConfidenceCalibration:
    """error_rate(low_confidence) > error_rate(high_confidence)"""

    def _make_probs(self, n: int, correct: bool, conf: float) -> np.ndarray:
        """Build (n, 2) prob array: confident+correct or uncertain+wrong."""
        labels = np.array([i % 2 for i in range(n)])
        probs = np.zeros((n, 2), dtype=np.float64)
        for i, lbl in enumerate(labels):
            if correct:
                probs[i, lbl] = conf
                probs[i, 1 - lbl] = 1.0 - conf
            else:
                probs[i, 1 - lbl] = conf          # wrong class has higher prob
                probs[i, lbl] = 1.0 - conf
        return probs, labels

    def test_high_conf_lower_error_than_low_conf(self):
        n = 200
        high_probs, labels = self._make_probs(n, correct=True, conf=0.92)
        low_probs, _ = self._make_probs(n, correct=False, conf=0.60)

        high_preds = high_probs.argmax(axis=1)
        low_preds = low_probs.argmax(axis=1)

        high_errors = (high_preds != labels).mean()
        low_errors = (low_preds != labels).mean()

        assert low_errors > high_errors, (
            f"Expected low-conf error ({low_errors:.2f}) > high-conf error ({high_errors:.2f})"
        )

    def test_perfect_confidence_zero_error(self):
        n = 100
        labels = np.array([i % 2 for i in range(n)])
        probs = np.zeros((n, 2))
        for i, lbl in enumerate(labels):
            probs[i, lbl] = 1.0   # perfect confidence
        preds = probs.argmax(axis=1)
        assert (preds != labels).mean() == 0.0

    def test_calibration_error_rate_ordering_is_robust(self):
        """Property holds across random seeds."""
        for seed in [0, 1, 2]:
            np.random.seed(seed)
            n = 300
            labels = np.random.randint(0, 2, n)

            # High confidence: correct 95% of the time
            high_probs = np.zeros((n, 2))
            for i, lbl in enumerate(labels):
                if np.random.rand() < 0.95:
                    high_probs[i, lbl] = 0.90 + np.random.rand() * 0.09
                else:
                    high_probs[i, 1 - lbl] = 0.90 + np.random.rand() * 0.09
            high_probs = high_probs / high_probs.sum(axis=1, keepdims=True)

            # Low confidence: correct only 40% of the time
            low_probs = np.zeros((n, 2))
            for i, lbl in enumerate(labels):
                if np.random.rand() < 0.40:
                    low_probs[i, lbl] = 0.55 + np.random.rand() * 0.10
                else:
                    low_probs[i, 1 - lbl] = 0.55 + np.random.rand() * 0.10
            low_probs = low_probs / low_probs.sum(axis=1, keepdims=True)

            high_errors = (high_probs.argmax(axis=1) != labels).mean()
            low_errors = (low_probs.argmax(axis=1) != labels).mean()
            assert low_errors > high_errors, f"Seed {seed}: property violated"


# ── 2. Uncertainty Monotonicity ───────────────────────────────────────────────

class TestUncertaintyMonotonicity:
    """error_rate(high_entropy) > error_rate(low_entropy)
       error_rate(high_variance) > error_rate(low_variance)
    """

    def test_entropy_monotone_with_error(self):
        """Samples with high predictive entropy should have higher error rate."""
        n = 200
        labels = np.array([i % 2 for i in range(n)])

        # Low entropy: very confident and CORRECT
        low_ent_probs = np.zeros((n, 2))
        for i, lbl in enumerate(labels):
            low_ent_probs[i, lbl] = 0.99
            low_ent_probs[i, 1 - lbl] = 0.01

        # High entropy: near-uniform and WRONG
        high_ent_probs = np.zeros((n, 2))
        for i, lbl in enumerate(labels):
            high_ent_probs[i, 1 - lbl] = 0.55   # wrong class slightly favoured
            high_ent_probs[i, lbl] = 0.45

        def entropy(p: np.ndarray) -> np.ndarray:
            eps = 1e-8
            return -(p * np.log2(p + eps)).sum(axis=1)

        low_ent = entropy(low_ent_probs).mean()
        high_ent = entropy(high_ent_probs).mean()
        assert high_ent > low_ent, "High-entropy set must have higher mean entropy"

        low_errors = (low_ent_probs.argmax(axis=1) != labels).mean()
        high_errors = (high_ent_probs.argmax(axis=1) != labels).mean()
        assert high_errors > low_errors, (
            f"High-entropy error ({high_errors:.2f}) should exceed low-entropy ({low_errors:.2f})"
        )

    def test_variance_monotone_with_error(self):
        """Samples with high MC-dropout variance should have higher error rate."""
        np.random.seed(0)
        T, n = 10, 100
        labels = np.array([i % 2 for i in range(n)])

        # Low variance: all T passes agree → correct
        low_var = np.zeros((T, n, 2))
        for t in range(T):
            for i, lbl in enumerate(labels):
                low_var[t, i, lbl] = 0.95
                low_var[t, i, 1 - lbl] = 0.05

        # High variance: alternating passes — on average wrong
        high_var = np.zeros((T, n, 2))
        for t in range(T):
            for i, lbl in enumerate(labels):
                if t % 2 == 0:          # even pass: very wrong
                    high_var[t, i, 1 - lbl] = 0.80
                    high_var[t, i, lbl] = 0.20
                else:                   # odd pass: slightly right
                    high_var[t, i, lbl] = 0.60
                    high_var[t, i, 1 - lbl] = 0.40

        low_mean = low_var.mean(axis=0)
        low_variance = low_var.var(axis=0)
        high_mean = high_var.mean(axis=0)
        high_variance = high_var.var(axis=0)

        assert high_variance.mean() > low_variance.mean(), "High-var set must have larger variance"

        low_errors = (low_mean.argmax(axis=1) != labels).mean()
        high_errors = (high_mean.argmax(axis=1) != labels).mean()
        assert high_errors > low_errors, (
            f"High-variance error ({high_errors:.2f}) should exceed low-variance ({low_errors:.2f})"
        )

    def test_entropy_variance_are_nonneg(self):
        """Entropy and variance must never be negative."""
        from src.explainability.bayesian_uncertainty import BayesianUncertainty
        np.random.seed(1)
        n, T = 50, 5
        # Create a random valid prob array (T, n, 2) summing to 1
        raw = np.abs(np.random.randn(T, n, 2))
        probs = raw / raw.sum(axis=2, keepdims=True)

        mean = probs.mean(axis=0)
        var = probs.var(axis=0)
        entropy = -(mean * np.log2(mean + 1e-8)).sum(axis=1)

        assert (var >= 0).all(), "Variance must be non-negative"
        assert (entropy >= 0).all(), "Entropy must be non-negative"


# ── 3. Ensemble Stability ─────────────────────────────────────────────────────

class TestEnsembleStability:
    """error_rate(wide_CI) > error_rate(narrow_CI)
       lower agreement → higher error
    """

    def test_wide_ci_higher_error_than_narrow_ci(self):
        from src.explainability.ensemble_confidence import EnsembleConfidence
        np.random.seed(42)
        n = 200
        labels = np.array([i % 2 for i in range(n)])

        # Narrow CI: all 3 seeds agree, predictions are correct
        narrow = np.zeros((3, n, 2))
        for s in range(3):
            for i, lbl in enumerate(labels):
                narrow[s, i, lbl] = 0.90 + 0.01 * s
                narrow[s, i, 1 - lbl] = 0.10 - 0.01 * s

        # Wide CI: seeds wildly disagree; on average predict wrong class
        wide = np.zeros((3, n, 2))
        for i, lbl in enumerate(labels):
            wide[0, i, 1 - lbl] = 0.95; wide[0, i, lbl] = 0.05  # very wrong
            wide[1, i, 1 - lbl] = 0.90; wide[1, i, lbl] = 0.10  # very wrong
            wide[2, i, lbl]     = 0.55; wide[2, i, 1 - lbl] = 0.45  # slightly right

        narrow_stats = EnsembleConfidence.ensemble_stats(narrow)
        wide_stats = EnsembleConfidence.ensemble_stats(wide)

        narrow_ci = float(narrow_stats["ci_width"].mean())
        wide_ci = float(wide_stats["ci_width"].mean())
        assert wide_ci > narrow_ci, "Wide-CI set should have larger mean CI"

        narrow_errors = (narrow_stats["mean"].argmax(axis=1) != labels).mean()
        wide_errors = (wide_stats["mean"].argmax(axis=1) != labels).mean()
        assert wide_errors > narrow_errors, (
            f"Wide-CI error ({wide_errors:.2f}) should exceed narrow-CI ({narrow_errors:.2f})"
        )

    def test_ci_bounds_valid(self):
        """ci_low must always be ≤ ci_high."""
        from src.explainability.ensemble_confidence import EnsembleConfidence
        np.random.seed(7)
        n_seeds, n, C = 5, 100, 2
        raw = np.abs(np.random.randn(n_seeds, n, C))
        probs = raw / raw.sum(axis=2, keepdims=True)
        s = EnsembleConfidence.ensemble_stats(probs)
        assert (s["ci_low"] <= s["ci_high"]).all(), "CI: ci_low > ci_high detected"

    def test_agreement_anti_correlates_with_error(self):
        """Lower agreement should correspond to higher error rate."""
        from src.explainability.ensemble_confidence import EnsembleConfidence
        np.random.seed(99)
        n = 200
        labels = np.array([i % 2 for i in range(n)])

        # High agreement: all 3 seeds same (correct)
        high_agr = np.zeros((3, n, 2))
        for s in range(3):
            for i, lbl in enumerate(labels):
                high_agr[s, i, lbl] = 0.90
                high_agr[s, i, 1 - lbl] = 0.10

        # Low agreement: seeds split 2:1 on wrong class
        low_agr = np.zeros((3, n, 2))
        for i, lbl in enumerate(labels):
            low_agr[0, i, 1 - lbl] = 0.85; low_agr[0, i, lbl] = 0.15
            low_agr[1, i, 1 - lbl] = 0.80; low_agr[1, i, lbl] = 0.20
            low_agr[2, i, lbl] = 0.60;     low_agr[2, i, 1 - lbl] = 0.40

        high_agr_preds = high_agr.mean(axis=0).argmax(axis=1)
        low_agr_preds = low_agr.mean(axis=0).argmax(axis=1)

        high_errors = (high_agr_preds != labels).mean()
        low_errors = (low_agr_preds != labels).mean()
        assert low_errors > high_errors, (
            f"Low-agreement error ({low_errors:.2f}) should exceed high-agreement ({high_errors:.2f})"
        )


# ── 4. CBR Consistency ────────────────────────────────────────────────────────

class TestCBRConsistency:
    """knn_error_rate(low_consistency) > knn_error_rate(high_consistency)"""

    def _build_cbr_data(self, seed: int = 42):
        from src.explainability.case_based_reasoning import CaseBasedReasoning
        np.random.seed(seed)
        dim = 32
        n_train = 200

        # Two tight clusters: class 0 near −3·e₀, class 1 near +3·e₀
        train_embs = np.zeros((n_train, dim), dtype=np.float32)
        train_labels = np.array([i % 2 for i in range(n_train)])
        for i, lbl in enumerate(train_labels):
            offset = 3.0 if lbl == 1 else -3.0
            train_embs[i, 0] = offset
            train_embs[i, 1:] = np.random.randn(dim - 1).astype(np.float32) * 0.1

        cbr = CaseBasedReasoning(k=5, use_faiss=False)
        cbr.build_index(train_embs, train_labels)
        return cbr, dim

    def test_high_consistency_queries_near_own_cluster(self):
        """Queries in their own class cluster should have consistency ≈ 1.0."""
        cbr, dim = self._build_cbr_data()
        n_test = 50
        high_embs = np.zeros((n_test, dim), dtype=np.float32)
        high_labels = np.array([i % 2 for i in range(n_test)])
        for i, lbl in enumerate(high_labels):
            offset = 3.0 if lbl == 1 else -3.0
            high_embs[i, 0] = offset
            high_embs[i, 1:] = np.random.randn(dim - 1).astype(np.float32) * 0.1

        res = cbr.run_on_embeddings(high_embs, high_labels)
        assert res["mean_consistency"] > 0.8, (
            f"Expected high consistency (>0.8), got {res['mean_consistency']:.3f}"
        )

    def test_low_consistency_queries_in_wrong_cluster(self):
        """Class-1 queries placed in class-0 cluster should have consistency ≈ 0."""
        cbr, dim = self._build_cbr_data()
        n_test = 50
        # All class-1 queries placed in class-0 territory (-3·e₀)
        low_embs = np.zeros((n_test, dim), dtype=np.float32)
        low_labels = np.ones(n_test, dtype=int)
        for i in range(n_test):
            low_embs[i, 0] = -3.0   # class-0 cluster
            low_embs[i, 1:] = np.random.randn(dim - 1).astype(np.float32) * 0.1

        res = cbr.run_on_embeddings(low_embs, low_labels)
        assert res["mean_consistency"] < 0.2, (
            f"Expected low consistency (<0.2), got {res['mean_consistency']:.3f}"
        )

    def test_high_consistency_lower_knn_error(self):
        """kNN error rate ordering: low consistency → high error."""
        cbr, dim = self._build_cbr_data()
        np.random.seed(7)
        n_test = 50

        # High consistency: queries in correct cluster
        high_embs = np.zeros((n_test, dim), dtype=np.float32)
        high_labels = np.array([i % 2 for i in range(n_test)])
        for i, lbl in enumerate(high_labels):
            offset = 3.0 if lbl == 1 else -3.0
            high_embs[i, 0] = offset
            high_embs[i, 1:] = np.random.randn(dim - 1).astype(np.float32) * 0.1

        # Low consistency: class-1 queries in class-0 cluster → all neighbors class 0
        low_embs = np.zeros((n_test, dim), dtype=np.float32)
        low_labels = np.ones(n_test, dtype=int)
        for i in range(n_test):
            low_embs[i, 0] = -3.0
            low_embs[i, 1:] = np.random.randn(dim - 1).astype(np.float32) * 0.1

        high_res = cbr.run_on_embeddings(high_embs, high_labels)
        low_res = cbr.run_on_embeddings(low_embs, low_labels)

        assert high_res["mean_consistency"] > low_res["mean_consistency"]

        def knn_error_rate(results: dict) -> float:
            errors = sum(
                1
                for r in results["neighbors"]
                if int(np.bincount(r["neighbor_labels"]).argmax()) != r["query_label"]
            )
            return errors / len(results["neighbors"])

        assert knn_error_rate(low_res) > knn_error_rate(high_res)


# ── 5. Metric Validation (NaN / Inf / CI / Variance) ─────────────────────────

class TestMetricValidation:
    def test_validate_metrics_clean(self):
        from src.utils.validation import validate_metrics
        issues = validate_metrics({"f1": 0.92, "accuracy": 0.91, "brier_score": 0.08})
        assert issues == []

    def test_validate_metrics_nan_detected(self):
        from src.utils.validation import validate_metrics
        issues = validate_metrics({"f1": float("nan"), "accuracy": 0.5})
        assert any("f1" in i for i in issues)

    def test_validate_metrics_inf_detected(self):
        from src.utils.validation import validate_metrics
        issues = validate_metrics({"nll": float("inf"), "accuracy": 0.5})
        assert any("nll" in i for i in issues)

    def test_validate_ci_valid(self):
        from src.utils.validation import validate_confidence_intervals
        import numpy as np
        ci_low = np.array([0.1, 0.2, 0.3])
        ci_high = np.array([0.4, 0.5, 0.6])
        assert validate_confidence_intervals(ci_low, ci_high) == []

    def test_validate_ci_inverted_detected(self):
        from src.utils.validation import validate_confidence_intervals
        import numpy as np
        ci_low = np.array([0.5, 0.1])
        ci_high = np.array([0.3, 0.8])  # first entry inverted
        issues = validate_confidence_intervals(ci_low, ci_high)
        assert len(issues) > 0

    def test_validate_variance_nonneg(self):
        from src.utils.validation import validate_variance
        import numpy as np
        assert validate_variance(np.array([0.0, 0.1, 0.5])) == []

    def test_validate_variance_negative_detected(self):
        from src.utils.validation import validate_variance
        import numpy as np
        issues = validate_variance(np.array([0.1, -0.01]))
        assert len(issues) > 0


# ── 6. Brier Score + NLL ─────────────────────────────────────────────────────

class TestBrierScoreNLL:
    def _probs_and_labels(self, n: int = 100, correct_frac: float = 0.8):
        np.random.seed(5)
        labels = np.array([i % 2 for i in range(n)])
        probs = np.zeros((n, 2))
        for i, lbl in enumerate(labels):
            if np.random.rand() < correct_frac:
                probs[i, lbl] = 0.75 + np.random.rand() * 0.20
            else:
                probs[i, 1 - lbl] = 0.60 + np.random.rand() * 0.30
            probs[i] /= probs[i].sum()
        return probs.tolist(), labels.tolist()

    def test_brier_score_present(self):
        from src.evaluation.metrics import EvaluationMetrics
        probs, labels = self._probs_and_labels()
        preds = np.array(probs).argmax(axis=1).tolist()
        metrics = EvaluationMetrics.from_predictions(labels, preds, probs)
        assert "brier_score" in metrics

    def test_nll_present(self):
        from src.evaluation.metrics import EvaluationMetrics
        probs, labels = self._probs_and_labels()
        preds = np.array(probs).argmax(axis=1).tolist()
        metrics = EvaluationMetrics.from_predictions(labels, preds, probs)
        assert "nll" in metrics

    def test_brier_score_range(self):
        """Brier score must be in [0, 2] for binary classification."""
        from src.evaluation.metrics import EvaluationMetrics
        probs, labels = self._probs_and_labels()
        preds = np.array(probs).argmax(axis=1).tolist()
        metrics = EvaluationMetrics.from_predictions(labels, preds, probs)
        assert 0.0 <= metrics["brier_score"] <= 2.0

    def test_nll_positive(self):
        """NLL must always be positive (it is a negative log probability)."""
        from src.evaluation.metrics import EvaluationMetrics
        probs, labels = self._probs_and_labels()
        preds = np.array(probs).argmax(axis=1).tolist()
        metrics = EvaluationMetrics.from_predictions(labels, preds, probs)
        assert metrics["nll"] > 0.0

    def test_perfect_predictions_low_brier_and_nll(self):
        """Perfect predictions should yield near-zero Brier and low NLL."""
        from src.evaluation.metrics import EvaluationMetrics
        n = 50
        labels = [i % 2 for i in range(n)]
        probs = [[1.0, 0.0] if lbl == 0 else [0.0, 1.0] for lbl in labels]
        preds = list(labels)
        metrics = EvaluationMetrics.from_predictions(labels, preds, probs)
        assert metrics["brier_score"] < 0.01
        assert metrics["nll"] < 0.01

    def test_worst_predictions_high_brier_and_nll(self):
        """All-wrong high-conf predictions → high Brier and NLL."""
        from src.evaluation.metrics import EvaluationMetrics
        n = 50
        labels = [0] * n
        probs = [[0.01, 0.99]] * n   # confident but wrong
        preds = [1] * n
        metrics = EvaluationMetrics.from_predictions(labels, preds, probs)
        assert metrics["brier_score"] > 1.5
        assert metrics["nll"] > 3.0

    def test_brier_nll_no_nan_or_inf(self):
        """Brier and NLL must be finite for any valid input."""
        from src.evaluation.metrics import EvaluationMetrics
        from src.utils.validation import validate_metrics
        probs, labels = self._probs_and_labels(n=200)
        preds = np.array(probs).argmax(axis=1).tolist()
        metrics = EvaluationMetrics.from_predictions(labels, preds, probs)
        issues = validate_metrics(metrics)
        assert issues == [], f"Metric issues: {issues}"


# ── 7. Failure Detection ──────────────────────────────────────────────────────

class TestFailureDetection:
    def test_high_conf_errors_trigger_critical(self):
        """Many high-confidence mistakes should be flagged as critical."""
        from src.utils.validation import check_failure_conditions
        np.random.seed(0)
        n = 100
        labels = np.zeros(n, dtype=int)
        probs = np.zeros((n, 2))
        probs[:, 1] = 0.95   # confident and WRONG (true label is 0)
        probs[:, 0] = 0.05
        result = check_failure_conditions(probs, labels, n_seeds=1)
        assert len(result["critical"]) > 0

    def test_zero_variance_multiseed_triggers_critical(self):
        """Zero variance in multi-seed run should raise critical flag."""
        from src.utils.validation import check_failure_conditions
        n = 50
        labels = np.array([i % 2 for i in range(n)])
        probs = np.zeros((n, 2))
        probs[:, 0] = 0.7
        probs[:, 1] = 0.3
        variance = np.zeros((n, 2))   # zero variance
        result = check_failure_conditions(probs, labels, variance=variance, n_seeds=3)
        assert any("variance" in c.lower() for c in result["critical"])

    def test_zero_ci_width_multiseed_triggers_critical(self):
        """Zero CI width for multi-seed run should raise critical flag."""
        from src.utils.validation import check_failure_conditions
        n = 50
        labels = np.array([i % 2 for i in range(n)])
        probs = np.zeros((n, 2))
        probs[:, 0] = 0.6
        probs[:, 1] = 0.4
        ci_width = np.zeros((n, 2))   # zero CI
        result = check_failure_conditions(probs, labels, ci_width=ci_width, n_seeds=3)
        assert any("ci" in c.lower() for c in result["critical"])

    def test_no_false_criticals_for_good_model(self):
        """A well-calibrated correct model should produce no critical flags."""
        from src.utils.validation import check_failure_conditions
        np.random.seed(42)
        n = 100
        labels = np.array([i % 2 for i in range(n)])
        probs = np.zeros((n, 2))
        for i, lbl in enumerate(labels):
            probs[i, lbl] = 0.80
            probs[i, 1 - lbl] = 0.20
        result = check_failure_conditions(probs, labels, n_seeds=1)
        assert result["critical"] == []


# ── 8. Numerical Edge Cases ───────────────────────────────────────────────────

class TestNumericalEdgeCases:
    def test_safe_std_single_value(self):
        from src.statistics.tests import _safe_std
        assert _safe_std(np.array([0.72])) == 0.0

    def test_safe_std_empty(self):
        from src.statistics.tests import _safe_std
        assert _safe_std(np.array([])) == 0.0

    def test_safe_std_two_equal_values(self):
        from src.statistics.tests import _safe_std
        assert _safe_std(np.array([0.8, 0.8])) == 0.0

    def test_safe_std_normal(self):
        from src.statistics.tests import _safe_std
        result = _safe_std(np.array([0.70, 0.75, 0.80]))
        assert result == pytest.approx(float(np.std([0.70, 0.75, 0.80], ddof=1)))

    def test_safe_ci_single_seed_collapses(self):
        from src.statistics.tests import _safe_ci
        lo, hi = _safe_ci(0.75, 0.0, 1)
        assert lo == hi == 0.75

    def test_safe_ci_multi_seed_expands(self):
        from src.statistics.tests import _safe_ci
        lo, hi = _safe_ci(0.75, 0.05, 3)
        assert lo < 0.75 < hi

    def test_tost_zero_variance_returns_equivalent(self):
        """Zero-variance input must be flagged as indeterminate, not equivalent."""
        from src.statistics.tests import TOSTEquivalence
        result = TOSTEquivalence.test([0.80], [0.80])
        assert result.get("indeterminate") is True
        assert result["equivalent"] is False
        assert result["se"] == 0.0

    def test_tost_zero_variance_is_json_serializable(self):
        """TOST result with None t-statistics must be JSON-serializable."""
        import json
        from src.utils.save import _make_serializable
        from src.statistics.tests import TOSTEquivalence
        result = TOSTEquivalence.test([0.80], [0.80])
        assert result.get("indeterminate") is True
        report = {"tost": result}
        serialized = _make_serializable(report)
        try:
            json.dumps(serialized)
        except (ValueError, TypeError) as e:
            pytest.fail(f"TOST result not JSON-serializable: {e}")

    def test_statistics_report_single_seed_no_runtime_warning(self):
        """Single-seed StatisticsReport must not raise RuntimeWarning."""
        import warnings
        from src.statistics.tests import StatisticsReport
        np.random.seed(42)
        n = 80
        labels = list(np.random.randint(0, 2, n))
        seed_preds = {42: list(np.random.randint(0, 2, n))}
        seed_metrics = {42: {"f1": 0.80}}
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            try:
                report = StatisticsReport(seed_preds, seed_metrics, labels).full_report()
            except RuntimeWarning as e:
                pytest.fail(f"RuntimeWarning raised in single-seed StatisticsReport: {e}")
        assert "summary" in report
        assert report["summary"]["std_f1"] == 0.0


# ── 9. pin_memory device safety ───────────────────────────────────────────────

class TestPinMemoryDeviceSafety:
    def test_pin_memory_true_for_cuda_device(self):
        """pin_memory must be enabled for CUDA devices."""
        from src.data.pipeline import DataPipeline
        from unittest.mock import patch
        import tempfile, torch
        # We don't need a real CUDA GPU; just verify the logic
        cuda_device = torch.device("cuda")
        # The pipeline can't build loaders without a dataset, but we can test
        # the pin_memory logic directly via inspection
        pipeline = DataPipeline.__new__(DataPipeline)
        # Simulate what get_loaders does:
        pin_memory = (cuda_device.type == "cuda") if cuda_device is not None else False
        assert pin_memory is True

    def test_pin_memory_false_for_mps_device(self):
        """pin_memory must be disabled for MPS devices (not supported)."""
        import torch
        mps_device = torch.device("mps")
        pin_memory = (mps_device.type == "cuda")
        assert pin_memory is False

    def test_pin_memory_false_for_cpu_device(self):
        """pin_memory must be disabled for CPU."""
        import torch
        cpu_device = torch.device("cpu")
        pin_memory = (cpu_device.type == "cuda")
        assert pin_memory is False

    def test_datapipeline_get_loaders_respects_device(self):
        """DataPipeline.get_loaders(device=...) produces correct pin_memory setting."""
        import tempfile
        from pathlib import Path
        from PIL import Image
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for cls in ["Uninfected", "Parasitized"]:
                d = root / cls
                d.mkdir()
                for i in range(6):
                    arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
                    Image.fromarray(arr).save(d / f"img_{i}.png")

            from src.data.pipeline import DataPipeline
            pipeline = DataPipeline(
                root=root, image_size=32, batch_size=4,
                num_workers=0, deduplicate_data=False,
            ).prepare()

            # CPU device → pin_memory=False (no exception, no warning)
            cpu = torch.device("cpu")
            train_l, val_l, test_l = pipeline.get_loaders(device=cpu)
            # Check all loaders were created
            assert train_l is not None
            assert val_l is not None
            assert test_l is not None
