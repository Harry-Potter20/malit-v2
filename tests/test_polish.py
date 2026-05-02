"""
Tests for polish.md requirements:
  - McNemar majority-vote policy + mean p-value as descriptive only
  - Robust confidence monotonicity (tolerance-based)
  - Non-regression severity levels (pass/warning/fail)
  - TOST indeterminate surfaced in statistical table
  - Confidence histogram plot
  - Reporting mentions majority vote
  - mcnemar_summary.json saved with decision_method field
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ── McNemar majority-vote policy ──────────────────────────────────────────────

class TestMcNemarMajorityVotePolicy:
    def test_aggregated_contains_significance_decision(self):
        from src.statistics.tests import McNemarTest
        labels = [i % 2 for i in range(100)]
        perfect = list(labels)
        np.random.seed(7)
        bad = list(np.random.randint(0, 2, 100))
        result = McNemarTest.multi_seed_mcnemar(
            {42: perfect, 123: perfect},
            {42: bad, 123: bad},
            labels,
        )
        assert "majority_significant" in result["aggregated"]
        assert "mean_p_value" in result["aggregated"]

    def test_mean_p_value_present_as_descriptive(self):
        from src.statistics.tests import McNemarTest
        labels = [i % 2 for i in range(100)]
        perfect = list(labels)
        result = McNemarTest.multi_seed_mcnemar(
            {42: perfect},
            {42: perfect},
            labels,
        )
        # mean_p_value present even when not significant
        assert "mean_p_value" in result["aggregated"]
        assert result["aggregated"]["mean_p_value"] == pytest.approx(1.0)

    def test_generate_text_mentions_majority_vote(self):
        from src.comparison.protocol import ComparisonProtocol
        from src.comparison.reporting import generate_comparison_text
        labels = [i % 2 for i in range(100)]
        perfect = list(labels)
        np.random.seed(7)
        bad = list(np.random.randint(0, 2, 100))
        malit = {
            "seed_preds":   {42: perfect, 123: perfect},
            "seed_metrics": {42: {"f1": 0.99}, 123: {"f1": 0.98}},
        }
        baseline = {
            "weak": {
                "seed_preds":   {42: bad, 123: bad},
                "seed_metrics": {42: {"f1": 0.50}, 123: {"f1": 0.51}},
                "seed_labels":  labels,
            }
        }
        results = ComparisonProtocol(malit, baseline, labels).compare_all()
        text = generate_comparison_text(results)
        assert "majority vote" in text["statistical_statement"].lower()

    def test_generate_text_mentions_mean_p_descriptive(self):
        from src.comparison.protocol import ComparisonProtocol
        from src.comparison.reporting import generate_comparison_text
        labels = [i % 2 for i in range(100)]
        perfect = list(labels)
        np.random.seed(7)
        bad = list(np.random.randint(0, 2, 100))
        malit = {
            "seed_preds":   {42: perfect},
            "seed_metrics": {42: {"f1": 0.99}},
        }
        baseline = {
            "weak": {
                "seed_preds":   {42: bad},
                "seed_metrics": {42: {"f1": 0.50}},
                "seed_labels":  labels,
            }
        }
        results = ComparisonProtocol(malit, baseline, labels).compare_all()
        text = generate_comparison_text(results)
        assert "reference only" in text["statistical_statement"].lower()


# ── Robust confidence monotonicity ────────────────────────────────────────────

class TestRobustMonotonicity:
    def test_small_violation_within_tolerance_passes(self):
        from src.comparison.reporting import check_confidence_monotonicity
        # Rise of 0.01 < tolerance=0.02 → not counted as violation
        df = pd.DataFrame({
            "bin_low": [0.0, 0.1, 0.2],
            "bin_high": [0.1, 0.2, 0.3],
            "n": [10, 10, 10],
            "error_rate": [0.5, 0.51, 0.1],  # tiny rise of 0.01 < tolerance
        })
        result = check_confidence_monotonicity(df, tolerance=0.02)
        assert result["monotonic"] is True
        assert result["n_violations"] == 0

    def test_one_large_violation_still_passes_with_max_1(self):
        from src.comparison.reporting import check_confidence_monotonicity
        # One rise of 0.3 > tolerance → 1 violation = max_violations → passes
        df = pd.DataFrame({
            "bin_low": [0.0, 0.1, 0.2],
            "bin_high": [0.1, 0.2, 0.3],
            "n": [10, 10, 10],
            "error_rate": [0.5, 0.8, 0.1],
        })
        result = check_confidence_monotonicity(df, tolerance=0.02, max_violations=1)
        assert result["monotonic"] is True
        assert result["n_violations"] == 1

    def test_two_violations_fails(self):
        from src.comparison.reporting import check_confidence_monotonicity
        df = pd.DataFrame({
            "bin_low":   [0.0, 0.1, 0.2, 0.3],
            "bin_high":  [0.1, 0.2, 0.3, 0.4],
            "n":         [10, 10, 10, 10],
            "error_rate": [0.2, 0.7, 0.1, 0.9],
        })
        result = check_confidence_monotonicity(df, tolerance=0.02, max_violations=1)
        assert result["monotonic"] is False
        assert result["n_violations"] == 2

    def test_tolerance_field_in_result(self):
        from src.comparison.reporting import check_confidence_monotonicity
        df = pd.DataFrame({
            "bin_low": [0.0], "bin_high": [1.0], "n": [10], "error_rate": [0.3],
        })
        result = check_confidence_monotonicity(df, tolerance=0.03)
        assert result["tolerance"] == pytest.approx(0.03)

    def test_default_tolerance_is_0_02(self):
        from src.comparison.reporting import check_confidence_monotonicity
        import inspect
        sig = inspect.signature(check_confidence_monotonicity)
        assert sig.parameters["tolerance"].default == pytest.approx(0.02)


# ── Non-regression severity ───────────────────────────────────────────────────

class TestNonRegressionSeverity:
    def test_above_baseline_is_pass(self):
        from src.comparison.reporting import check_malit_non_regression
        baselines = {"m": {"seed_metrics": {42: {"f1": 0.80}}}}
        result = check_malit_non_regression(0.85, baselines)
        assert result["severity"] == "pass"
        assert result["passed"] is True

    def test_below_but_within_tolerance_is_warning(self):
        from src.comparison.reporting import check_malit_non_regression
        baselines = {"m": {"seed_metrics": {42: {"f1": 0.90}}}}
        result = check_malit_non_regression(0.895, baselines, tolerance=0.01)
        assert result["severity"] == "warning"
        assert result["passed"] is True

    def test_significantly_below_is_fail(self):
        from src.comparison.reporting import check_malit_non_regression
        baselines = {"m": {"seed_metrics": {42: {"f1": 0.90}}}}
        result = check_malit_non_regression(0.70, baselines, tolerance=0.01)
        assert result["severity"] == "fail"
        assert result["passed"] is False

    def test_exactly_at_threshold_is_warning(self):
        from src.comparison.reporting import check_malit_non_regression
        baselines = {"m": {"seed_metrics": {42: {"f1": 0.90}}}}
        # malit=0.89 = best(0.90) - tolerance(0.01) = threshold → warning (>= threshold)
        result = check_malit_non_regression(0.89, baselines, tolerance=0.01)
        assert result["severity"] == "warning"
        assert result["passed"] is True


# ── TOST indeterminate surfaced in table ──────────────────────────────────────

class TestTOSTIndeterminateInTable:
    def test_indeterminate_shown_in_tost_result_column(self):
        from src.comparison.protocol import ComparisonProtocol
        from src.comparison.reporting import build_statistical_table
        labels = [i % 2 for i in range(100)]
        perfect = list(labels)
        malit = {
            "seed_preds":   {42: perfect},
            "seed_metrics": {42: {"f1": 0.99}},
        }
        baseline = {
            "copy": {
                "seed_preds":   {42: perfect},
                "seed_metrics": {42: {"f1": 0.99}},
                "seed_labels":  labels,
            }
        }
        results = ComparisonProtocol(malit, baseline, labels).compare_all()
        df = build_statistical_table(results)
        tost_val = df[df["Baseline"] == "copy"]["TOST_result"].iloc[0]
        assert "indeterminate" in tost_val.lower()

    def test_non_indeterminate_shows_true_or_false(self):
        from src.comparison.protocol import ComparisonProtocol
        from src.comparison.reporting import build_statistical_table
        labels = [i % 2 for i in range(100)]
        np.random.seed(7)
        bad = list(np.random.randint(0, 2, 100))
        perfect = list(labels)
        malit = {
            "seed_preds":   {42: perfect, 123: perfect},
            "seed_metrics": {42: {"f1": 0.99}, 123: {"f1": 0.98}},
        }
        baseline = {
            "weak": {
                "seed_preds":   {42: bad, 123: bad},
                "seed_metrics": {42: {"f1": 0.50}, 123: {"f1": 0.51}},
                "seed_labels":  labels,
            }
        }
        results = ComparisonProtocol(malit, baseline, labels).compare_all()
        df = build_statistical_table(results)
        tost_val = df[df["Baseline"] == "weak"]["TOST_result"].iloc[0]
        assert tost_val in ("True", "False")

    def test_generate_text_mentions_indeterminate(self):
        from src.comparison.protocol import ComparisonProtocol
        from src.comparison.reporting import generate_comparison_text
        labels = [i % 2 for i in range(100)]
        perfect = list(labels)
        malit = {
            "seed_preds":   {42: perfect},
            "seed_metrics": {42: {"f1": 0.99}},
        }
        baseline = {
            "copy": {
                "seed_preds":   {42: perfect},
                "seed_metrics": {42: {"f1": 0.99}},
                "seed_labels":  labels,
            }
        }
        results = ComparisonProtocol(malit, baseline, labels).compare_all()
        text = generate_comparison_text(results)
        assert "indeterminate" in text["comparison_statement"].lower()


# ── Confidence histogram ──────────────────────────────────────────────────────

class TestConfidenceHistogram:
    def _make_data(self):
        np.random.seed(0)
        conf = list(np.random.uniform(0.5, 1.0, 200))
        errors = [0 if c > 0.75 else 1 for c in conf]
        return conf, errors

    def test_returns_figure_or_none(self):
        from src.comparison.reporting import plot_confidence_histogram
        conf, errors = self._make_data()
        with tempfile.TemporaryDirectory() as tmp:
            result = plot_confidence_histogram(conf, errors, Path(tmp) / "hist.png")
        assert result is None or hasattr(result, "savefig")

    def test_output_file_created(self):
        from src.comparison.reporting import plot_confidence_histogram
        conf, errors = self._make_data()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "confidence_histogram.png"
            result = plot_confidence_histogram(conf, errors, out)
            if result is not None:
                assert out.exists()

    def test_all_correct_does_not_crash(self):
        from src.comparison.reporting import plot_confidence_histogram
        conf = [0.9] * 50
        errors = [0] * 50
        with tempfile.TemporaryDirectory() as tmp:
            plot_confidence_histogram(conf, errors, Path(tmp) / "h.png")

    def test_all_incorrect_does_not_crash(self):
        from src.comparison.reporting import plot_confidence_histogram
        conf = [0.6] * 50
        errors = [1] * 50
        with tempfile.TemporaryDirectory() as tmp:
            plot_confidence_histogram(conf, errors, Path(tmp) / "h.png")


# ── mcnemar_summary.json ──────────────────────────────────────────────────────

class TestMcNemarSummaryJson:
    @pytest.fixture
    def setup_comparison(self):
        labels = [i % 2 for i in range(100)]
        np.random.seed(7)
        bad = list(np.random.randint(0, 2, 100))
        perfect = list(labels)
        malit = {
            "seed_preds":   {42: perfect, 123: perfect},
            "seed_metrics": {42: {"f1": 0.99}, 123: {"f1": 0.98}},
        }
        baseline = {
            "weak": {
                "seed_preds":   {42: bad, 123: bad},
                "seed_metrics": {42: {"f1": 0.50}, 123: {"f1": 0.51}},
                "seed_labels":  labels,
            }
        }
        from src.comparison.protocol import ComparisonProtocol
        protocol = ComparisonProtocol(malit, baseline, labels)
        results = protocol.compare_all()
        return protocol, results

    def test_mcnemar_summary_json_created(self, setup_comparison, tmp_path):
        protocol, results = setup_comparison
        protocol.save_reproducibility(results, tmp_path)
        assert (tmp_path / "results" / "statistics" / "mcnemar_summary.json").exists()

    def test_summary_contains_significance_decision(self, setup_comparison, tmp_path):
        protocol, results = setup_comparison
        protocol.save_reproducibility(results, tmp_path)
        data = json.loads(
            (tmp_path / "results" / "statistics" / "mcnemar_summary.json").read_text()
        )
        assert "weak" in data
        assert "significance_decision" in data["weak"]
        assert data["weak"]["significance_decision"] in ("Significant", "Not Significant")

    def test_summary_contains_decision_method(self, setup_comparison, tmp_path):
        protocol, results = setup_comparison
        protocol.save_reproducibility(results, tmp_path)
        data = json.loads(
            (tmp_path / "results" / "statistics" / "mcnemar_summary.json").read_text()
        )
        assert data["weak"]["decision_method"] == "majority_vote"

    def test_summary_note_mentions_descriptive(self, setup_comparison, tmp_path):
        protocol, results = setup_comparison
        protocol.save_reproducibility(results, tmp_path)
        data = json.loads(
            (tmp_path / "results" / "statistics" / "mcnemar_summary.json").read_text()
        )
        assert "reference only" in data["weak"]["note"].lower()
