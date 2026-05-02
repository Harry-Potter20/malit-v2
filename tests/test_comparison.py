"""
Tests for the statistical comparison protocol and paper reporting.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def shared_labels():
    return [i % 2 for i in range(100)]


@pytest.fixture
def malit_results_superior(shared_labels):
    """MALIT with near-perfect F1 on seed 42 & 123."""
    labels = shared_labels
    return {
        "seed_preds":   {42: list(labels), 123: list(labels)},
        "seed_metrics": {
            42:  {"f1": 0.99, "sensitivity": 0.99, "specificity": 0.99},
            123: {"f1": 0.98, "sensitivity": 0.98, "specificity": 0.98},
        },
        "seed_probs": {
            42: np.eye(2)[labels].tolist(),
        },
        "seed_labels": labels,
    }


@pytest.fixture
def baseline_results_inferior(shared_labels):
    """Baseline (efficientnet) with random ~50% accuracy."""
    np.random.seed(7)
    labels = shared_labels
    bad_preds = list(np.random.randint(0, 2, 100))
    return {
        "efficientnet": {
            "seed_preds":   {42: bad_preds, 123: bad_preds},
            "seed_metrics": {
                42:  {"f1": 0.50, "sensitivity": 0.50, "specificity": 0.50},
                123: {"f1": 0.51, "sensitivity": 0.51, "specificity": 0.51},
            },
            "seed_probs": {42: (np.random.rand(100, 2) / 2).tolist()},
            "seed_labels": labels,
        }
    }


@pytest.fixture
def baseline_results_identical(shared_labels):
    """Baseline that is IDENTICAL to MALIT (for equivalence test)."""
    labels = shared_labels
    return {
        "efficientnet": {
            "seed_preds":   {42: list(labels), 123: list(labels)},
            "seed_metrics": {
                42:  {"f1": 0.99, "sensitivity": 0.99, "specificity": 0.99},
                123: {"f1": 0.98, "sensitivity": 0.98, "specificity": 0.98},
            },
            "seed_probs": {42: np.eye(2)[labels].tolist()},
            "seed_labels": labels,
        }
    }


@pytest.fixture
def all_results_for_table(shared_labels, malit_results_superior, baseline_results_inferior):
    """Combined results dict for performance table."""
    combined = {"malit_v2": malit_results_superior}
    combined.update(baseline_results_inferior)
    return combined


# ── ComparisonProtocol verdicts ───────────────────────────────────────────────

class TestComparisonProtocolVerdicts:
    def test_superior_verdict_when_malit_clearly_better(
        self, malit_results_superior, baseline_results_inferior, shared_labels
    ):
        from src.comparison.protocol import ComparisonProtocol
        protocol = ComparisonProtocol(
            malit_results_superior, baseline_results_inferior, shared_labels
        )
        results = protocol.compare_all()
        assert results["efficientnet"]["verdict"] == "Superior"

    def test_equivalent_verdict_when_predictions_identical(
        self, malit_results_superior, baseline_results_identical, shared_labels
    ):
        from src.comparison.protocol import ComparisonProtocol
        protocol = ComparisonProtocol(
            malit_results_superior, baseline_results_identical, shared_labels
        )
        results = protocol.compare_all()
        # Identical predictions → McNemar not significant, TOST equivalent
        assert results["efficientnet"]["verdict"] in ("Equivalent", "Inconclusive")

    def test_failure_flag_mismatched_sample_counts(
        self, malit_results_superior, shared_labels
    ):
        from src.comparison.protocol import ComparisonProtocol
        # Baseline with wrong number of predictions
        bad_baseline = {
            "bad_model": {
                "seed_preds":   {42: [0] * 50},   # wrong length!
                "seed_metrics": {42: {"f1": 0.70}},
                "seed_labels":  shared_labels,
            }
        }
        protocol = ComparisonProtocol(
            malit_results_superior, bad_baseline, shared_labels
        )
        results = protocol.compare_all()
        assert any("mismatch" in f for f in results["bad_model"]["failure_flags"])

    def test_failure_flag_no_shared_seeds(
        self, malit_results_superior, shared_labels
    ):
        from src.comparison.protocol import ComparisonProtocol
        # Baseline only has seed 999, not 42 or 123
        baseline_no_shared = {
            "no_seed": {
                "seed_preds":   {999: list(shared_labels)},
                "seed_metrics": {999: {"f1": 0.80}},
                "seed_labels":  shared_labels,
            }
        }
        protocol = ComparisonProtocol(
            malit_results_superior, baseline_no_shared, shared_labels
        )
        results = protocol.compare_all()
        assert "no_shared_seeds" in results["no_seed"]["failure_flags"]

    def test_delta_f1_sign_is_correct(
        self, malit_results_superior, baseline_results_inferior, shared_labels
    ):
        from src.comparison.protocol import ComparisonProtocol
        protocol = ComparisonProtocol(
            malit_results_superior, baseline_results_inferior, shared_labels
        )
        results = protocol.compare_all()
        assert results["efficientnet"]["delta_f1"] > 0, "MALIT should have higher F1"

    def test_nan_metrics_trigger_failure_flag(self, shared_labels):
        from src.comparison.protocol import ComparisonProtocol
        malit = {
            "seed_preds":   {42: list(shared_labels)},
            "seed_metrics": {42: {"f1": float("nan")}},
        }
        baseline = {
            "model": {
                "seed_preds":   {42: list(shared_labels)},
                "seed_metrics": {42: {"f1": 0.7}},
                "seed_labels":  shared_labels,
            }
        }
        protocol = ComparisonProtocol(malit, baseline, shared_labels)
        results = protocol.compare_all()
        assert results["model"]["failure_flags"] or np.isnan(results["model"]["delta_f1"])


# ── Error overlap analysis ────────────────────────────────────────────────────

class TestErrorOverlapAnalysis:
    def _setup(self, shared_labels):
        from src.comparison.protocol import ComparisonProtocol
        labels = shared_labels
        # MALIT correct on all; baseline wrong on first 20
        malit_preds = list(labels)
        bl_preds = list(labels)
        for i in range(20):
            bl_preds[i] = 1 - labels[i]   # flip first 20

        malit = {
            "seed_preds":   {42: malit_preds},
            "seed_metrics": {42: {"f1": 0.99}},
        }
        baseline = {
            "model": {
                "seed_preds":   {42: bl_preds},
                "seed_metrics": {42: {"f1": 0.80}},
                "seed_labels":  labels,
            }
        }
        protocol = ComparisonProtocol(malit, baseline, labels)
        return protocol.compare_all()["model"]["error_overlap"]

    def test_both_correct_count(self, shared_labels):
        overlap = self._setup(shared_labels)
        assert overlap["both_correct"] == 80

    def test_malit_correct_baseline_wrong(self, shared_labels):
        overlap = self._setup(shared_labels)
        assert overlap["malit_correct_bl_wrong"] == 20

    def test_both_wrong_count(self, shared_labels):
        overlap = self._setup(shared_labels)
        assert overlap["both_wrong"] == 0

    def test_overlap_sums_to_n(self, shared_labels):
        overlap = self._setup(shared_labels)
        total = sum(overlap.values())
        assert total == 100


# ── Performance table ─────────────────────────────────────────────────────────

class TestPerformanceTable:
    def test_correct_column_names(self, all_results_for_table):
        from src.comparison.reporting import build_performance_table
        df = build_performance_table(all_results_for_table)
        for col in ("Model", "F1 (mean±CI)", "Sensitivity", "Specificity", "AUC"):
            assert col in df.columns, f"Missing column: {col}"

    def test_row_per_model(self, all_results_for_table):
        from src.comparison.reporting import build_performance_table
        df = build_performance_table(all_results_for_table)
        assert len(df) == len(all_results_for_table)

    def test_f1_format_contains_pm_sign(self, all_results_for_table):
        from src.comparison.reporting import build_performance_table
        df = build_performance_table(all_results_for_table)
        for val in df["F1 (mean±CI)"]:
            if val != "—":
                assert "±" in val, f"Expected ± in F1 column, got: {val}"

    def test_empty_seed_metrics_handled(self):
        from src.comparison.reporting import build_performance_table
        results = {"empty_model": {"seed_metrics": {}}}
        df = build_performance_table(results)
        assert len(df) == 0   # empty metrics → row not added


# ── Statistical comparison table ──────────────────────────────────────────────

class TestStatisticalTable:
    def test_correct_column_names(
        self, malit_results_superior, baseline_results_inferior, shared_labels
    ):
        from src.comparison.protocol import ComparisonProtocol
        from src.comparison.reporting import build_statistical_table
        results = ComparisonProtocol(
            malit_results_superior, baseline_results_inferior, shared_labels
        ).compare_all()
        df = build_statistical_table(results)
        for col in ("Baseline", "ΔF1", "McNemar_p", "TOST_result", "Conclusion"):
            assert col in df.columns, f"Missing column: {col}"

    def test_conclusion_values_are_valid(
        self, malit_results_superior, baseline_results_inferior, shared_labels
    ):
        from src.comparison.protocol import ComparisonProtocol
        from src.comparison.reporting import build_statistical_table
        results = ComparisonProtocol(
            malit_results_superior, baseline_results_inferior, shared_labels
        ).compare_all()
        df = build_statistical_table(results)
        valid_verdicts = {"Superior", "Equivalent", "Inconclusive"}
        for verdict in df["Conclusion"]:
            assert verdict in valid_verdicts, f"Invalid verdict: {verdict}"

    def test_one_row_per_baseline(
        self, malit_results_superior, baseline_results_inferior, shared_labels
    ):
        from src.comparison.protocol import ComparisonProtocol
        from src.comparison.reporting import build_statistical_table
        results = ComparisonProtocol(
            malit_results_superior, baseline_results_inferior, shared_labels
        ).compare_all()
        df = build_statistical_table(results)
        assert len(df) == 1   # one baseline in fixture


# ── Reporting outputs ─────────────────────────────────────────────────────────

class TestReportingOutputs:
    def test_save_paper_tables_creates_csv_files(self, all_results_for_table):
        from src.comparison.reporting import build_performance_table, save_paper_tables
        df = build_performance_table(all_results_for_table)
        with tempfile.TemporaryDirectory() as tmp:
            saved = save_paper_tables({"perf": df}, tmp)
            assert "perf" in saved
            assert saved["perf"].exists()
            # Must be valid CSV
            import pandas as pd
            df2 = pd.read_csv(saved["perf"])
            assert len(df2) == len(df)

    def test_latex_output_does_not_crash(self, all_results_for_table):
        from src.comparison.reporting import build_performance_table, save_paper_tables
        df = build_performance_table(all_results_for_table)
        with tempfile.TemporaryDirectory() as tmp:
            save_paper_tables({"perf": df}, tmp)   # should not raise

    def test_generate_comparison_text_keys(
        self, malit_results_superior, baseline_results_inferior, shared_labels
    ):
        from src.comparison.protocol import ComparisonProtocol
        from src.comparison.reporting import generate_comparison_text
        results = ComparisonProtocol(
            malit_results_superior, baseline_results_inferior, shared_labels
        ).compare_all()
        text = generate_comparison_text(results)
        assert "comparison_statement" in text
        assert "statistical_statement" in text
        assert len(text["comparison_statement"]) > 0
        assert len(text["statistical_statement"]) > 0

    def test_comparison_statement_mentions_model(
        self, malit_results_superior, baseline_results_inferior, shared_labels
    ):
        from src.comparison.protocol import ComparisonProtocol
        from src.comparison.reporting import generate_comparison_text
        results = ComparisonProtocol(
            malit_results_superior, baseline_results_inferior, shared_labels
        ).compare_all()
        text = generate_comparison_text(results)
        # The word "efficientnet" (the baseline name) should appear in the statement
        assert "efficientnet" in text["comparison_statement"].lower()

    def test_plot_roc_returns_none_or_object(self, all_results_for_table):
        from src.comparison.reporting import plot_roc_curves
        with tempfile.TemporaryDirectory() as tmp:
            result = plot_roc_curves(all_results_for_table, Path(tmp) / "roc.png")
        # Returns None (if matplotlib unavailable) or a Figure
        assert result is None or hasattr(result, "savefig")

    def test_plot_calibration_returns_none_or_object(self, all_results_for_table):
        from src.comparison.reporting import plot_calibration
        with tempfile.TemporaryDirectory() as tmp:
            result = plot_calibration(all_results_for_table, Path(tmp) / "cal.png")
        assert result is None or hasattr(result, "savefig")

    def test_comparison_results_json_serializable(
        self, malit_results_superior, baseline_results_inferior, shared_labels
    ):
        from src.comparison.protocol import ComparisonProtocol
        from src.utils.save import _make_serializable
        results = ComparisonProtocol(
            malit_results_superior, baseline_results_inferior, shared_labels
        ).compare_all()
        serialized = _make_serializable(results)
        json.dumps(serialized)   # must not raise
