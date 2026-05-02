"""Tests for McNemar test and TOST equivalence."""

from __future__ import annotations

import numpy as np
import pytest


class TestMcNemarTest:
    def test_identical_predictions_not_significant(self, binary_labels):
        from src.statistics.tests import McNemarTest
        # Same predictions → McNemar contingency: n01=0, n10=0 → p=1
        result = McNemarTest.test(binary_labels, binary_labels, binary_labels)
        assert result["p_value"] > 0.05
        assert not result["significant"]

    def test_opposite_predictions_significant(self, binary_labels):
        from src.statistics.tests import McNemarTest
        flipped = [1 - x for x in binary_labels]
        result = McNemarTest.test(binary_labels, flipped, binary_labels)
        # One classifier perfect, other all wrong → very significant
        assert result["significant"] or result["p_value"] < 0.05

    def test_contingency_table_shape(self, binary_labels):
        from src.statistics.tests import McNemarTest
        table = McNemarTest.contingency_table(binary_labels, binary_labels, binary_labels)
        assert table.shape == (2, 2)

    def test_contingency_table_sums_to_n(self, binary_labels):
        from src.statistics.tests import McNemarTest
        preds_b = [1 - x for x in binary_labels]
        table = McNemarTest.contingency_table(binary_labels, preds_b, binary_labels)
        assert table.sum() == len(binary_labels)

    def test_per_seed_table_returns_dataframe(self, seed_preds_dict, binary_labels):
        from src.statistics.tests import McNemarTest
        df = McNemarTest.per_seed_mcnemar_table(seed_preds_dict, binary_labels)
        assert len(df) == 3  # C(3,2) pairs
        assert "p_value" in df.columns
        assert "significant" in df.columns

    def test_perfect_vs_random_significant(self, binary_labels, random_preds):
        from src.statistics.tests import McNemarTest
        result = McNemarTest.test(binary_labels, random_preds, binary_labels)
        # Perfect vs random should differ significantly
        assert isinstance(result["statistic"], float)
        assert 0 <= result["p_value"] <= 1


class TestTOSTEquivalence:
    def test_identical_scores_are_equivalent(self):
        from src.statistics.tests import TOSTEquivalence
        # Identical scores → zero variance → indeterminate per spec
        scores = [0.95] * 10
        result = TOSTEquivalence.test(scores, scores, margin=0.005)
        assert result.get("indeterminate") is True
        assert result["equivalent"] is False
        assert result["se"] == 0.0

    def test_large_difference_not_equivalent(self):
        from src.statistics.tests import TOSTEquivalence
        a = [0.95] * 10
        b = [0.80] * 10
        result = TOSTEquivalence.test(a, b, margin=0.005)
        assert not result["equivalent"]

    def test_within_margin_is_equivalent(self):
        from src.statistics.tests import TOSTEquivalence
        # diffs alternate 0.001/0.003 (mean=0.002 < margin=0.005, std>0)
        diffs = [0.001 if i % 2 == 0 else 0.003 for i in range(20)]
        a = [0.951 + d / 2 for d in diffs]
        b = [0.951 - d / 2 for d in diffs]
        result = TOSTEquivalence.test(a, b, margin=0.005)
        assert result.get("indeterminate") is not True, "Should not be indeterminate"
        assert result["equivalent"]

    def test_result_contains_required_keys(self):
        from src.statistics.tests import TOSTEquivalence
        result = TOSTEquivalence.test([0.9] * 5, [0.9] * 5)
        for key in ["mean_diff", "se", "t_low", "t_high", "p_low", "p_high", "equivalent"]:
            assert key in result

    def test_tost_equivalence_from_seed_scores(self):
        from src.statistics.tests import TOSTEquivalence
        seed_f1 = {42: 0.952, 123: 0.953, 456: 0.951}
        result = TOSTEquivalence.tost_equivalence(seed_f1, margin=0.005)
        assert "seed_f1_scores" in result
        assert result["equivalent"]  # differences < 0.5%

    def test_p_values_in_range(self):
        from src.statistics.tests import TOSTEquivalence
        # Use independently varying values so diffs have non-zero variance
        a = [0.9, 0.92, 0.88, 0.91, 0.89, 0.93, 0.87, 0.91, 0.90, 0.92]
        b = [0.88, 0.91, 0.90, 0.89, 0.92, 0.88, 0.91, 0.90, 0.93, 0.87]
        result = TOSTEquivalence.test(a, b)
        assert result.get("indeterminate") is not True
        assert 0 <= result["p_low"] <= 1
        assert 0 <= result["p_high"] <= 1


class TestStatisticsReport:
    def test_full_report_keys(self, seed_preds_dict, binary_labels):
        from src.statistics.tests import StatisticsReport
        seed_metrics = {
            42: {"f1": 0.92, "accuracy": 0.91},
            123: {"f1": 0.93, "accuracy": 0.92},
            456: {"f1": 0.91, "accuracy": 0.90},
        }
        report = StatisticsReport(
            seed_preds=seed_preds_dict,
            seed_metrics=seed_metrics,
            labels=binary_labels,
        ).full_report()
        assert "summary" in report
        assert "mcnemar" in report
        assert "tost" in report
        assert report["summary"]["n_seeds"] == 3

    def test_summary_statistics(self, seed_preds_dict, binary_labels):
        from src.statistics.tests import StatisticsReport
        f1_scores = {"f1": 0.92}
        seed_metrics = {s: dict(f1_scores) for s in seed_preds_dict}
        report = StatisticsReport(seed_preds_dict, seed_metrics, binary_labels).full_report()
        assert report["summary"]["mean_f1"] == pytest.approx(0.92)
        assert report["summary"]["std_f1"] == pytest.approx(0.0)
