"""
Tests for final_fix.md requirements:
  - Multi-seed McNemar (per-seed + aggregated)
  - TOST indeterminate on zero variance
  - _make_serializable handles non-finite floats
  - Confidence bin analysis and monotonicity
  - MALIT non-regression check
  - cudnn.benchmark flag on CUDA
  - Baseline hparams.json
  - Reproducibility JSON files
  - ComparisonProtocol multi-seed integration
"""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def labels_100():
    return [i % 2 for i in range(100)]


@pytest.fixture
def perfect_preds(labels_100):
    return list(labels_100)


@pytest.fixture
def random_preds():
    np.random.seed(7)
    return list(np.random.randint(0, 2, 100))


# ── Multi-seed McNemar ────────────────────────────────────────────────────────

class TestMultiSeedMcNemar:
    def test_returns_per_seed_and_aggregated(self, labels_100, perfect_preds, random_preds):
        from src.statistics.tests import McNemarTest
        result = McNemarTest.multi_seed_mcnemar(
            {42: perfect_preds, 123: perfect_preds},
            {42: random_preds, 123: random_preds},
            labels_100,
        )
        assert "per_seed" in result
        assert "aggregated" in result
        assert 42 in result["per_seed"]
        assert 123 in result["per_seed"]

    def test_per_seed_has_required_keys(self, labels_100, perfect_preds, random_preds):
        from src.statistics.tests import McNemarTest
        result = McNemarTest.multi_seed_mcnemar(
            {42: perfect_preds},
            {42: random_preds},
            labels_100,
        )
        for key in ("statistic", "p_value", "significant"):
            assert key in result["per_seed"][42]

    def test_aggregated_has_mean_p_value(self, labels_100, perfect_preds, random_preds):
        from src.statistics.tests import McNemarTest
        result = McNemarTest.multi_seed_mcnemar(
            {42: perfect_preds, 123: perfect_preds},
            {42: random_preds, 123: random_preds},
            labels_100,
        )
        agg = result["aggregated"]
        assert "mean_p_value" in agg
        assert 0.0 <= agg["mean_p_value"] <= 1.0

    def test_aggregated_majority_significant_perfect_vs_random(
        self, labels_100, perfect_preds, random_preds
    ):
        from src.statistics.tests import McNemarTest
        result = McNemarTest.multi_seed_mcnemar(
            {42: perfect_preds, 123: perfect_preds},
            {42: random_preds, 123: random_preds},
            labels_100,
        )
        assert result["aggregated"]["majority_significant"] is True

    def test_aggregated_not_significant_for_identical_preds(
        self, labels_100, perfect_preds
    ):
        from src.statistics.tests import McNemarTest
        result = McNemarTest.multi_seed_mcnemar(
            {42: perfect_preds, 123: perfect_preds},
            {42: perfect_preds, 123: perfect_preds},
            labels_100,
        )
        assert result["aggregated"]["majority_significant"] is False

    def test_no_shared_seeds_returns_failure(self, labels_100, perfect_preds):
        from src.statistics.tests import McNemarTest
        result = McNemarTest.multi_seed_mcnemar(
            {42: perfect_preds},
            {999: perfect_preds},
            labels_100,
        )
        assert result["aggregated"]["n_seeds"] == 0
        assert "failure" in result

    def test_n_seeds_counts_shared(self, labels_100, perfect_preds, random_preds):
        from src.statistics.tests import McNemarTest
        result = McNemarTest.multi_seed_mcnemar(
            {42: perfect_preds, 123: perfect_preds, 456: perfect_preds},
            {42: random_preds, 123: random_preds},
            labels_100,
        )
        assert result["aggregated"]["n_seeds"] == 2


# ── TOST indeterminate ────────────────────────────────────────────────────────

class TestTOSTIndeterminate:
    def test_zero_variance_returns_indeterminate(self):
        from src.statistics.tests import TOSTEquivalence
        result = TOSTEquivalence.test([0.80] * 10, [0.80] * 10)
        assert result.get("indeterminate") is True
        assert result["equivalent"] is False

    def test_zero_variance_t_low_t_high_are_none(self):
        from src.statistics.tests import TOSTEquivalence
        result = TOSTEquivalence.test([0.90] * 5, [0.91] * 5)
        assert result.get("indeterminate") is True
        assert result["t_low"] is None
        assert result["t_high"] is None

    def test_zero_variance_p_values_are_none(self):
        from src.statistics.tests import TOSTEquivalence
        result = TOSTEquivalence.test([0.90] * 5, [0.91] * 5)
        assert result["p_low"] is None
        assert result["p_high"] is None

    def test_large_diff_zero_variance_not_equivalent(self):
        from src.statistics.tests import TOSTEquivalence
        result = TOSTEquivalence.test([0.95] * 10, [0.80] * 10)
        assert result["equivalent"] is False
        assert result.get("indeterminate") is True

    def test_nonzero_variance_not_indeterminate(self):
        from src.statistics.tests import TOSTEquivalence
        diffs = [0.001 if i % 2 == 0 else 0.003 for i in range(20)]
        a = [0.951 + d / 2 for d in diffs]
        b = [0.951 - d / 2 for d in diffs]
        result = TOSTEquivalence.test(a, b, margin=0.005)
        assert result.get("indeterminate") is not True
        assert result["p_low"] is not None
        assert result["p_high"] is not None

    def test_indeterminate_result_json_serializable(self):
        import json
        from src.statistics.tests import TOSTEquivalence
        from src.utils.save import _make_serializable
        result = TOSTEquivalence.test([0.80] * 5, [0.80] * 5)
        serialized = _make_serializable({"tost": result})
        json.dumps(serialized)  # must not raise


# ── _make_serializable non-finite float handling ──────────────────────────────

class TestMakeSerializableNonFinite:
    def test_inf_becomes_none(self):
        from src.utils.save import _make_serializable
        assert _make_serializable(float("inf")) is None

    def test_neg_inf_becomes_none(self):
        from src.utils.save import _make_serializable
        assert _make_serializable(float("-inf")) is None

    def test_nan_becomes_none(self):
        from src.utils.save import _make_serializable
        assert _make_serializable(float("nan")) is None

    def test_finite_float_preserved(self):
        from src.utils.save import _make_serializable
        assert _make_serializable(3.14) == pytest.approx(3.14)

    def test_nested_non_finite_handled(self):
        from src.utils.save import _make_serializable
        data = {"t_low": float("inf"), "t_high": float("-inf"), "se": 0.0}
        result = _make_serializable(data)
        assert result["t_low"] is None
        assert result["t_high"] is None
        assert result["se"] == 0.0

    def test_numpy_non_finite_handled(self):
        from src.utils.save import _make_serializable
        import numpy as np
        assert _make_serializable(np.float64("inf")) is None
        assert _make_serializable(np.float64("nan")) is None

    def test_json_dumps_after_serialization(self):
        from src.utils.save import _make_serializable
        data = {"a": float("inf"), "b": [1.0, float("nan"), 3.0], "c": "ok"}
        json.dumps(_make_serializable(data))  # must not raise


# ── Confidence bins ───────────────────────────────────────────────────────────

class TestConfidenceBins:
    def _make_data(self):
        # High confidence (0.9-1.0) → correct; low confidence (0.5-0.6) → wrong
        np.random.seed(0)
        n = 200
        # High-conf correct
        conf_hi = list(np.random.uniform(0.9, 1.0, n // 2))
        err_hi = [0] * (n // 2)
        # Low-conf wrong
        conf_lo = list(np.random.uniform(0.5, 0.6, n // 2))
        err_lo = [1] * (n // 2)
        return conf_hi + conf_lo, err_hi + err_lo

    def test_returns_dataframe_with_correct_columns(self):
        from src.comparison.reporting import compute_confidence_bins
        conf, err = self._make_data()
        df = compute_confidence_bins(conf, err)
        for col in ("bin_low", "bin_high", "n", "error_rate"):
            assert col in df.columns

    def test_n_rows_equals_bins(self):
        from src.comparison.reporting import compute_confidence_bins
        conf, err = self._make_data()
        df = compute_confidence_bins(conf, err, bins=10)
        assert len(df) == 10

    def test_n_sums_to_total(self):
        from src.comparison.reporting import compute_confidence_bins
        conf, err = self._make_data()
        df = compute_confidence_bins(conf, err)
        assert df["n"].sum() == len(conf)

    def test_custom_bin_count(self):
        from src.comparison.reporting import compute_confidence_bins
        conf, err = self._make_data()
        df = compute_confidence_bins(conf, err, bins=5)
        assert len(df) == 5

    def test_high_conf_lower_error_rate(self):
        from src.comparison.reporting import compute_confidence_bins
        conf, err = self._make_data()
        df = compute_confidence_bins(conf, err)
        populated = df[df["n"] > 0]
        low_conf_err = populated.iloc[0]["error_rate"]   # lowest confidence bin
        high_conf_err = populated.iloc[-1]["error_rate"] # highest confidence bin
        assert high_conf_err < low_conf_err


# ── Confidence monotonicity ───────────────────────────────────────────────────

class TestConfidenceMonotonicity:
    def test_monotonic_data_passes(self):
        from src.comparison.reporting import compute_confidence_bins, check_confidence_monotonicity
        import pandas as pd
        # Strictly decreasing error rates
        df = pd.DataFrame({
            "bin_low": [0.0, 0.1, 0.2],
            "bin_high": [0.1, 0.2, 0.3],
            "n": [10, 10, 10],
            "error_rate": [0.8, 0.5, 0.1],
        })
        result = check_confidence_monotonicity(df)
        assert result["monotonic"] is True
        assert result["n_violations"] == 0

    def test_non_monotonic_data_fails(self):
        from src.comparison.reporting import check_confidence_monotonicity
        import pandas as pd
        # Two large rises (> tolerance=0.02) → exceeds max_violations=1 → monotonic=False
        df = pd.DataFrame({
            "bin_low":   [0.0, 0.1, 0.2, 0.3],
            "bin_high":  [0.1, 0.2, 0.3, 0.4],
            "n":         [10, 10, 10, 10],
            "error_rate": [0.3, 0.7, 0.2, 0.8],  # two rises: +0.4 and +0.6
        })
        result = check_confidence_monotonicity(df)
        assert result["monotonic"] is False
        assert result["n_violations"] == 2

    def test_empty_bins_excluded(self):
        from src.comparison.reporting import check_confidence_monotonicity
        import pandas as pd
        df = pd.DataFrame({
            "bin_low": [0.0, 0.1, 0.2],
            "bin_high": [0.1, 0.2, 0.3],
            "n": [0, 10, 10],           # first bin empty
            "error_rate": [float("nan"), 0.5, 0.1],
        })
        result = check_confidence_monotonicity(df)
        assert result["monotonic"] is True
        assert result["n_bins_checked"] == 2

    def test_single_bin_returns_none(self):
        from src.comparison.reporting import check_confidence_monotonicity
        import pandas as pd
        df = pd.DataFrame({
            "bin_low": [0.9], "bin_high": [1.0], "n": [10], "error_rate": [0.1],
        })
        result = check_confidence_monotonicity(df)
        assert result["monotonic"] is None


# ── MALIT non-regression ──────────────────────────────────────────────────────

class TestMalitNonRegression:
    def test_malit_above_best_passes(self):
        from src.comparison.reporting import check_malit_non_regression
        baselines = {
            "modelA": {"seed_metrics": {42: {"f1": 0.80}, 123: {"f1": 0.82}}},
            "modelB": {"seed_metrics": {42: {"f1": 0.75}}},
        }
        result = check_malit_non_regression(malit_f1=0.90, baseline_results=baselines)
        assert result["passed"] is True
        assert result["severity"] == "pass"

    def test_malit_far_below_fails(self):
        from src.comparison.reporting import check_malit_non_regression
        baselines = {
            "modelA": {"seed_metrics": {42: {"f1": 0.90}}},
        }
        result = check_malit_non_regression(malit_f1=0.75, baseline_results=baselines, tolerance=0.01)
        assert result["passed"] is False
        assert result["severity"] == "fail"

    def test_malit_within_tolerance_passes_with_warning(self):
        from src.comparison.reporting import check_malit_non_regression
        baselines = {
            "modelA": {"seed_metrics": {42: {"f1": 0.90}}},
        }
        result = check_malit_non_regression(malit_f1=0.895, baseline_results=baselines, tolerance=0.01)
        assert result["passed"] is True
        assert result["severity"] == "warning"

    def test_identifies_best_baseline(self):
        from src.comparison.reporting import check_malit_non_regression
        baselines = {
            "weak":   {"seed_metrics": {42: {"f1": 0.70}}},
            "strong": {"seed_metrics": {42: {"f1": 0.88}}},
        }
        result = check_malit_non_regression(malit_f1=0.90, baseline_results=baselines)
        assert result["best_baseline_name"] == "strong"

    def test_no_baselines_returns_none(self):
        from src.comparison.reporting import check_malit_non_regression
        result = check_malit_non_regression(malit_f1=0.90, baseline_results={})
        assert result["passed"] is None

    def test_result_contains_threshold(self):
        from src.comparison.reporting import check_malit_non_regression
        baselines = {"m": {"seed_metrics": {42: {"f1": 0.85}}}}
        result = check_malit_non_regression(0.90, baselines, tolerance=0.01)
        assert result["threshold"] == pytest.approx(0.84)

    def test_severity_field_present(self):
        from src.comparison.reporting import check_malit_non_regression
        baselines = {"m": {"seed_metrics": {42: {"f1": 0.85}}}}
        result = check_malit_non_regression(0.90, baselines)
        assert "severity" in result
        assert result["severity"] in ("pass", "warning", "fail")


# ── cudnn.benchmark ───────────────────────────────────────────────────────────

class TestCudnnBenchmark:
    def test_get_device_sets_cudnn_benchmark_on_cuda(self, monkeypatch):
        import torch
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda i: "FakeGPU")

        class FakeProps:
            total_memory = int(8e9)

        monkeypatch.setattr(
            torch.cuda, "get_device_properties", lambda i: FakeProps()
        )
        old_val = torch.backends.cudnn.benchmark
        try:
            from src.utils.gpu import get_device
            get_device()
            assert torch.backends.cudnn.benchmark is True
        finally:
            torch.backends.cudnn.benchmark = old_val

    def test_get_device_no_cudnn_benchmark_on_cpu(self, monkeypatch):
        import torch
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        if hasattr(torch.backends, "mps"):
            monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
        # benchmark should not be set to True by CPU path
        torch.backends.cudnn.benchmark = False
        from src.utils.gpu import get_device
        get_device()
        assert torch.backends.cudnn.benchmark is False


# ── Baseline hparams.json ─────────────────────────────────────────────────────

class TestBaselineHparams:
    def _run_minimal(self, tmp_path: Path):
        from PIL import Image
        from src.baselines.runner import BaselineRunner
        from src.data.pipeline import DataPipeline
        from src.utils.save import ArtifactSaver

        root = tmp_path / "data"
        for cls in ("Uninfected", "Parasitized"):
            d = root / cls
            d.mkdir(parents=True)
            for i in range(8):
                arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                Image.fromarray(arr).save(d / f"img_{i:04d}.png")

        loaders = DataPipeline(
            root=root, image_size=64, batch_size=4, num_workers=0,
            deduplicate_data=False,
        ).prepare().get_loaders()

        saver = ArtifactSaver(tmp_path)
        runner = BaselineRunner(
            *loaders, saver=saver, device="cpu", epochs=1, amp=False,
            patience=99, seeds=[42], pretrained=False,
        )
        runner.run(["efficientnet"])
        return tmp_path

    def test_hparams_json_created(self, tmp_path):
        base = self._run_minimal(tmp_path)
        hparams_path = base / "results" / "baselines" / "hparams.json"
        assert hparams_path.exists(), "hparams.json not found"

    def test_hparams_contains_lr(self, tmp_path):
        base = self._run_minimal(tmp_path)
        hparams = json.loads((base / "results" / "baselines" / "hparams.json").read_text())
        assert "lr" in hparams
        assert hparams["lr"] == pytest.approx(3e-4)

    def test_hparams_contains_seeds(self, tmp_path):
        base = self._run_minimal(tmp_path)
        hparams = json.loads((base / "results" / "baselines" / "hparams.json").read_text())
        assert "seeds" in hparams
        assert 42 in hparams["seeds"]

    def test_hparams_contains_note(self, tmp_path):
        base = self._run_minimal(tmp_path)
        hparams = json.loads((base / "results" / "baselines" / "hparams.json").read_text())
        assert "note" in hparams


# ── Reproducibility files ─────────────────────────────────────────────────────

class TestReproducibilityFiles:
    @pytest.fixture
    def comparison_results(self):
        labels = [i % 2 for i in range(100)]
        np.random.seed(7)
        bad = list(np.random.randint(0, 2, 100))
        malit = {
            "seed_preds":   {42: list(labels), 123: list(labels)},
            "seed_metrics": {42: {"f1": 0.99}, 123: {"f1": 0.98}},
        }
        baseline = {
            "efficientnet": {
                "seed_preds":   {42: bad, 123: bad},
                "seed_metrics": {42: {"f1": 0.50}, 123: {"f1": 0.51}},
                "seed_labels":  labels,
            }
        }
        from src.comparison.protocol import ComparisonProtocol
        protocol = ComparisonProtocol(malit, baseline, labels)
        results = protocol.compare_all()
        return protocol, results

    def test_reproducibility_files_created(self, comparison_results, tmp_path):
        protocol, results = comparison_results
        protocol.save_reproducibility(results, tmp_path)
        stats_dir = tmp_path / "results" / "statistics"
        for fname in ("mcnemar_per_seed.json", "mcnemar_aggregated.json", "tost_results.json"):
            assert (stats_dir / fname).exists(), f"{fname} not found"

    def test_mcnemar_per_seed_json_valid(self, comparison_results, tmp_path):
        protocol, results = comparison_results
        protocol.save_reproducibility(results, tmp_path)
        data = json.loads(
            (tmp_path / "results" / "statistics" / "mcnemar_per_seed.json").read_text()
        )
        assert "efficientnet" in data

    def test_tost_results_json_valid(self, comparison_results, tmp_path):
        protocol, results = comparison_results
        protocol.save_reproducibility(results, tmp_path)
        data = json.loads(
            (tmp_path / "results" / "statistics" / "tost_results.json").read_text()
        )
        assert "efficientnet" in data

    def test_all_files_are_valid_json(self, comparison_results, tmp_path):
        protocol, results = comparison_results
        protocol.save_reproducibility(results, tmp_path)
        stats_dir = tmp_path / "results" / "statistics"
        for fname in ("mcnemar_per_seed.json", "mcnemar_aggregated.json", "tost_results.json"):
            content = (stats_dir / fname).read_text()
            json.loads(content)  # must not raise


# ── ComparisonProtocol multi-seed integration ─────────────────────────────────

class TestComparisonMultiSeedIntegration:
    @pytest.fixture
    def protocol_and_results(self, labels_100, perfect_preds, random_preds):
        from src.comparison.protocol import ComparisonProtocol
        malit = {
            "seed_preds":   {42: perfect_preds, 123: perfect_preds},
            "seed_metrics": {42: {"f1": 0.99}, 123: {"f1": 0.98}},
        }
        baseline = {
            "weak": {
                "seed_preds":   {42: random_preds, 123: random_preds},
                "seed_metrics": {42: {"f1": 0.50}, 123: {"f1": 0.51}},
                "seed_labels":  labels_100,
            }
        }
        protocol = ComparisonProtocol(malit, baseline, labels_100)
        return protocol.compare_all()

    def test_result_contains_multi_seed_mcnemar(self, protocol_and_results):
        assert "multi_seed_mcnemar" in protocol_and_results["weak"]

    def test_multi_seed_aggregated_present(self, protocol_and_results):
        ms = protocol_and_results["weak"]["multi_seed_mcnemar"]
        assert "aggregated" in ms
        assert "per_seed" in ms

    def test_superior_verdict_uses_majority_vote(self, protocol_and_results):
        assert protocol_and_results["weak"]["verdict"] == "Superior"

    def test_tost_indeterminate_not_equivalent_verdict(self, labels_100, perfect_preds):
        from src.comparison.protocol import ComparisonProtocol
        # Identical models → TOST indeterminate → should NOT be "Equivalent"
        malit = {
            "seed_preds":   {42: perfect_preds},
            "seed_metrics": {42: {"f1": 0.99}},
        }
        baseline = {
            "copy": {
                "seed_preds":   {42: perfect_preds},
                "seed_metrics": {42: {"f1": 0.99}},
                "seed_labels":  labels_100,
            }
        }
        protocol = ComparisonProtocol(malit, baseline, labels_100)
        results = protocol.compare_all()
        # TOST zero-variance → indeterminate → verdict cannot be "Equivalent"
        assert results["copy"]["tost"].get("indeterminate") is True
        assert results["copy"]["verdict"] != "Equivalent"
