"""Tests for clinical confidence, bayesian uncertainty, and ensemble confidence."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch


class TestClinicalConfidence:
    def test_compute_entropy_uniform(self):
        from src.explainability.clinical_confidence import compute_entropy
        # Uniform attention → max entropy = log2(C)
        attn = np.ones(16)
        h = compute_entropy(attn)
        assert h == pytest.approx(math.log2(16), abs=0.1)

    def test_compute_entropy_peaked(self):
        from src.explainability.clinical_confidence import compute_entropy
        # All weight on one channel → entropy ≈ 0
        attn = np.zeros(16)
        attn[0] = 100.0
        h = compute_entropy(attn)
        assert h < 0.01

    def test_reliability_is_between_0_and_1(self):
        from src.explainability.clinical_confidence import compute_reliability
        for entropy in [0.0, 1.0, 5.0, 10.0, 100.0]:
            r = compute_reliability(entropy, num_channels=1280)
            assert 0.0 <= r <= 1.0

    def test_ccs_range(self):
        from src.explainability.clinical_confidence import compute_ccs
        assert compute_ccs(0.9, 0.8) == pytest.approx(0.72)
        assert 0.0 <= compute_ccs(1.0, 1.0) <= 1.0
        assert 0.0 <= compute_ccs(0.0, 0.0) <= 1.0

    def test_seed_agreement_all_same(self):
        from src.explainability.clinical_confidence import compute_seed_agreement
        assert compute_seed_agreement([1, 1, 1]) == pytest.approx(1.0)

    def test_seed_agreement_split(self):
        from src.explainability.clinical_confidence import compute_seed_agreement
        assert compute_seed_agreement([1, 0, 1]) == pytest.approx(2 / 3)

    def test_dominance_ratio_range(self):
        from src.explainability.clinical_confidence import compute_dominance_ratio
        energies = np.abs(np.random.randn(1280))
        dr = compute_dominance_ratio(energies)
        assert 0.0 <= dr <= 1.0

    def test_classify_uncertainty_structural(self):
        from src.explainability.clinical_confidence import classify_uncertainty
        result = classify_uncertainty(h_norm=0.85, seed_agreement=0.9, dominance_ratio=0.3)
        assert result == "Structural"

    def test_classify_uncertainty_epistemic(self):
        from src.explainability.clinical_confidence import classify_uncertainty
        result = classify_uncertainty(h_norm=0.3, seed_agreement=0.4, dominance_ratio=0.6)
        assert result == "Epistemic"

    def test_classify_uncertainty_data_ambiguity(self):
        from src.explainability.clinical_confidence import classify_uncertainty
        result = classify_uncertainty(h_norm=0.4, seed_agreement=0.9, dominance_ratio=0.5)
        assert result == "Data ambiguity"

    def test_clinical_report_structure(self):
        from src.explainability.clinical_confidence import generate_clinical_report
        report = generate_clinical_report(
            cell_id="test_001",
            pred_class=1,
            calibrated_prob=0.92,
            reliability=0.85,
            ccs=0.782,
            seed_agreement=1.0,
            n_seeds=3,
            entropy=2.1,
            h_norm=0.19,
            dominance_ratio=0.6,
            uncertainty_type="Data ambiguity",
        )
        assert report["cell_id"] == "test_001"
        assert report["prediction"] == "Parasitized"
        assert "clinical_confidence_pct" in report
        assert "breakdown" in report
        assert "recommendation" in report
        assert report["recommendation"] in ("Accept", "Review", "Reject")

    def test_temperature_scaler_calibrates(self):
        from src.explainability.clinical_confidence import TemperatureScaler
        scaler = TemperatureScaler()
        np.random.seed(0)
        logits = np.random.randn(100, 2)
        labels = np.random.randint(0, 2, 100)
        temp = scaler.fit(logits, labels)
        assert 0.1 <= temp <= 10.0

    def test_calibrated_probs_sum_to_one(self):
        from src.explainability.clinical_confidence import TemperatureScaler
        scaler = TemperatureScaler()
        logits = np.random.randn(10, 2)
        probs = scaler.calibrate(logits)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_engine_score_keys(self):
        from src.explainability.clinical_confidence import ClinicalConfidenceEngine
        engine = ClinicalConfidenceEngine(num_channels=64)
        logits = np.array([1.2, -0.5])
        attn = np.random.rand(64)
        energies = np.abs(np.random.randn(64))
        result = engine.score(logits, attn, energies, [1, 1, 0], cell_id="c1")
        for key in ["prediction", "clinical_confidence_pct", "breakdown", "recommendation"]:
            assert key in result


class TestBayesianUncertainty:
    def test_mc_predict_shapes(self, tiny_model, sample_batch):
        from src.explainability.bayesian_uncertainty import BayesianUncertainty
        mc = BayesianUncertainty(T=5)
        mean, var = mc.mc_dropout_predict(tiny_model, sample_batch)
        assert mean.shape == (8, 2)
        assert var.shape == (8, 2)

    def test_mean_sums_to_one(self, tiny_model, sample_batch):
        from src.explainability.bayesian_uncertainty import BayesianUncertainty
        mc = BayesianUncertainty(T=5)
        mean, _ = mc.mc_dropout_predict(tiny_model, sample_batch)
        assert torch.allclose(mean.sum(dim=1), torch.ones(8), atol=1e-4)

    def test_variance_non_negative(self, tiny_model, sample_batch):
        from src.explainability.bayesian_uncertainty import BayesianUncertainty
        mc = BayesianUncertainty(T=5)
        _, var = mc.mc_dropout_predict(tiny_model, sample_batch)
        assert (var >= 0).all()

    def test_uncertainty_summary_keys(self, tiny_model, sample_loader):
        from src.explainability.bayesian_uncertainty import BayesianUncertainty
        mc = BayesianUncertainty(T=3)
        results = mc.run_on_loader(tiny_model, sample_loader)
        summary = BayesianUncertainty.uncertainty_summary(results)
        assert "mean_variance" in summary
        assert "mean_entropy" in summary


class TestEnsembleConfidence:
    def test_ensemble_stats_shapes(self):
        from src.explainability.ensemble_confidence import EnsembleConfidence
        probs = np.random.dirichlet([1, 1], size=(3, 50))  # (3, 50, 2)
        stats = EnsembleConfidence.ensemble_stats(probs)
        assert stats["mean"].shape == (50, 2)
        assert stats["ci_low"].shape == (50, 2)
        assert stats["ci_high"].shape == (50, 2)

    def test_ci_low_leq_ci_high(self):
        from src.explainability.ensemble_confidence import EnsembleConfidence
        probs = np.random.dirichlet([1, 1], size=(3, 50))
        stats = EnsembleConfidence.ensemble_stats(probs)
        assert (stats["ci_low"] <= stats["ci_high"]).all()

    def test_agreement_score_all_same(self):
        from src.explainability.ensemble_confidence import EnsembleConfidence
        preds = np.ones((3, 50), dtype=int)  # all seeds predict class 1
        agreement = EnsembleConfidence.agreement_score(preds)
        assert np.allclose(agreement, 1.0)

    def test_agreement_score_range(self):
        from src.explainability.ensemble_confidence import EnsembleConfidence
        preds = np.random.randint(0, 2, (3, 50))
        agreement = EnsembleConfidence.agreement_score(preds)
        assert (agreement >= 0).all() and (agreement <= 1).all()

    def test_run_returns_expected_keys(self):
        from src.explainability.ensemble_confidence import EnsembleConfidence
        probs = np.random.dirichlet([1, 1], size=(3, 50))
        preds = probs.argmax(axis=2)
        labels = np.random.randint(0, 2, 50)
        results = EnsembleConfidence().run(probs, preds, labels)
        for key in ["mean_probs", "agreement", "ci_width", "mean_ci_width", "mean_agreement"]:
            assert key in results
