"""
Integration tests: end-to-end pipeline on a synthetic mini-dataset.

These tests verify that all modules work together correctly without
needing real Kaggle data. They run the full training + uncertainty +
CBR + statistics pipeline in quick mode on randomly generated images.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

# ── Helpers ──────────────────────────────────────────────────────────────────

def _create_fake_dataset(root: Path, n_per_class: int = 30) -> None:
    from PIL import Image
    for class_name in ["Uninfected", "Parasitized"]:
        d = root / class_name
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            Image.fromarray(arr).save(d / f"cell_{i:04d}.png")


def _tiny_model():
    from src.models.malit import MALITV2
    return MALITV2(
        num_classes=2,
        gabor_n_orientations=2, gabor_n_scales=2, gabor_kernel_size=7,
        efficientnet_backbone="efficientnet_b0",
        efficientnet_pretrained=False, efficientnet_freeze_blocks=0,
        lcci_reduction=4, attention_reduction=4, ms_rfs=[1, 3],
    )


def _build_loaders(root: Path, image_size: int = 64, batch_size: int = 8):
    from src.data.pipeline import DataPipeline
    return DataPipeline(
        root=root, image_size=image_size, batch_size=batch_size,
        num_workers=0, deduplicate_data=False,
    ).prepare().get_loaders()


# ── Data + Model integration ──────────────────────────────────────────────────

class TestDataModelIntegration:
    def test_dataloader_feeds_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _create_fake_dataset(root, n_per_class=10)
            train_l, _, _ = _build_loaders(root)
            model = _tiny_model().eval()
            imgs, labels = next(iter(train_l))
            with torch.no_grad():
                logits, info = model(imgs)
            assert logits.shape == (len(imgs), 2)
            assert "channel_energy" in info

    def test_embedding_extraction_from_loader(self):
        from src.explainability.case_based_reasoning import EmbeddingExtractor
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _create_fake_dataset(root, n_per_class=10)
            _, _, test_l = _build_loaders(root)
            model = _tiny_model()
            extractor = EmbeddingExtractor(model, device=torch.device("cpu"))
            embs, labels = extractor.extract(test_l)
        assert embs.shape[1] == 512
        assert len(labels) > 0


# ── Trainer integration ───────────────────────────────────────────────────────

class TestTrainerIntegration:
    def test_trainer_full_cycle(self):
        from src.training.trainer import Trainer
        from src.utils.save import ArtifactSaver
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            _create_fake_dataset(root, n_per_class=20)
            train_l, val_l, test_l = _build_loaders(root, batch_size=8)
            model = _tiny_model()
            saver = ArtifactSaver(tmp)
            trainer = Trainer(
                model=model, train_loader=train_l, val_loader=val_l,
                lr=1e-3, epochs=2, amp=False, early_stopping_patience=99,
                device="cpu", saver=saver,
            )
            best = trainer.fit()
            metrics, preds, labels, history = trainer.evaluate(test_l)
        assert "f1" in metrics
        assert "accuracy" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert len(preds) == len(labels)
        assert len(history) == 2

    def test_metrics_saved_after_training(self):
        from src.training.trainer import Trainer
        from src.utils.save import ArtifactSaver
        import json
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            _create_fake_dataset(root, n_per_class=20)
            train_l, val_l, _ = _build_loaders(root, batch_size=8)
            model = _tiny_model()
            saver = ArtifactSaver(tmp)
            trainer = Trainer(
                model=model, train_loader=train_l, val_loader=val_l,
                lr=1e-3, epochs=1, amp=False, device="cpu", saver=saver,
            )
            trainer.fit()
            # Model checkpoint should have been saved
            model_files = list((Path(tmp) / "results" / "models").glob("*.pt"))
        assert len(model_files) >= 1, "No model checkpoint saved after training"


# ── Uncertainty integration ───────────────────────────────────────────────────

class TestUncertaintyIntegration:
    def test_mc_dropout_on_trained_model(self):
        from src.explainability.bayesian_uncertainty import BayesianUncertainty
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            _create_fake_dataset(root, n_per_class=15)
            _, _, test_l = _build_loaders(root, batch_size=8)
            model = _tiny_model()
            mc = BayesianUncertainty(T=3, device=torch.device("cpu"))
            results = mc.run_on_loader(model, test_l)
        assert "mean_probs" in results
        assert "variance" in results
        assert (results["variance"] >= 0).all()
        # mean probs must sum to 1
        assert np.allclose(results["mean_probs"].sum(axis=1), 1.0, atol=1e-4)

    def test_ensemble_confidence_multiple_models(self):
        from src.explainability.ensemble_confidence import EnsembleConfidence
        # Simulate 3 seeds with different predictions
        np.random.seed(0)
        n_samples = 50
        n_classes = 2
        n_seeds = 3
        probs = np.abs(np.random.randn(n_seeds, n_samples, n_classes))
        probs /= probs.sum(axis=2, keepdims=True)
        preds = probs.argmax(axis=2)
        labels = np.random.randint(0, 2, n_samples)
        results = EnsembleConfidence().run(probs, preds, labels)
        assert "agreement" in results
        assert "ci_width" in results
        assert (results["agreement"] >= 0).all() and (results["agreement"] <= 1).all()


# ── CBR integration ───────────────────────────────────────────────────────────

class TestCBRIntegration:
    def test_build_and_query_cbr(self):
        from src.explainability.case_based_reasoning import CaseBasedReasoning
        np.random.seed(42)
        train_embs = np.random.randn(100, 512).astype(np.float32)
        train_labels = np.array([i % 2 for i in range(100)])
        test_embs = np.random.randn(20, 512).astype(np.float32)
        test_labels = np.array([i % 2 for i in range(20)])
        cbr = CaseBasedReasoning(k=5, use_faiss=False)
        cbr.build_index(train_embs, train_labels)
        results = cbr.run_on_embeddings(test_embs, test_labels)
        assert 0.0 <= results["mean_consistency"] <= 1.0
        assert len(results["neighbors"]) == 20

    def test_cbr_full_cycle_with_model(self):
        from src.explainability.case_based_reasoning import CaseBasedReasoning, EmbeddingExtractor
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            _create_fake_dataset(root, n_per_class=20)
            train_l, _, test_l = _build_loaders(root, batch_size=8)
            model = _tiny_model()
            extractor = EmbeddingExtractor(model, torch.device("cpu"))
            train_embs, train_labels = extractor.extract(train_l)
            test_embs, test_labels = extractor.extract(test_l)
            cbr = CaseBasedReasoning(k=3, use_faiss=False)
            cbr.build_index(train_embs, train_labels)
            results = cbr.run_on_embeddings(test_embs, test_labels)
        assert results["mean_consistency"] >= 0.0


# ── Clinical confidence integration ──────────────────────────────────────────

class TestClinicalConfidenceIntegration:
    def test_engine_scores_batch(self):
        from src.explainability.clinical_confidence import ClinicalConfidenceEngine
        engine = ClinicalConfidenceEngine(num_channels=128)
        # Fit on synthetic data
        np.random.seed(0)
        val_logits = np.random.randn(100, 2)
        val_labels = np.random.randint(0, 2, 100)
        engine.fit_calibration(val_logits, val_labels)
        # Score a sample
        logits = np.array([1.5, -0.3])
        attn = np.abs(np.random.randn(128))
        energies = np.abs(np.random.randn(128))
        report = engine.score(logits, attn, energies, [1, 1, 0], cell_id="integration_test")
        assert 0.0 <= report["clinical_confidence_pct"] <= 100.0
        assert report["recommendation"] in ("Accept", "Review", "Reject")

    def test_uncertainty_type_coverage(self):
        """All three uncertainty types can be produced by the classifier."""
        from src.explainability.clinical_confidence import classify_uncertainty
        types = {
            classify_uncertainty(h_norm=0.85, seed_agreement=0.95, dominance_ratio=0.25),
            classify_uncertainty(h_norm=0.3, seed_agreement=0.3, dominance_ratio=0.7),
            classify_uncertainty(h_norm=0.4, seed_agreement=0.95, dominance_ratio=0.6),
        }
        assert "Structural" in types
        assert "Epistemic" in types
        assert "Data ambiguity" in types


# ── Statistics integration ────────────────────────────────────────────────────

class TestStatisticsIntegration:
    def test_mcnemar_plus_tost_pipeline(self):
        from src.statistics.tests import StatisticsReport
        np.random.seed(42)
        n = 100
        labels = list(np.random.randint(0, 2, n))
        seed_preds = {
            42: list(np.random.randint(0, 2, n)),
            123: list(np.random.randint(0, 2, n)),
        }
        seed_metrics = {
            42: {"f1": 0.72, "accuracy": 0.70},
            123: {"f1": 0.74, "accuracy": 0.72},
        }
        report = StatisticsReport(seed_preds, seed_metrics, labels).full_report()
        assert "summary" in report
        assert "mcnemar" in report
        assert "tost" in report
        assert report["summary"]["mean_f1"] == pytest.approx(0.73)

    def test_statistics_report_serializable(self):
        """Report output must be JSON-serializable (for saving)."""
        import json
        from src.statistics.tests import StatisticsReport
        np.random.seed(0)
        n = 50
        labels = list(np.random.randint(0, 2, n))
        seed_preds = {42: list(np.random.randint(0, 2, n))}
        seed_metrics = {42: {"f1": 0.80}}
        report = StatisticsReport(seed_preds, seed_metrics, labels).full_report()
        # Should not raise
        json.dumps(report)


# ── Checkpoint save/load round-trip ──────────────────────────────────────────

class TestCheckpointRoundTrip:
    def test_save_and_reload_weights_only(self):
        """Verify weights_only=True can load a checkpoint saved by ArtifactSaver."""
        import torch
        from src.utils.save import ArtifactSaver
        with tempfile.TemporaryDirectory() as tmp:
            model = _tiny_model()
            saver = ArtifactSaver(tmp)
            saved_path = saver.save_model(model, "malit_v2", seed=42, epoch=1)
            # Must load with weights_only=True (security requirement)
            ckpt = torch.load(saved_path, map_location="cpu", weights_only=True)
        assert "model_state" in ckpt
        assert "seed" in ckpt
        assert ckpt["seed"] == 42

    def test_reloaded_model_produces_same_output(self):
        import torch
        from src.utils.save import ArtifactSaver
        with tempfile.TemporaryDirectory() as tmp:
            model = _tiny_model().eval()
            x = torch.randn(2, 3, 224, 224)
            with torch.no_grad():
                logits_before, _ = model(x)
            saver = ArtifactSaver(tmp)
            saved_path = saver.save_model(model, "malit_v2", seed=0)
            # Reload
            model2 = _tiny_model().eval()
            ckpt = torch.load(saved_path, map_location="cpu", weights_only=True)
            model2.load_state_dict(ckpt["model_state"])
            with torch.no_grad():
                logits_after, _ = model2(x)
        assert torch.allclose(logits_before, logits_after, atol=1e-5), \
            "Model output changed after checkpoint round-trip"
