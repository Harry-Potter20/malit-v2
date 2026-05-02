"""Tests for Trainer, MultiSeedRunner, and EvaluationMetrics."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch


class TestEvaluationMetrics:
    def test_perfect_predictions(self):
        from src.evaluation.metrics import EvaluationMetrics
        labels = [0, 0, 1, 1, 0, 1]
        preds = [0, 0, 1, 1, 0, 1]
        m = EvaluationMetrics.from_predictions(labels, preds)
        assert m["accuracy"] == pytest.approx(1.0)
        assert m["f1"] == pytest.approx(1.0)
        assert m["precision"] == pytest.approx(1.0)
        assert m["recall"] == pytest.approx(1.0)

    def test_all_wrong(self):
        from src.evaluation.metrics import EvaluationMetrics
        labels = [0, 0, 1, 1]
        preds = [1, 1, 0, 0]
        m = EvaluationMetrics.from_predictions(labels, preds)
        assert m["accuracy"] == pytest.approx(0.0)

    def test_auroc_with_probs(self):
        from src.evaluation.metrics import EvaluationMetrics
        labels = [0, 0, 1, 1]
        preds = [0, 0, 1, 1]
        probs = [[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]]
        m = EvaluationMetrics.from_predictions(labels, preds, probs)
        assert m["auroc"] == pytest.approx(1.0)

    def test_ece_perfect_calibration(self):
        from src.evaluation.metrics import EvaluationMetrics
        labels = np.array([0, 1, 0, 1])
        probs = np.array([[0.9, 0.1], [0.1, 0.9], [0.85, 0.15], [0.05, 0.95]])
        ece = EvaluationMetrics.expected_calibration_error(labels, probs)
        assert 0.0 <= ece <= 1.0

    def test_confusion_matrix_binary(self):
        from src.evaluation.metrics import EvaluationMetrics
        labels = [0, 0, 1, 1]
        preds = [0, 1, 0, 1]
        m = EvaluationMetrics.from_predictions(labels, preds)
        assert "confusion_matrix" in m
        cm = np.array(m["confusion_matrix"])
        assert cm.sum() == 4

    def test_sensitivity_specificity(self):
        from src.evaluation.metrics import EvaluationMetrics
        labels = [1, 1, 0, 0]
        preds = [1, 1, 0, 0]
        m = EvaluationMetrics.from_predictions(labels, preds)
        assert m["sensitivity"] == pytest.approx(1.0)
        assert m["specificity"] == pytest.approx(1.0)


class TestTrainer:
    """Integration smoke-test for trainer on synthetic data."""

    def test_trainer_one_epoch(self, tiny_model, sample_loader):
        from src.training.trainer import Trainer
        from src.utils.save import ArtifactSaver
        with tempfile.TemporaryDirectory() as tmp:
            saver = ArtifactSaver(tmp)
            trainer = Trainer(
                model=tiny_model,
                train_loader=sample_loader,
                val_loader=sample_loader,
                lr=1e-3,
                epochs=1,
                amp=False,
                early_stopping_patience=5,
                device="cpu",
                saver=saver,
            )
            best = trainer.fit()
        assert isinstance(best, dict)

    def test_early_stopping_triggers(self):
        from src.training.trainer import EarlyStopping
        import torch.nn as nn
        stopper = EarlyStopping(patience=2, mode="max")
        model = nn.Linear(2, 2)
        # Improve once, then no improvement
        assert not stopper(0.5, model)
        assert not stopper(0.5, model)   # no improvement (counter=1)
        assert stopper(0.5, model)       # no improvement (counter=2 == patience)

    def test_early_stopping_restores_best(self):
        from src.training.trainer import EarlyStopping
        import torch.nn as nn
        stopper = EarlyStopping(patience=3, mode="max")
        model = nn.Linear(2, 2)
        nn.init.constant_(model.weight, 1.0)
        stopper(0.9, model)  # save this as best
        nn.init.constant_(model.weight, 0.0)  # corrupt weights
        stopper(0.5, model)
        stopper.restore_best(model)
        assert torch.allclose(model.weight, torch.ones_like(model.weight))

    def test_trainer_evaluate_returns_metrics(self, tiny_model, sample_loader):
        from src.training.trainer import Trainer
        with tempfile.TemporaryDirectory() as tmp:
            from src.utils.save import ArtifactSaver
            trainer = Trainer(
                model=tiny_model,
                train_loader=sample_loader,
                val_loader=sample_loader,
                lr=1e-3,
                epochs=1,
                amp=False,
                device="cpu",
                saver=ArtifactSaver(tmp),
            )
            trainer.fit()
            metrics, preds, labels, history = trainer.evaluate(sample_loader)
        assert "accuracy" in metrics
        assert "f1" in metrics
        assert len(preds) == 8
        assert len(labels) == 8
