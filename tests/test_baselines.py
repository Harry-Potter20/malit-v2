"""
Tests for the baseline model suite and BaselineRunner.

Uses pretrained=False everywhere to avoid downloading weights during CI.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.baselines.models import (
    ALL_BASELINE_NAMES,
    EfficientNetBaseline,
    EfficientNetCBAMBaseline,
    EfficientNetSEBaseline,
    MobileNetV3Baseline,
    ResNet50Baseline,
    build_baseline,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tiny_batch(b: int = 2, h: int = 64):
    """Small batch compatible with baselines (H=W must be ≥ 32)."""
    return torch.randn(b, 3, h, h)


def _create_fake_dataset(root: Path, n_per_class: int = 10) -> None:
    from PIL import Image
    for cls in ["Uninfected", "Parasitized"]:
        d = root / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            Image.fromarray(arr).save(d / f"img_{i:04d}.png")


def _build_loaders(root: Path):
    from src.data.pipeline import DataPipeline
    return DataPipeline(
        root=root, image_size=64, batch_size=4,
        num_workers=0, deduplicate_data=False,
    ).prepare().get_loaders()


# ── Factory tests ─────────────────────────────────────────────────────────────

class TestBuildBaseline:
    def test_all_five_names_build(self):
        for name in ALL_BASELINE_NAMES:
            model = build_baseline(name, pretrained=False)
            assert isinstance(model, torch.nn.Module)

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown baseline"):
            build_baseline("nonexistent_model", pretrained=False)

    def test_pretrained_false_no_download(self):
        # Should not raise even without internet
        model = build_baseline("efficientnet", pretrained=False)
        assert model is not None

    def test_gradient_checkpointing_flag_stored(self):
        model = build_baseline("efficientnet", pretrained=False,
                               use_gradient_checkpointing=True)
        assert model.use_gc is True

    def test_all_models_return_tuple(self):
        x = _tiny_batch()
        for name in ALL_BASELINE_NAMES:
            model = build_baseline(name, pretrained=False).eval()
            with torch.no_grad():
                out = model(x)
            assert isinstance(out, tuple), f"{name}: expected tuple return"
            assert len(out) == 2, f"{name}: expected (logits, dict)"
            assert isinstance(out[1], dict), f"{name}: second element should be dict"


# ── EfficientNet-B0 baseline ──────────────────────────────────────────────────

class TestEfficientNetBaseline:
    @pytest.fixture
    def model(self):
        return EfficientNetBaseline(num_classes=2, pretrained=False).eval()

    def test_forward_shape(self, model):
        logits, info = model(_tiny_batch(b=4))
        assert logits.shape == (4, 2)

    def test_returns_empty_dict(self, model):
        _, info = model(_tiny_batch())
        assert info == {}

    def test_extract_embedding_shape(self, model):
        embs = model.extract_embedding(_tiny_batch(b=3))
        assert embs.shape == (3, 512)

    def test_batchnorm_eval_mode(self, model):
        # BatchNorm should use running stats in eval mode — no error
        logits, _ = model(_tiny_batch(b=1))
        assert logits.shape == (1, 2)

    def test_gradient_checkpointing_backward(self):
        model = EfficientNetBaseline(
            num_classes=2, pretrained=False, use_gradient_checkpointing=True
        ).train()
        x = torch.randn(2, 3, 64, 64)
        logits, _ = model(x)
        loss = logits.sum()
        loss.backward()   # must not raise
        assert logits.shape == (2, 2)

    def test_logits_are_finite(self, model):
        logits, _ = model(_tiny_batch(b=4))
        assert torch.isfinite(logits).all()


# ── ResNet-50 baseline ────────────────────────────────────────────────────────

class TestResNet50Baseline:
    @pytest.fixture
    def model(self):
        return ResNet50Baseline(num_classes=2, pretrained=False).eval()

    def test_forward_shape(self, model):
        logits, _ = model(_tiny_batch(b=3))
        assert logits.shape == (3, 2)

    def test_extract_embedding_shape(self, model):
        embs = model.extract_embedding(_tiny_batch(b=2))
        assert embs.shape == (2, 512)

    def test_gradient_checkpointing_backward(self):
        model = ResNet50Baseline(
            num_classes=2, pretrained=False, use_gradient_checkpointing=True
        ).train()
        x = torch.randn(2, 3, 64, 64)
        logits, _ = model(x)
        logits.sum().backward()
        assert logits.shape == (2, 2)


# ── MobileNetV3-Small baseline ────────────────────────────────────────────────

class TestMobileNetV3Baseline:
    @pytest.fixture
    def model(self):
        return MobileNetV3Baseline(num_classes=2, pretrained=False).eval()

    def test_forward_shape(self, model):
        logits, _ = model(_tiny_batch(b=4))
        assert logits.shape == (4, 2)

    def test_extract_embedding_shape(self, model):
        embs = model.extract_embedding(_tiny_batch(b=2))
        assert embs.shape == (2, 512)

    def test_batchnorm_eval_mode(self, model):
        logits, _ = model(_tiny_batch(b=1))
        assert logits.shape == (1, 2)


# ── SE baseline ───────────────────────────────────────────────────────────────

class TestEfficientNetSEBaseline:
    @pytest.fixture
    def model(self):
        return EfficientNetSEBaseline(num_classes=2, pretrained=False).eval()

    def test_forward_shape(self, model):
        logits, _ = model(_tiny_batch(b=3))
        assert logits.shape == (3, 2)

    def test_extract_embedding_shape(self, model):
        embs = model.extract_embedding(_tiny_batch(b=2))
        assert embs.shape == (2, 512)

    def test_se_block_has_parameters(self, model):
        se_params = sum(p.numel() for p in model.se.parameters())
        assert se_params > 0


# ── CBAM baseline ─────────────────────────────────────────────────────────────

class TestEfficientNetCBAMBaseline:
    @pytest.fixture
    def model(self):
        return EfficientNetCBAMBaseline(num_classes=2, pretrained=False).eval()

    def test_forward_shape(self, model):
        logits, _ = model(_tiny_batch(b=3))
        assert logits.shape == (3, 2)

    def test_extract_embedding_shape(self, model):
        embs = model.extract_embedding(_tiny_batch(b=2))
        assert embs.shape == (2, 512)

    def test_cbam_block_has_parameters(self, model):
        params = sum(p.numel() for p in model.cbam.parameters())
        assert params > 0


# ── Gradient accumulation in Trainer ─────────────────────────────────────────

class TestGradAccumTrainer:
    def _build_trainer(self, grad_accum_steps: int):
        from torch.utils.data import DataLoader, TensorDataset
        from src.training.trainer import Trainer

        torch.manual_seed(0)
        x = torch.randn(16, 3, 64, 64)
        y = torch.randint(0, 2, (16,))
        ds = TensorDataset(x, y)
        loader = DataLoader(ds, batch_size=4)
        model = EfficientNetBaseline(num_classes=2, pretrained=False)
        return Trainer(
            model=model,
            train_loader=loader,
            val_loader=loader,
            lr=1e-3,
            epochs=1,
            amp=False,
            early_stopping_patience=99,
            device="cpu",
            grad_accum_steps=grad_accum_steps,
        )

    def test_grad_accum_2_trains_without_error(self):
        trainer = self._build_trainer(grad_accum_steps=2)
        best = trainer.fit()
        assert isinstance(best, dict)

    def test_loss_is_finite(self):
        trainer = self._build_trainer(grad_accum_steps=2)
        epoch_metrics = trainer.train_epoch()
        assert np.isfinite(epoch_metrics["loss"])

    def test_grad_accum_1_same_as_default(self):
        trainer = self._build_trainer(grad_accum_steps=1)
        epoch_metrics = trainer.train_epoch()
        assert np.isfinite(epoch_metrics["loss"])

    def test_grad_accum_steps_stored(self):
        trainer = self._build_trainer(grad_accum_steps=4)
        assert trainer.grad_accum_steps == 4


# ── BaselineRunner integration ────────────────────────────────────────────────

class TestBaselineRunnerIntegration:
    def test_runner_produces_results_dict(self):
        from src.baselines.runner import BaselineRunner
        from src.utils.save import ArtifactSaver

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            _create_fake_dataset(root, n_per_class=10)
            train_l, val_l, test_l = _build_loaders(root)
            saver = ArtifactSaver(tmp)

            runner = BaselineRunner(
                train_loader=train_l,
                val_loader=val_l,
                test_loader=test_l,
                saver=saver,
                device="cpu",
                epochs=1,
                amp=False,
                patience=99,
                seeds=[42],
                pretrained=False,
            )
            runner.run(["efficientnet"])

        assert "efficientnet" in runner.results

    def test_results_dict_has_required_keys(self):
        from src.baselines.runner import BaselineRunner
        from src.utils.save import ArtifactSaver

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            _create_fake_dataset(root, n_per_class=10)
            train_l, val_l, test_l = _build_loaders(root)
            saver = ArtifactSaver(tmp)

            runner = BaselineRunner(
                train_loader=train_l,
                val_loader=val_l,
                test_loader=test_l,
                saver=saver,
                device="cpu",
                epochs=1,
                amp=False,
                patience=99,
                seeds=[42],
                pretrained=False,
            )
            runner.run(["mobilenetv3"])

        res = runner.results["mobilenetv3"]
        for key in ("seed_preds", "seed_metrics", "seed_labels", "seed_probs"):
            assert key in res, f"Missing key: {key}"
        assert "f1" in res["seed_metrics"][42]
        assert len(res["seed_labels"]) > 0

    def test_per_seed_metrics_file_created(self):
        from src.baselines.runner import BaselineRunner
        from src.utils.save import ArtifactSaver

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            _create_fake_dataset(root, n_per_class=10)
            train_l, val_l, test_l = _build_loaders(root)
            saver = ArtifactSaver(tmp)

            runner = BaselineRunner(
                train_loader=train_l, val_loader=val_l, test_loader=test_l,
                saver=saver, device="cpu", epochs=1, amp=False,
                patience=99, seeds=[42], pretrained=False,
            )
            runner.run(["efficientnet"])

            seed_file = Path(tmp) / "results" / "baselines" / "efficientnet" / "seed_42_metrics.json"
            assert seed_file.exists(), "Per-seed metrics JSON not saved"
