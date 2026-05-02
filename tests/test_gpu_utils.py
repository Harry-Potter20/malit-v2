"""Tests for GPU utility functions."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import torch


class TestDeviceDetection:
    def test_get_device_returns_valid_device(self):
        from src.utils.gpu import get_device
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ("cuda", "mps", "cpu")

    def test_get_device_is_cpu_or_better(self):
        from src.utils.gpu import get_device
        device = get_device()
        # At minimum should be CPU — never None or invalid
        assert device is not None

    def test_get_device_cpu_fallback_when_no_gpu(self):
        from src.utils.gpu import get_device
        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch.backends.mps.is_available", return_value=False):
                device = get_device()
        assert device.type == "cpu"

    def test_get_device_cuda_when_available(self):
        from src.utils.gpu import get_device
        # Mock CUDA availability
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.get_device_name", return_value="Test GPU"):
                with patch("torch.cuda.get_device_properties") as mock_props:
                    mock_props.return_value.total_memory = 8 * 1024**3
                    device = get_device()
        assert device.type == "cuda"


class TestAutoBatchSize:
    def test_returns_positive_int(self):
        from src.utils.gpu import auto_batch_size
        device = torch.device("cpu")
        bs = auto_batch_size(device, base=32)
        assert isinstance(bs, int)
        assert bs > 0

    def test_cpu_returns_base(self):
        from src.utils.gpu import auto_batch_size
        bs = auto_batch_size(torch.device("cpu"), base=32)
        assert bs == 32

    def test_large_gpu_gets_large_batch(self):
        from src.utils.gpu import auto_batch_size
        with patch("torch.cuda.is_available", return_value=True):
            with patch("src.utils.gpu.gpu_memory_gb", return_value=40.0):
                bs = auto_batch_size(torch.device("cuda"), base=32)
        assert bs >= 128

    def test_small_gpu_gets_small_batch(self):
        from src.utils.gpu import auto_batch_size
        with patch("torch.cuda.is_available", return_value=True):
            with patch("src.utils.gpu.gpu_memory_gb", return_value=3.0):
                bs = auto_batch_size(torch.device("cuda"), base=32)
        assert bs <= 8

    def test_auto_batch_size_increases_with_vram(self):
        """More VRAM should yield larger or equal batch size."""
        from src.utils.gpu import auto_batch_size
        sizes = []
        for vram in [4.0, 8.0, 16.0, 24.0, 40.0]:
            with patch("torch.cuda.is_available", return_value=True):
                with patch("src.utils.gpu.gpu_memory_gb", return_value=vram):
                    sizes.append(auto_batch_size(torch.device("cuda"), base=8))
        # Must be non-decreasing
        assert sizes == sorted(sizes)


class TestEnvironmentPaths:
    def test_get_dataset_path_default(self):
        from src.utils.gpu import get_dataset_path
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DATASET_PATH", None)
            path = get_dataset_path()
        assert isinstance(path, str)
        assert len(path) > 0

    def test_get_dataset_path_from_env(self):
        from src.utils.gpu import get_dataset_path
        with patch.dict(os.environ, {"DATASET_PATH": "/custom/path"}):
            path = get_dataset_path()
        assert path == "/custom/path"

    def test_get_output_dir_default(self):
        from src.utils.gpu import get_output_dir
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OUTPUT_DIR", None)
            out = get_output_dir()
        assert isinstance(out, str)

    def test_get_output_dir_from_env(self):
        from src.utils.gpu import get_output_dir
        with patch.dict(os.environ, {"OUTPUT_DIR": "/kaggle/working/results"}):
            out = get_output_dir()
        assert out == "/kaggle/working/results"


class TestKaggleDetection:
    def test_is_kaggle_returns_bool(self):
        from src.utils.gpu import is_kaggle
        result = is_kaggle()
        assert isinstance(result, bool)

    def test_is_kaggle_false_in_test_env(self):
        from src.utils.gpu import is_kaggle
        # In test env, /kaggle/input doesn't exist and KAGGLE env var isn't set
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("KAGGLE_KERNEL_RUN_TYPE", None)
            with patch("src.utils.gpu._KAGGLE_MARKER") as mock_marker:
                mock_marker.exists.return_value = False
                result = is_kaggle()
        assert result is False

    def test_is_kaggle_true_with_env_var(self):
        from src.utils.gpu import is_kaggle
        with patch.dict(os.environ, {"KAGGLE_KERNEL_RUN_TYPE": "Batch"}):
            result = is_kaggle()
        assert result is True


class TestGPUMemory:
    def test_gpu_memory_gb_non_negative(self):
        from src.utils.gpu import gpu_memory_gb
        mem = gpu_memory_gb()
        assert mem >= 0.0

    def test_gpu_memory_gb_zero_on_cpu(self):
        from src.utils.gpu import gpu_memory_gb
        with patch("torch.cuda.is_available", return_value=False):
            mem = gpu_memory_gb()
        assert mem == 0.0

    def test_log_gpu_memory_no_crash_on_cpu(self):
        from src.utils.gpu import log_gpu_memory
        with patch("torch.cuda.is_available", return_value=False):
            log_gpu_memory("test")  # Should not raise


class TestKaggleValidation:
    def test_validate_kaggle_missing_dataset_raises(self, tmp_path):
        from src.utils.gpu import validate_kaggle_environment
        with patch.dict(os.environ, {"DATASET_PATH": str(tmp_path / "nonexistent")}):
            with pytest.raises(RuntimeError, match="Dataset not found"):
                validate_kaggle_environment(require_gpu=False)

    def test_validate_kaggle_no_gpu_raises_when_required(self, tmp_path):
        from src.utils.gpu import validate_kaggle_environment
        # Create a fake dataset path
        fake_ds = tmp_path / "dataset"
        fake_ds.mkdir()
        with patch.dict(os.environ, {"DATASET_PATH": str(fake_ds), "OUTPUT_DIR": str(tmp_path)}):
            with patch("torch.cuda.is_available", return_value=False):
                with pytest.raises(RuntimeError, match="GPU not available"):
                    validate_kaggle_environment(require_gpu=True)

    def test_validate_kaggle_passes_with_dataset(self, tmp_path):
        from src.utils.gpu import validate_kaggle_environment
        fake_ds = tmp_path / "dataset"
        fake_ds.mkdir()
        with patch.dict(os.environ, {"DATASET_PATH": str(fake_ds), "OUTPUT_DIR": str(tmp_path / "out")}):
            with patch("torch.cuda.is_available", return_value=False):
                checks = validate_kaggle_environment(require_gpu=False)
        assert checks["dataset_exists"] is True
        assert checks["output_writable"] is True
