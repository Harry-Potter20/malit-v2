from __future__ import annotations

import logging
import os
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

_KAGGLE_MARKER = Path("/kaggle/input")
_DEFAULT_DATASET_PATH = "data/cell_images"
_DEFAULT_OUTPUT_DIR = "results"


# ── Runtime detection ────────────────────────────────────────────────────────

def is_kaggle() -> bool:
    """Detect whether code is running inside a Kaggle kernel."""
    return _KAGGLE_MARKER.exists() or os.getenv("KAGGLE_KERNEL_RUN_TYPE") is not None


def get_device() -> torch.device:
    """Return best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info("GPU detected: %s  (%.1f GB VRAM)", name, mem_gb)
        torch.backends.cudnn.benchmark = True
        return device
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Apple MPS detected.")
        return torch.device("mps")
    logger.warning("No GPU detected — using CPU. Training will be slow.")
    return torch.device("cpu")


# ── Environment-driven paths ─────────────────────────────────────────────────

def get_dataset_path() -> str:
    """
    Return dataset root from DATASET_PATH env var or sensible default.
    On Kaggle: /kaggle/input/cell-images-for-detecting-malaria
    Local:     data/cell_images
    """
    path = os.getenv("DATASET_PATH", _DEFAULT_DATASET_PATH)
    if not Path(path).exists():
        logger.warning("Dataset path '%s' does not exist.", path)
    return path


def get_output_dir() -> str:
    """Return output root from OUTPUT_DIR env var or default 'results'."""
    return os.getenv("OUTPUT_DIR", _DEFAULT_OUTPUT_DIR)


# ── GPU memory utilities ─────────────────────────────────────────────────────

def gpu_memory_gb() -> float:
    """Total VRAM in GB (0.0 if no CUDA device)."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / 1e9


def gpu_memory_used_gb() -> float:
    """Currently allocated VRAM in GB."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated(0) / 1e9


def log_gpu_memory(tag: str = "") -> None:
    if not torch.cuda.is_available():
        return
    alloc = gpu_memory_used_gb()
    total = gpu_memory_gb()
    pct = 100 * alloc / (total + 1e-8)
    logger.info("[GPU%s] %.2f / %.2f GB (%.1f%%)", f" {tag}" if tag else "", alloc, total, pct)
    if pct > 90:
        logger.warning("GPU memory >90%% — risk of OOM. Consider reducing batch size.")


# ── Auto batch size ───────────────────────────────────────────────────────────

def auto_batch_size(device: torch.device | None = None, base: int = 32) -> int:
    """
    Estimate optimal batch size from available GPU VRAM.
    Falls back to `base` on non-CUDA devices.
    """
    if device is None:
        device = get_device()

    if device.type != "cuda":
        return base

    mem_gb = gpu_memory_gb()
    if mem_gb >= 40:   # A100 / H100
        return 256
    if mem_gb >= 24:   # 3090 / 4090 / A6000
        return 128
    if mem_gb >= 16:   # V100 16G / T4 16G
        return 64
    if mem_gb >= 8:    # T4 / RTX 3070
        return 32
    if mem_gb >= 4:    # GTX 1650 / etc
        return 16
    return 8


# ── Kaggle environment validation ────────────────────────────────────────────

def validate_kaggle_environment(require_gpu: bool = True) -> dict[str, bool]:
    """
    Validate Kaggle execution prerequisites.
    Returns a dict of checks and raises RuntimeError on critical failures.
    """
    checks: dict[str, bool] = {}

    # Dataset path
    dataset_path = get_dataset_path()
    checks["dataset_exists"] = Path(dataset_path).exists()
    if not checks["dataset_exists"]:
        raise RuntimeError(
            f"CRITICAL: Dataset not found at '{dataset_path}'. "
            "Ensure the dataset is attached in kernel settings."
        )

    # GPU
    checks["gpu_available"] = torch.cuda.is_available()
    if require_gpu and not checks["gpu_available"]:
        raise RuntimeError(
            "CRITICAL: GPU not available. Enable GPU in Kaggle kernel settings."
        )

    # Output dir writable
    out_dir = Path(get_output_dir())
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        test_file = out_dir / ".write_test"
        test_file.write_text("ok")
        test_file.unlink()
        checks["output_writable"] = True
    except OSError:
        checks["output_writable"] = False
        raise RuntimeError(f"CRITICAL: Output directory '{out_dir}' is not writable.")

    logger.info("Kaggle environment validated: %s", checks)
    return checks
