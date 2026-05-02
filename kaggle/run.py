"""
Kaggle entrypoint for MALIT V2.

This is the ONLY script Kaggle executes.  It:
  1. Validates the GPU and dataset are available
  2. Sets DATASET_PATH and OUTPUT_DIR environment variables
  3. Installs any missing packages from requirements.txt
  4. Calls scripts/run_full_pipeline.py

SECURITY: This file MUST NOT contain any API keys or credentials.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# ── Constants (Kaggle standard paths) ────────────────────────────────────────
KAGGLE_INPUT = Path("/kaggle/input")
DATASET_SLUG = "cell-images-for-detecting-malaria"
DATASET_PATH = KAGGLE_INPUT / DATASET_SLUG
OUTPUT_DIR = Path("/kaggle/working/results")

# ── Validation ────────────────────────────────────────────────────────────────

def _abort(msg: str) -> None:
    print(f"[MALIT FATAL] {msg}", file=sys.stderr)
    sys.exit(1)


def _check_dataset() -> None:
    if not DATASET_PATH.exists():
        _abort(
            f"Dataset not found at {DATASET_PATH}. "
            "Attach 'iarunava/cell-images-for-detecting-malaria' in kernel settings."
        )
    n = sum(1 for _ in DATASET_PATH.rglob("*.png"))
    if n == 0:
        _abort(f"Dataset directory {DATASET_PATH} contains no PNG files.")
    print(f"[MALIT] Dataset found: {n} images at {DATASET_PATH}")


def _check_gpu() -> None:
    import torch
    if not torch.cuda.is_available():
        _abort("GPU not available. Enable GPU runtime in Kaggle kernel settings.")
    name = torch.cuda.get_device_name(0)
    mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[MALIT] GPU: {name} ({mem_gb:.1f} GB VRAM)")


def _install_extras() -> None:
    req = Path(__file__).parent / "requirements.txt"
    if req.exists():
        print("[MALIT] Installing extra requirements…")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(req)],
            check=True,
        )


def _setup_env() -> None:
    os.environ["DATASET_PATH"] = str(DATASET_PATH)
    os.environ["OUTPUT_DIR"] = str(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[MALIT] DATASET_PATH = {DATASET_PATH}")
    print(f"[MALIT] OUTPUT_DIR   = {OUTPUT_DIR}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  MALIT V2 — Kaggle GPU Execution")
    print("=" * 60)

    _check_dataset()
    _check_gpu()
    _install_extras()
    _setup_env()

    # Locate project root (kaggle/ is one level below root)
    project_root = Path(__file__).resolve().parent.parent
    pipeline_script = project_root / "scripts" / "run_full_pipeline.py"

    if not pipeline_script.exists():
        _abort(f"Pipeline script not found: {pipeline_script}")

    print("[MALIT] Launching full pipeline…")
    result = subprocess.run(
        [sys.executable, str(pipeline_script)],
        cwd=str(project_root),
        check=False,
    )

    if result.returncode != 0:
        _abort(f"Pipeline failed with exit code {result.returncode}")

    print("[MALIT] Pipeline complete. Results at", OUTPUT_DIR)


if __name__ == "__main__":
    main()
