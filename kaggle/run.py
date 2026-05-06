"""
Kaggle entrypoint for MALIT V2.

This is the ONLY script Kaggle executes.  It:
  1. Validates the GPU and dataset are available
  2. Sets DATASET_PATH and OUTPUT_DIR environment variables
  3. Installs any missing packages from requirements.txt
  4. Calls scripts/run_full_pipeline.py  (MALIT V2 + visualizations)
  5. Calls scripts/run_baselines.py      (5 baselines + comparison)
  6. Calls scripts/run_ablation.py       (4 ablation variants + table)
  7. Zips all results to /kaggle/working/malit_results.zip for download

SECURITY: This file MUST NOT contain any API keys or credentials.
"""

from __future__ import annotations

import os
import subprocess
import sys
import zipfile
from pathlib import Path

# ── Constants (Kaggle standard paths) ────────────────────────────────────────
KAGGLE_INPUT = Path("/kaggle/input")
OUTPUT_DIR = Path("/kaggle/working/results")

# Kaggle mounts datasets at different paths depending on how they were added.
# Search common locations and pick the first that contains PNG files.
_DATASET_CANDIDATES = [
    KAGGLE_INPUT / "cell-images-for-detecting-malaria",
    KAGGLE_INPUT / "cell-images-for-detecting-malaria" / "cell_images",
    KAGGLE_INPUT / "datasets" / "iarunava" / "cell-images-for-detecting-malaria" / "cell_images",
    KAGGLE_INPUT / "iarunava" / "cell-images-for-detecting-malaria" / "cell_images",
]


def _find_dataset() -> Path | None:
    for candidate in _DATASET_CANDIDATES:
        if candidate.exists() and any(candidate.rglob("*.png")):
            return candidate
    # Last-resort: walk /kaggle/input looking for a dir containing Parasitized/ + Uninfected/
    for p in KAGGLE_INPUT.rglob("Parasitized"):
        parent = p.parent
        if (parent / "Uninfected").exists():
            return parent
    return None

# ── Validation ────────────────────────────────────────────────────────────────

def _abort(msg: str) -> None:
    print(f"[MALIT FATAL] {msg}", file=sys.stderr)
    sys.exit(1)


def _check_dataset() -> Path:
    dataset_path = _find_dataset()
    if dataset_path is None:
        tried = "\n  ".join(str(c) for c in _DATASET_CANDIDATES)
        _abort(
            f"Dataset not found. Tried:\n  {tried}\n"
            "Actual path on disk: run  !find /kaggle/input -name '*.png' | head -3  to locate it."
        )
    n = sum(1 for _ in dataset_path.rglob("*.png"))
    print(f"[MALIT] Dataset found: {n} images at {dataset_path}")
    return dataset_path


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

    # Verify critical packages are importable; install individually if missing
    for pkg, import_name in [("imagehash", "imagehash"), ("tqdm", "tqdm"), ("statsmodels", "statsmodels")]:
        try:
            __import__(import_name)
        except ImportError:
            print(f"[MALIT] {pkg} missing — installing…")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", pkg],
                check=True,
            )


def _setup_env(dataset_path: Path) -> None:
    os.environ["DATASET_PATH"] = str(dataset_path)
    os.environ["OUTPUT_DIR"] = str(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[MALIT] DATASET_PATH = {dataset_path}")
    print(f"[MALIT] OUTPUT_DIR   = {OUTPUT_DIR}")


# ── Results packaging ────────────────────────────────────────────────────────

def _zip_results() -> None:
    """Zip all results into a single file for easy Kaggle output download."""
    zip_path = Path("/kaggle/working/malit_results.zip")
    results_dir = OUTPUT_DIR  # /kaggle/working/results

    if not results_dir.exists():
        print("[MALIT] No results directory to zip.", file=sys.stderr)
        return

    print(f"[MALIT] Zipping results → {zip_path}")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in sorted(results_dir.rglob("*")):
            if file.is_file():
                zf.write(file, file.relative_to(results_dir.parent))

    size_mb = zip_path.stat().st_size / 1_000_000
    print(f"[MALIT] malit_results.zip ready ({size_mb:.1f} MB) — download from Kaggle Output tab")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  MALIT V2 — Kaggle GPU Execution")
    print("=" * 60)

    dataset_path = _check_dataset()
    _check_gpu()
    _install_extras()
    _setup_env(dataset_path)

    # Locate project root (kaggle/ is one level below root)
    project_root = Path(__file__).resolve().parent.parent
    pipeline_script  = project_root / "scripts" / "run_full_pipeline.py"
    baselines_script = project_root / "scripts" / "run_baselines.py"
    ablation_script  = project_root / "scripts" / "run_ablation.py"

    for script in (pipeline_script, baselines_script, ablation_script):
        if not script.exists():
            _abort(f"Script not found: {script}")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)

    print("[MALIT] Step 1/2 — Training MALIT V2…")
    result = subprocess.run(
        [sys.executable, str(pipeline_script)],
        cwd=str(project_root),
        env=env,
        check=False,
    )
    if result.returncode != 0:
        _abort(f"MALIT pipeline failed with exit code {result.returncode}")

    print("[MALIT] Step 2/3 — Training baselines and running comparison…")
    result = subprocess.run(
        [sys.executable, str(baselines_script)],
        cwd=str(project_root),
        env=env,
        check=False,
    )
    if result.returncode != 0:
        _abort(f"Baselines pipeline failed with exit code {result.returncode}")

    print("[MALIT] Step 3/3 — Ablation study (component contribution analysis)…")
    result = subprocess.run(
        [sys.executable, str(ablation_script)],
        cwd=str(project_root),
        env=env,
        check=False,
    )
    if result.returncode != 0:
        _abort(f"Ablation study failed with exit code {result.returncode}")

    _zip_results()
    print("[MALIT] All done. Results at", OUTPUT_DIR)


if __name__ == "__main__":
    main()
