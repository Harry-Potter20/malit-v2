"""
Kaggle entrypoint — ablation study only.

Use this when MALIT V2 has already been trained (results/statistics/
seed_*_metrics.json already exists) and you only want to run the four
component-ablation variants (no_gabor, no_lcci, no_attention,
no_multiscale) plus comparison table.

Steps:
  1. Validates GPU and dataset
  2. Sets DATASET_PATH / OUTPUT_DIR environment variables
  3. Installs missing packages
  4. Calls scripts/run_ablation.py
  5. Zips all results to /kaggle/working/malit_results.zip

To use in Kaggle notebook:
    import subprocess, sys, os
    os.chdir("/kaggle/working/malit-v2")
    subprocess.run([sys.executable, "kaggle/run_ablation_only.py"], check=True)

SECURITY: This file MUST NOT contain any API keys or credentials.
"""

from __future__ import annotations

import os
import subprocess
import sys
import zipfile
from pathlib import Path

KAGGLE_INPUT = Path("/kaggle/input")
OUTPUT_DIR   = Path("/kaggle/working/results")

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
    for p in KAGGLE_INPUT.rglob("Parasitized"):
        parent = p.parent
        if (parent / "Uninfected").exists():
            return parent
    return None


def _abort(msg: str) -> None:
    print(f"[MALIT FATAL] {msg}", file=sys.stderr)
    sys.exit(1)


def _check_dataset() -> Path:
    dataset_path = _find_dataset()
    if dataset_path is None:
        tried = "\n  ".join(str(c) for c in _DATASET_CANDIDATES)
        _abort(f"Dataset not found. Tried:\n  {tried}")
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


def _check_malit_results() -> None:
    stats_dir = OUTPUT_DIR / "results" / "statistics"
    seed_files = list(stats_dir.glob("seed_*_metrics.json")) if stats_dir.exists() else []
    if not seed_files:
        _abort(
            f"No MALIT results found in {stats_dir}.\n"
            "Run the full pipeline first (kaggle/run.py or run_full_pipeline.py)."
        )
    # Check predictions are present
    import json
    for f in seed_files:
        data = json.loads(f.read_text())
        if not data.get("predictions"):
            _abort(
                f"No predictions in {f.name}. "
                "Re-run run_full_pipeline.py to regenerate with the latest code."
            )
    print(f"[MALIT] Found {len(seed_files)} MALIT seed result(s) — ready for ablation.")


def _install_extras() -> None:
    req = Path(__file__).parent / "requirements.txt"
    if req.exists():
        print("[MALIT] Installing extra requirements…")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(req)],
            check=True,
        )
    for pkg, import_name in [("imagehash", "imagehash"), ("tqdm", "tqdm"), ("statsmodels", "statsmodels")]:
        try:
            __import__(import_name)
        except ImportError:
            print(f"[MALIT] {pkg} missing — installing…")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)


def _setup_env(dataset_path: Path) -> None:
    os.environ["DATASET_PATH"] = str(dataset_path)
    os.environ["OUTPUT_DIR"]   = str(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[MALIT] DATASET_PATH = {dataset_path}")
    print(f"[MALIT] OUTPUT_DIR   = {OUTPUT_DIR}")


def _zip_results() -> None:
    zip_path = Path("/kaggle/working/malit_results.zip")
    if not OUTPUT_DIR.exists():
        return
    print(f"[MALIT] Zipping results → {zip_path}")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in sorted(OUTPUT_DIR.rglob("*")):
            if file.is_file():
                zf.write(file, file.relative_to(OUTPUT_DIR.parent))
    size_mb = zip_path.stat().st_size / 1_000_000
    print(f"[MALIT] malit_results.zip ready ({size_mb:.1f} MB) — download from Kaggle Output tab")


def main() -> None:
    print("=" * 60)
    print("  MALIT V2 — Ablation Study Only")
    print("=" * 60)

    dataset_path = _check_dataset()
    _check_gpu()
    _check_malit_results()
    _install_extras()
    _setup_env(dataset_path)

    project_root    = Path(__file__).resolve().parent.parent
    ablation_script = project_root / "scripts" / "run_ablation.py"

    if not ablation_script.exists():
        _abort(f"Script not found: {ablation_script}")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)

    print("[MALIT] Running ablation study…")
    result = subprocess.run(
        [sys.executable, str(ablation_script)],
        cwd=str(project_root),
        env=env,
        check=False,
    )
    if result.returncode != 0:
        _abort(f"Ablation script failed with exit code {result.returncode}")

    _zip_results()
    print("[MALIT] Done. Download malit_results.zip from the Kaggle Output tab.")


if __name__ == "__main__":
    main()
