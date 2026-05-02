#!/usr/bin/env python3
"""Download NIH Malaria Cell Images dataset via Kaggle API."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    load_dotenv()
    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")

    if not username or not key or username == "your_username":
        logger.error(
            "Kaggle credentials not set. Edit .env with real KAGGLE_USERNAME and KAGGLE_KEY."
        )
        raise SystemExit(1)

    # Write kaggle.json in expected location (never logged, never committed)
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    kaggle_json = kaggle_dir / "kaggle.json"
    kaggle_json.write_text(f'{{"username":"{username}","key":"{key}"}}')
    kaggle_json.chmod(0o600)

    import kaggle  # noqa: F401 — imported after env setup

    dataset = "iarunava/cell-images-for-detecting-malaria"
    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)

    logger.info("Downloading dataset: %s → %s", dataset, out_dir)
    from kaggle.api.kaggle_api_extended import KaggleApiExtended
    api = KaggleApiExtended()
    api.authenticate()
    api.dataset_download_files(dataset, path=str(out_dir), unzip=True)
    logger.info("Download complete. Files at %s", out_dir)


if __name__ == "__main__":
    main()
