from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_confidence_histogram(
    confidences,
    save_path: str | Path,
    bins: int = 20,
) -> None:
    """Histogram of prediction confidences to visualise uncertainty spread."""
    confidences = np.asarray(confidences)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5))
    plt.hist(confidences, bins=bins, edgecolor="white")
    plt.xlabel("Confidence")
    plt.ylabel("Frequency")
    plt.title("Prediction Confidence Distribution")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
