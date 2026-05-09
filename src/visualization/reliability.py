from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_reliability_diagram(
    confidences,
    correctness,
    save_path: str | Path,
    n_bins: int = 10,
) -> None:
    """Reliability diagram comparing model confidence to empirical accuracy."""
    confidences = np.asarray(confidences)
    correctness = np.asarray(correctness)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centres = (bins[:-1] + bins[1:]) / 2

    empirical_acc, empirical_conf = [], []
    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if mask.sum() == 0:
            empirical_acc.append(0.0)
            empirical_conf.append(float(bin_centres[i]))
        else:
            empirical_acc.append(float(correctness[mask].mean()))
            empirical_conf.append(float(confidences[mask].mean()))

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    plt.plot(empirical_conf, empirical_acc, marker="o", label="Model")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
