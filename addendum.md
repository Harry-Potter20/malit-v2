MALIT-H v2 Addendum
Advanced Reliability + HIS Tracking Enhancements

This document contains only the additional enhancements recommended after the successful implementation of the main HIS + CAL architecture.

These additions improve:

interpretability,
calibration transparency,
reviewer confidence,
longitudinal analysis,
and uncertainty auditing.

The core HIS + CAL implementation has already been completed.

Included Enhancements

This addendum covers only:

Reliability Diagram Generation
Confidence Histogram Generation
Per-Sample Prediction Logging
HIS Dynamics Tracking Across Epochs
Enhancement 1 — Reliability Diagram Generation
Purpose

ECE is a useful scalar metric, but reviewers usually expect visual calibration evidence.

Reliability diagrams show:

confidence alignment,
overconfidence,
underconfidence,
calibration improvements after CAL.

These figures are extremely valuable in medical AI submissions.

Create File
src/visualization/reliability.py
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_reliability_diagram(
    confidences,
    correctness,
    save_path,
    n_bins: int = 10,
):
    """
    Reliability diagram for calibration analysis.
    """

    confidences = np.asarray(confidences)
    correctness = np.asarray(correctness)

    bins = np.linspace(0.0, 1.0, n_bins + 1)

    bin_centres = (bins[:-1] + bins[1:]) / 2

    empirical_acc = []
    empirical_conf = []

    for i in range(n_bins):

        mask = (
            (confidences >= bins[i])
            & (confidences < bins[i + 1])
        )

        if mask.sum() == 0:
            empirical_acc.append(0.0)
            empirical_conf.append(bin_centres[i])
            continue

        empirical_acc.append(
            correctness[mask].mean()
        )

        empirical_conf.append(
            confidences[mask].mean()
        )

    plt.figure(figsize=(6, 6))

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle='--',
        label='Perfect Calibration',
    )

    plt.plot(
        empirical_conf,
        empirical_acc,
        marker='o',
        label='Model',
    )

    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')

    plt.title('Reliability Diagram')

    plt.legend()

    plt.tight_layout()

    plt.savefig(save_path, dpi=300)

    plt.close()
Recommended Output
results/reliability_seedN.png
Suggested Integration

After evaluation:

from src.visualization.reliability import (
    plot_reliability_diagram,
)

plot_reliability_diagram(
    confidences=all_confidences,
    correctness=all_correctness,
    save_path=save_dir / f"reliability_seed{seed}.png",
)
Enhancement 2 — Confidence Histogram Generation
Purpose

Confidence histograms help identify:

confidence collapse,
uncertainty spread,
overconfident predictions,
calibration drift.

These complement reliability diagrams well.

Create File
src/visualization/confidence_histogram.py
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_confidence_histogram(
    confidences,
    save_path,
):
    """
    Histogram of prediction confidences.
    """

    confidences = np.asarray(confidences)

    plt.figure(figsize=(7, 5))

    plt.hist(
        confidences,
        bins=20,
    )

    plt.xlabel('Confidence')
    plt.ylabel('Frequency')

    plt.title(
        'Prediction Confidence Distribution'
    )

    plt.tight_layout()

    plt.savefig(save_path, dpi=300)

    plt.close()
Recommended Output
results/confidence_hist_seedN.png
Suggested Integration
from src.visualization.confidence_histogram import (
    plot_confidence_histogram,
)

plot_confidence_histogram(
    confidences=all_confidences,
    save_path=save_dir / f"confidence_hist_seed{seed}.png",
)
Enhancement 3 — Per-Sample Prediction Logging
Purpose

Aggregate metrics are insufficient for detailed uncertainty analysis.

Per-sample outputs enable:

subgroup calibration analysis,
failure mode analysis,
uncertainty auditing,
escalation behaviour analysis,
future explainability studies.

This is extremely useful later.

Recommended Output
results/sample_predictions_seedN.csv
Recommended Saved Fields
Field	Description
image_id	Sample identifier
true_label	Ground-truth label
predicted_label	Predicted class
confidence	Max softmax confidence
entropy	Prediction entropy
correct	Binary correctness
uncertainty_width	Ensemble CI width
escalation_flag	Escalation triggered or not
Create File
src/utils/prediction_logging.py
from __future__ import annotations

import pandas as pd


class PredictionLogger:
    """
    Stores per-sample prediction statistics.
    """

    def __init__(self):
        self.rows = []

    def add(
        self,
        image_id,
        true_label,
        predicted_label,
        confidence,
        entropy,
        correct,
        uncertainty_width=None,
        escalation_flag=None,
    ):
        self.rows.append({
            'image_id': image_id,
            'true_label': true_label,
            'predicted_label': predicted_label,
            'confidence': confidence,
            'entropy': entropy,
            'correct': correct,
            'uncertainty_width': uncertainty_width,
            'escalation_flag': escalation_flag,
        })

    def save(self, path):
        pd.DataFrame(self.rows).to_csv(
            path,
            index=False,
        )
Suggested Integration

Inside evaluation loop:

prediction_logger.add(
    image_id=image_id,
    true_label=label.item(),
    predicted_label=pred.item(),
    confidence=confidence.item(),
    entropy=entropy.item(),
    correct=(pred == label).item(),
    uncertainty_width=ci_width,
    escalation_flag=escalated,
)

After evaluation:

prediction_logger.save(
    save_dir / f"sample_predictions_seed{seed}.csv"
)
Enhancement 4 — HIS Dynamics Tracking Across Epochs
Purpose

Final δ values alone do not show emergence behaviour.

Tracking HIS across epochs enables:

emergence analysis,
hierarchical separation visualization,
reviewer evidence for non-hardcoded inhibition hierarchy,
temporal inhibition analysis.

This is potentially one of the strongest figures in the paper.

Recommended Output
results/his_dynamics_seedN.json
Recommended Structure
{
  "epoch_1": {
    "delta_1": 0.83,
    "delta_2": 0.76,
    "delta_3": 0.71
  },
  "epoch_2": {
    "delta_1": 0.89,
    "delta_2": 0.73,
    "delta_3": 0.68
  }
}
Create File
src/interpretability/his_tracking.py
from __future__ import annotations


class HISTracker:
    """
    Tracks hierarchical inhibition evolution.
    """

    def __init__(self):
        self.history = {}

    def update(self, epoch, his_stats):

        self.history[f'epoch_{epoch}'] = {
            'delta_1': his_stats['delta_1'],
            'delta_2': his_stats['delta_2'],
            'delta_3': his_stats['delta_3_mean'],
        }
Suggested Integration

Inside training loop after each epoch:

from src.interpretability.his_analysis import (
    extract_his_stats,
)

his_stats = extract_his_stats(model)

his_tracker.update(
    epoch=epoch,
    his_stats=his_stats,
)

After training:

import json

with open(
    save_dir / f"his_dynamics_seed{seed}.json",
    "w",
) as f:
    json.dump(his_tracker.history, f, indent=2)
Recommended Future Visualization

Plot:

δ₁,
δ₂,
δ₃

across epochs.

If hierarchical separation emerges gradually during optimization, this becomes very strong evidence for emergent depth-wise inhibition organization.

Expected Runtime Impact
Component	Estimated Cost
Reliability diagrams	<1 min
Confidence histograms	<1 min
Per-sample logging	negligible
HIS tracking	negligible

These additions are lightweight compared to retraining.

Final Recommendation

These four additions substantially improve:

interpretability,
reviewer trust,
calibration transparency,
and scientific maturity.

Most importantly, they help position MALIT-H as:

a trustworthy uncertainty-aware clinical AI system

rather than only a high-performing classifier.