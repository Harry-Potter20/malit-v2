from __future__ import annotations

from pathlib import Path


class PredictionLogger:
    """
    Stores per-sample prediction statistics and saves to CSV.

    Fields: image_id, true_label, predicted_label, confidence, entropy,
            correct, uncertainty_width, escalation_flag.
    """

    def __init__(self):
        self.rows: list[dict] = []

    def add(
        self,
        image_id,
        true_label: int,
        predicted_label: int,
        confidence: float,
        entropy: float,
        correct: int,
        uncertainty_width: float | None = None,
        escalation_flag: bool | None = None,
    ) -> None:
        self.rows.append({
            "image_id": image_id,
            "true_label": true_label,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "entropy": entropy,
            "correct": correct,
            "uncertainty_width": uncertainty_width,
            "escalation_flag": escalation_flag,
        })

    def save(self, path: str | Path) -> None:
        try:
            import pandas as pd
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(self.rows).to_csv(path, index=False)
        except ImportError:
            import csv
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(self.rows[0].keys()))
                writer.writeheader()
                writer.writerows(self.rows)
