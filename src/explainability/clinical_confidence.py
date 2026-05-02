from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize_scalar

logger = logging.getLogger(__name__)

# WHO constraint: ≥99% sensitivity
WHO_SENSITIVITY_THRESHOLD = 0.99


# ── Temperature Scaling Calibration ──────────────────────────────────────────

class TemperatureScaler(nn.Module):
    """Post-hoc probability calibration via temperature scaling."""

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=1e-3)

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> float:
        """Fit temperature on validation logits. Returns best temperature."""
        logits_t = torch.tensor(logits, dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.long)
        criterion = nn.CrossEntropyLoss()

        def nll(temp: float) -> float:
            scaled = logits_t / max(temp, 1e-3)
            return criterion(scaled, labels_t).item()

        result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
        best_temp = float(result.x)
        self.temperature.data.fill_(best_temp)
        logger.info("Temperature scaling: T=%.4f", best_temp)
        return best_temp

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            scaled = torch.tensor(logits) / self.temperature.clamp(min=1e-3)
            return torch.softmax(scaled, dim=-1).numpy()


# ── Core Computations ─────────────────────────────────────────────────────────

def compute_entropy(attention_weights: np.ndarray) -> float:
    """H = -Σ p_c log₂(p_c) over softmax(channel_attention)."""
    p = np.exp(attention_weights - attention_weights.max())
    p = p / (p.sum() + 1e-8)
    return float(-(p * np.log2(p + 1e-8)).sum())


def compute_reliability(entropy: float, num_channels: int) -> float:
    """R_model = 1 - H_norm  where H_norm = H / log₂(C)."""
    h_max = math.log2(max(num_channels, 2))
    h_norm = entropy / h_max
    return float(np.clip(1.0 - h_norm, 0.0, 1.0))


def compute_ccs(prob: float, reliability: float) -> float:
    """CCS = P_calibrated × R_model."""
    return float(np.clip(prob * reliability, 0.0, 1.0))


def compute_seed_agreement(predictions: list[int]) -> float:
    """Agreement = (# seeds predicting majority class) / total seeds."""
    if not predictions:
        return 0.0
    majority = max(set(predictions), key=predictions.count)
    return float(predictions.count(majority) / len(predictions))


def compute_dominance_ratio(channel_energies: np.ndarray, top_pct: float = 0.15) -> float:
    """DR = top_15% energy / total energy."""
    n_top = max(1, int(len(channel_energies) * top_pct))
    top_energy = np.sort(channel_energies)[-n_top:].sum()
    total_energy = channel_energies.sum() + 1e-8
    return float(top_energy / total_energy)


def classify_uncertainty(
    h_norm: float,
    seed_agreement: float,
    dominance_ratio: float,
) -> str:
    """
    Classify uncertainty type:
      Structural   — high entropy + low dominance ratio
      Epistemic    — low seed agreement
      Data ambiguity — moderate entropy + high agreement
    """
    if h_norm > 0.7 and dominance_ratio < 0.4:
        return "Structural"
    if seed_agreement < 0.67:
        return "Epistemic"
    return "Data ambiguity"


# ── Clinical Report Generator ─────────────────────────────────────────────────

def generate_clinical_report(
    cell_id: str,
    pred_class: int,
    calibrated_prob: float,
    reliability: float,
    ccs: float,
    seed_agreement: float,
    n_seeds: int,
    entropy: float,
    h_norm: float,
    dominance_ratio: float,
    uncertainty_type: str,
    mc_variance: float | None = None,
    ensemble_ci_width: float | None = None,
    cbr_consistency: float | None = None,
) -> dict[str, Any]:
    class_names = ["Uninfected", "Parasitized"]

    # WHO sensitivity decision
    meets_who = calibrated_prob >= WHO_SENSITIVITY_THRESHOLD if pred_class == 1 else (1 - calibrated_prob) >= WHO_SENSITIVITY_THRESHOLD

    # Clinical decision
    if ccs >= 0.85 and seed_agreement >= 0.9:
        recommendation = "Accept"
        decision = "Screening-safe"
    elif ccs >= 0.60:
        recommendation = "Review"
        decision = "Review required"
    else:
        recommendation = "Reject"
        decision = "Manual inspection required"

    report: dict[str, Any] = {
        "cell_id": cell_id,
        "prediction": class_names[pred_class],
        "clinical_confidence_pct": round(ccs * 100, 2),
        "breakdown": {
            "calibrated_probability_pct": round(calibrated_prob * 100, 2),
            "reliability_factor_pct": round(reliability * 100, 2),
            "seed_agreement": f"{int(round(seed_agreement * n_seeds))}/{n_seeds}",
        },
        "model_reasoning": {
            "feature_focus": "High" if dominance_ratio > 0.6 else ("Medium" if dominance_ratio > 0.4 else "Low"),
            "channel_dominance": "Strong" if dominance_ratio > 0.5 else "Weak",
            "attention_entropy": round(entropy, 4),
            "attention_entropy_norm": round(h_norm, 4),
        },
        "uncertainty": {
            "type": uncertainty_type,
            "mc_dropout_variance": round(mc_variance, 6) if mc_variance is not None else None,
            "ensemble_ci_width": round(ensemble_ci_width, 4) if ensemble_ci_width is not None else None,
            "cbr_consistency": round(cbr_consistency, 4) if cbr_consistency is not None else None,
        },
        "clinical_safety": {
            "meets_who_sensitivity": bool(meets_who),
            "decision": decision,
        },
        "recommendation": recommendation,
    }
    return report


# ── Engine ────────────────────────────────────────────────────────────────────

class ClinicalConfidenceEngine:
    """
    Orchestrates calibration and clinical confidence scoring for MALIT V2.
    """

    def __init__(self, num_channels: int = 1280):
        self.scaler = TemperatureScaler()
        self.num_channels = num_channels
        self._fitted = False

    def fit_calibration(
        self, val_logits: np.ndarray, val_labels: np.ndarray
    ) -> float:
        temp = self.scaler.fit(val_logits, val_labels)
        self._fitted = True
        return temp

    def score(
        self,
        logits: np.ndarray,
        attention_weights: np.ndarray,
        channel_energies: np.ndarray,
        seed_predictions: list[int],
        cell_id: str = "unknown",
        mc_variance: float | None = None,
        ensemble_ci_width: float | None = None,
        cbr_consistency: float | None = None,
    ) -> dict[str, Any]:
        if not self._fitted:
            logger.warning("Calibration not fitted — using raw softmax probabilities.")
            probs = np.exp(logits - logits.max())
            probs = probs / probs.sum()
        else:
            probs = self.scaler.calibrate(logits[np.newaxis])[0]

        pred_class = int(np.argmax(probs))
        calibrated_prob = float(probs[pred_class])

        entropy = compute_entropy(attention_weights)
        h_max = math.log2(max(self.num_channels, 2))
        h_norm = entropy / h_max
        reliability = compute_reliability(entropy, self.num_channels)
        ccs = compute_ccs(calibrated_prob, reliability)
        agreement = compute_seed_agreement(seed_predictions)
        dr = compute_dominance_ratio(channel_energies)
        unc_type = classify_uncertainty(h_norm, agreement, dr)

        return generate_clinical_report(
            cell_id=cell_id,
            pred_class=pred_class,
            calibrated_prob=calibrated_prob,
            reliability=reliability,
            ccs=ccs,
            seed_agreement=agreement,
            n_seeds=len(seed_predictions),
            entropy=entropy,
            h_norm=h_norm,
            dominance_ratio=dr,
            uncertainty_type=unc_type,
            mc_variance=mc_variance,
            ensemble_ci_width=ensemble_ci_width,
            cbr_consistency=cbr_consistency,
        )
