# MALIT-H v2: HIS + CAL Enhanced Implementation Guide

## Production-Grade Integration for AIM Resubmission

---

# Overview

This document describes the finalized implementation of two major additions to the MALIT-H architecture:

1. **Hierarchical Inhibition Scheduling (HIS)**
2. **Calibration-Aware Learning (CAL)**

These additions improve:

* interpretability,
* biological plausibility,
* uncertainty reliability,
* clinical deployment readiness,
* and calibration robustness.

The implementation described here includes several engineering improvements beyond the original draft:

* positive-constrained inhibition scaling,
* numerically stable differentiable ECE,
* CAL warmup scheduling,
* per-epoch calibration tracking,
* and safer statistical reporting.

The goal is not merely higher accuracy, but a more trustworthy diagnostic system.

---

# Part 1 — Hierarchical Inhibition Scheduling (HIS)

## Concept

HIS introduces a learnable depth-wise inhibition scaling factor:

[
w_{eff} = \delta_d \cdot w_{inh}
]

where:

* (w_{inh}) is the learnable inhibition vector,
* (\delta_d) is a learnable scalar specific to depth level (d).

This allows inhibition strength to emerge dynamically across architectural depth.

Unlike hard-coded biological priors, HIS imposes:

* no explicit hierarchy,
* no ordering constraint,
* no manually designed schedule.

The ordering:

[
\delta_1 > \delta_2 > \delta_3
]

must emerge naturally during optimization.

---

# Engineering Improvement: Positive-Constrained Inhibition Scaling

The original implementation used:

```python
self.depth_scale = nn.Parameter(torch.tensor(1.0))
```

This allows the scalar to become negative during optimization.

Negative inhibition scaling effectively transforms inhibition into excitation.
This violates the intended biological interpretation.

Instead, use a Softplus-constrained parameterization.

---

# File to Modify

## `src/models/lcci.py`

---

# Final HIS Implementation

```python
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LCCIModule(nn.Module):
    """
    Lateral Competitive Cortical Inhibition (LCCI).

    HIS Extension:
        Adds a positive-constrained learnable depth scaling factor.
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()

        # Channel-wise inhibition weights
        self.w_inh = nn.Parameter(torch.ones(channels))

        # HIS: unconstrained raw parameter
        # Softplus ensures delta_d > 0
        self.depth_scale_raw = nn.Parameter(torch.tensor(0.0))

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    @property
    def depth_scale(self) -> torch.Tensor:
        """
        Positive-constrained inhibition scaling.

        delta_d = softplus(raw)
        """
        return F.softplus(self.depth_scale_raw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.gate(x).unsqueeze(-1).unsqueeze(-1)

        # Effective inhibition
        effective_w_inh = self.depth_scale * self.w_inh

        inhibition = g * (
            x * effective_w_inh.view(1, -1, 1, 1)
        )

        return torch.relu(x + inhibition)
```

---

# Why Softplus?

Softplus guarantees:

[
\delta_d > 0
]

without imposing any ordering constraints.

Advantages:

* preserves biological interpretability,
* prevents sign inversion,
* avoids unstable negative scaling,
* remains fully differentiable.

---

# HIS Analysis Utilities

Create:

## `src/interpretability/his_analysis.py`

```python
from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def extract_his_stats(model) -> dict:
    """
    Extracts hierarchical inhibition statistics.

    Returns:
        Dictionary containing:
            - delta values
            - inhibition magnitudes
            - dominance ratios
    """

    stats = {}

    # =========================================================
    # Depth 1 — Gabor-level LCCI
    # =========================================================

    lcci_d1 = model.gabor_lcci

    stats['delta_1'] = lcci_d1.depth_scale.item()
    stats['mean_w_inh_1'] = (
        lcci_d1.w_inh.abs().mean().item()
    )

    stats['dominance_ratio_1'] = _dominance_ratio(lcci_d1)

    # =========================================================
    # Depth 2 — Backbone-level LCCI
    # =========================================================

    lcci_d2 = model.backbone_lcci

    stats['delta_2'] = lcci_d2.depth_scale.item()
    stats['mean_w_inh_2'] = (
        lcci_d2.w_inh.abs().mean().item()
    )

    stats['dominance_ratio_2'] = _dominance_ratio(lcci_d2)

    # =========================================================
    # Depth 3 — Aggregator Branches
    # =========================================================

    branch_deltas = []
    branch_winh = []
    branch_dom = []

    for branch in model.aggregator.branches:
        branch_deltas.append(
            branch.lcci.depth_scale.item()
        )

        branch_winh.append(
            branch.lcci.w_inh.abs().mean().item()
        )

        branch_dom.append(
            _dominance_ratio(branch.lcci)
        )

    stats['delta_3_mean'] = float(np.mean(branch_deltas))
    stats['delta_3_range'] = [
        float(min(branch_deltas)),
        float(max(branch_deltas)),
    ]

    stats['mean_w_inh_3'] = float(np.mean(branch_winh))

    stats['dominance_ratio_3_mean'] = (
        float(np.mean(branch_dom))
    )

    return stats


@torch.no_grad()
def _dominance_ratio(lcci_module) -> float:
    """
    Fraction of channels whose effective inhibition
    exceeds the mean inhibition magnitude.
    """

    effective_w = (
        lcci_module.depth_scale * lcci_module.w_inh
    ).abs()

    return (
        (effective_w > effective_w.mean())
        .float()
        .mean()
        .item()
    )
```

---

# Expected HIS Behaviour

After training, verify whether:

[
\delta_1 > \delta_2 > \delta_3
]

emerges naturally.

If it does:

* report emergent hierarchical inhibition,
* discuss biological parallels.

If it does NOT:

* report the observed pattern honestly,
* avoid claiming cortical hierarchy mirroring.

Never force the ordering.

---

# Part 2 — Calibration-Aware Learning (CAL)

## Concept

Modern deep neural networks are frequently overconfident.

A clinically deployable system must not only classify correctly,
but estimate confidence reliably.

CAL augments the classification objective with a differentiable
Expected Calibration Error (ECE) term.

---

# Objective Function

[
\mathcal{L} = \mathcal{L}_{CE} + \lambda \cdot ECE
]

where:

* (\mathcal{L}_{CE}) = Cross Entropy Loss,
* (ECE) = differentiable Expected Calibration Error,
* (\lambda) = calibration weighting coefficient.

Recommended default:

```yaml
lambda_cal: 0.1
```

---

# Engineering Improvements Added

Compared to the initial draft, this implementation adds:

1. Numerical stability improvements
2. Stable soft binning
3. CAL warmup scheduling
4. Per-epoch calibration tracking
5. Safer logging

---

# Create File

## `src/training/cal_loss.py`

```python
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CalibrationAwareLoss(nn.Module):
    """
    Calibration-Aware Loss.

    Total Loss:
        L = CE + lambda * ECE
    """

    def __init__(
        self,
        lambda_cal: float = 0.1,
        n_bins: int = 10,
    ):
        super().__init__()

        self.lambda_cal = lambda_cal
        self.n_bins = n_bins

        self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        apply_calibration: bool = True,
    ):
        """
        Returns:
            total_loss
            logging dictionary
        """

        ce_loss = self.ce(logits, targets)

        if not apply_calibration:
            return ce_loss, {
                'ce': ce_loss.item(),
                'ece': 0.0,
                'total': ce_loss.item(),
            }

        ece_loss = self._soft_ece(logits, targets)

        total = ce_loss + self.lambda_cal * ece_loss

        return total, {
            'ce': ce_loss.item(),
            'ece': ece_loss.item(),
            'total': total.item(),
        }

    def _soft_ece(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Differentiable Expected Calibration Error.
        """

        probs = F.softmax(logits, dim=1)

        confidence = probs.max(dim=1).values

        predictions = probs.argmax(dim=1)

        correct = (
            predictions == targets
        ).float()

        # =====================================================
        # Soft Binning
        # =====================================================

        bin_edges = torch.linspace(
            0.0,
            1.0,
            self.n_bins + 1,
            device=logits.device,
        )

        bin_centres = (
            bin_edges[:-1] + bin_edges[1:]
        ) / 2

        bin_width = 1.0 / self.n_bins

        sigma = bin_width / 2.0

        dist = (
            confidence.unsqueeze(1)
            - bin_centres.unsqueeze(0)
        ) ** 2

        weights = torch.exp(
            -dist / (2 * sigma ** 2)
        )

        # Numerical stability
        weights = weights / (
            weights.sum(dim=1, keepdim=True)
            + 1e-6
        )

        # =====================================================
        # Per-bin Statistics
        # =====================================================

        bin_acc = (
            weights * correct.unsqueeze(1)
        ).sum(dim=0)

        bin_conf = (
            weights * confidence.unsqueeze(1)
        ).sum(dim=0)

        bin_n = weights.sum(dim=0)

        bin_n_safe = bin_n + 1e-6

        bin_acc_norm = bin_acc / bin_n_safe
        bin_conf_norm = bin_conf / bin_n_safe

        ece = (
            (bin_n / bin_n.sum())
            * (bin_acc_norm - bin_conf_norm).abs()
        ).sum()

        return ece
```

---

# Part 3 — CAL Warmup Scheduling

## Why Warmup Matters

Early training predictions are extremely unstable.

Applying ECE regularization too early can:

* destabilize representation learning,
* reduce convergence quality,
* increase gradient noise,
* hurt final F1.

Instead:

* train with CE only initially,
* activate CAL after feature stabilization.

---

# Recommended Warmup

```yaml
cal_warmup_epochs: 3
```

This means:

* Epochs 1–3:

  * CE only
* Epoch 4 onward:

  * CE + CAL

---

# Update Config

## `configs/malit_v2.yaml`

```yaml
training:
  lambda_cal: 0.1
  cal_warmup_epochs: 3
```

---

# Update Config Dataclass

## `src/utils/config.py`

```python
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    lambda_cal: float = 0.1
    cal_warmup_epochs: int = 3
```

---

# Part 4 — Modify Training Loop

## File

`src/training/runner.py`

---

# Replace Criterion

```python
from src.training.cal_loss import CalibrationAwareLoss

criterion = CalibrationAwareLoss(
    lambda_cal=cfg.training.lambda_cal,
)
```

---

# Update Training Step

```python
apply_cal = (
    epoch >= cfg.training.cal_warmup_epochs
)

loss, loss_components = criterion(
    logits,
    labels,
    apply_calibration=apply_cal,
)

loss.backward()
```

---

# Optional Logging

```python
logger.debug(
    "CE=%.4f | ECE=%.4f | TOTAL=%.4f",
    loss_components['ce'],
    loss_components['ece'],
    loss_components['total'],
)
```

---

# Part 5 — Per-Epoch Calibration Tracking

Calibration trends are extremely valuable for:

* reviewer confidence,
* interpretability,
* deployment justification,
* uncertainty analysis.

---

# Recommended Metrics to Track Per Epoch

Save:

* train ECE,
* validation ECE,
* Brier score,
* confidence mean,
* confidence variance,
* NLL.

---

# Recommended Output File

```text
results/calibration_curves_seedN.json
```

---

# Example Tracking Structure

```json
{
  "epoch_1": {
    "train_ece": 0.081,
    "val_ece": 0.093,
    "brier": 0.142
  },
  "epoch_2": {
    "train_ece": 0.054,
    "val_ece": 0.066,
    "brier": 0.121
  }
}
```

---

# Part 6 — Integrate HIS Analysis into Pipeline

## File

`scripts/run_full_pipeline.py`

---

# Add HIS Extraction

```python
logger.info("━━━ [2b/7] HIS Analysis ━━━")

from src.interpretability.his_analysis import (
    extract_his_stats,
)

for seed, model in runner.seed_models.items():

    his_stats = extract_his_stats(model)

    saver.save_metrics(
        his_stats,
        f"his_stats_seed{seed}",
    )

    logger.info(
        (
            "Seed %s HIS: "
            "δ₁=%.3f | "
            "δ₂=%.3f | "
            "δ₃=%.3f"
        ),
        seed,
        his_stats['delta_1'],
        his_stats['delta_2'],
        his_stats['delta_3_mean'],
    )
```

---

# Part 7 — Statistical Reporting Guidance

Avoid hard-coded calibration claims such as:

> “ECE should be ≤ 0.010.”

ECE depends heavily on:

* prevalence,
* class imbalance,
* confidence sharpness,
* batch composition,
* binning strategy.

Instead report:

> “CAL reduced ECE relative to the baseline model while preserving discriminative performance.”

This is statistically safer.

---

# Part 8 — What to Verify After Retraining

Run all three seeds.

Then verify the following:

| Check            | Desired Outcome         | If It Fails             |
| ---------------- | ----------------------- | ----------------------- |
| HIS ordering     | δ₁ > δ₂ > δ₃            | Report actual emergence |
| CAL improvement  | ECE lower than baseline | Increase λ to 0.2       |
| F1 stability     | Within ±0.3%            | Reduce λ to 0.05        |
| Brier stability  | Similar or improved     | Tune warmup             |
| CI width-error r | ≥ 0.35                  | Report actual value     |

---

# Part 9 — Recommended Visualizations

These plots substantially improve reviewer confidence.

## Strongly Recommended Figures

### 1. ECE vs Epoch

Shows:

* calibration improvement,
* convergence stabilization,
* effect of CAL warmup.

---

### 2. Reliability Diagram

Shows:

* confidence calibration,
* overconfidence reduction.

---

### 3. HIS Depth Distribution

Plot:

* δ₁,
* δ₂,
* δ₃.

Across all seeds.

---

### 4. Confidence vs Error Scatter

Supports:

* escalation logic,
* uncertainty-aware deployment.

---

# Part 10 — Manuscript Sections to Update

Update the following sections after retraining.

---

# Abstract

Update:

* ECE,
* calibration claims,
* uncertainty statements.

---

# Table 1

Update:

* F1,
* AUROC,
* ECE,
* Brier,
* NLL,
* Sensitivity,
* Specificity.

---

# Table 4

Add:

* δ values,
* dominance ratios,
* HIS emergence statistics.

---

# Section 3.7 — CAL

Confirm:

* λ used,
* warmup epochs,
* differentiable ECE implementation.

---

# Section 5.4 — Calibration Analysis

Update:

* ECE,
* reliability diagrams,
* uncertainty correlation,
* calibration trends.

---

# Part 11 — Recommended Final Positioning

The strongest framing for MALIT-H is no longer:

> “high-performing malaria classifier.”

The stronger framing is:

> “trustworthy clinically-oriented diagnostic AI system.”

The additions introduced here provide:

* uncertainty reliability,
* interpretable inhibition dynamics,
* confidence-aware escalation,
* biologically inspired modulation,
* clinically safer prediction behaviour.

These properties matter significantly for modern medical AI review standards.

---

# Final Runtime Estimate

| Stage                | Estimated Runtime |
| -------------------- | ----------------- |
| 3-seed retraining    | 45–60 min         |
| HIS extraction       | <1 min            |
| Uncertainty analysis | 10 min            |
| Calibration plots    | 5 min             |
| Statistics           | 2 min             |
| Total                | ~60–80 min        |

---

# Final File Checklist

| Action | File                                   |
| ------ | -------------------------------------- |
| Modify | `src/models/lcci.py`                   |
| Create | `src/training/cal_loss.py`             |
| Modify | `src/training/runner.py`               |
| Modify | `configs/malit_v2.yaml`                |
| Modify | `src/utils/config.py`                  |
| Create | `src/interpretability/his_analysis.py` |
| Modify | `scripts/run_full_pipeline.py`         |
| Create | Calibration tracking outputs           |

---

# Final Notes

Do not overstate biological claims.

Do not force HIS ordering.

Do not hard-code calibration targets.

The strength of the system comes from:

* honest uncertainty,
* reliable calibration,
* interpretable inhibition,
* clinically meaningful confidence behaviour,
* and rigorous reporting.

That positioning is substantially stronger scientifically than chasing marginal accuracy improvements alone.
