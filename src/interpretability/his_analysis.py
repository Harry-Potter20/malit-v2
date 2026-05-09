from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def extract_his_stats(model) -> dict:
    """
    Extracts Hierarchical Inhibition Scheduling statistics across all three LCCI depths.

    Returns depth scaling factors (delta), mean inhibition magnitudes, and dominance
    ratios for: gabor_lcci (depth 1), backbone_lcci (depth 2), aggregator branches (depth 3).
    """
    stats = {}

    lcci_d1 = model.gabor_lcci
    stats["delta_1"] = lcci_d1.depth_scale.item()
    stats["mean_w_inh_1"] = lcci_d1.w_inh.abs().mean().item()
    stats["dominance_ratio_1"] = _dominance_ratio(lcci_d1)

    lcci_d2 = model.backbone_lcci
    stats["delta_2"] = lcci_d2.depth_scale.item()
    stats["mean_w_inh_2"] = lcci_d2.w_inh.abs().mean().item()
    stats["dominance_ratio_2"] = _dominance_ratio(lcci_d2)

    branch_deltas, branch_winh, branch_dom = [], [], []
    for branch in model.aggregator.branches:
        branch_deltas.append(branch.lcci.depth_scale.item())
        branch_winh.append(branch.lcci.w_inh.abs().mean().item())
        branch_dom.append(_dominance_ratio(branch.lcci))

    stats["delta_3_mean"] = float(np.mean(branch_deltas))
    stats["delta_3_range"] = [float(min(branch_deltas)), float(max(branch_deltas))]
    stats["mean_w_inh_3"] = float(np.mean(branch_winh))
    stats["dominance_ratio_3_mean"] = float(np.mean(branch_dom))

    return stats


@torch.no_grad()
def _dominance_ratio(lcci_module) -> float:
    """Fraction of channels whose effective inhibition exceeds the mean magnitude."""
    effective_w = (lcci_module.depth_scale * lcci_module.w_inh).abs()
    return (effective_w > effective_w.mean()).float().mean().item()
