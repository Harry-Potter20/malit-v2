"""Visualization utilities for MALIT V2 interpretability outputs."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def plot_gabor_filters(gabor_layer: nn.Module, out_path: str | Path) -> None:
    """Plot all learned Gabor filter kernels as a scale×orientation grid."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        kernels = gabor_layer._build_kernels()  # (n_filters, 1, K, K)
    kernels_np = kernels.squeeze(1).cpu().numpy()  # (n_filters, K, K)

    n_rows = gabor_layer.n_scales
    n_cols = gabor_layer.n_orientations

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows))
    fig.suptitle("Learned Gabor Filters (rows=scale, cols=orientation)", fontsize=13)

    for i, k in enumerate(kernels_np):
        r, c = i // n_cols, i % n_cols
        ax = axes[r, c] if n_rows > 1 else axes[c]
        vmax = max(abs(k.min()), abs(k.max())) + 1e-8
        ax.imshow(k, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"s{r} θ{c}", fontsize=7)
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Gabor filters → %s", out_path)


def plot_lcci_gates(
    model: nn.Module,
    loader: DataLoader,
    device: str | torch.device,
    out_path: str | Path,
    n_batches: int = 5,
) -> None:
    """Bar chart of mean LCCI gate values per channel (shows lateral inhibition pattern)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not hasattr(model, "lcci"):
        logger.warning("Model has no lcci module — skipping LCCI visualization.")
        return

    device = torch.device(device)
    model.eval().to(device)
    gate_vals: list[torch.Tensor] = []

    def _hook(module, inp, out):
        x = inp[0]
        g = module.gate(x)  # (B, C)
        gate_vals.append(g.detach().cpu())

    handle = model.lcci.gate.register_forward_hook(_hook)
    with torch.no_grad():
        for i, (imgs, _) in enumerate(loader):
            if i >= n_batches:
                break
            model(imgs.to(device))
    handle.remove()

    if not gate_vals:
        return

    all_gates = torch.cat(gate_vals, dim=0).numpy()  # (N, C)
    mean_gate = all_gates.mean(axis=0)
    std_gate  = all_gates.std(axis=0)

    C = len(mean_gate)
    fig, ax = plt.subplots(figsize=(max(10, C // 25), 4))
    colors = ["#d73027" if v < 0.5 else "#4575b4" for v in mean_gate]
    ax.bar(np.arange(C), mean_gate, yerr=std_gate, capsize=1, width=0.9,
           color=colors, alpha=0.75)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8,
               label="0.5 = neutral gate")
    ax.set_xlabel("Channel index")
    ax.set_ylabel("Mean gate (sigmoid)")
    ax.set_title("LCCI Lateral Inhibition Gates  (red=suppressed, blue=enhanced)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("LCCI gate plot → %s", out_path)


def plot_cbr_consistency(cbr_results: dict[str, Any], out_path: str | Path) -> None:
    """Histogram of CBR consistency scores split by true label."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    neighbors = cbr_results.get("neighbors", [])
    if not neighbors:
        logger.warning("No CBR neighbors to plot.")
        return

    scores_0 = [r["consistency_score"] for r in neighbors if r["query_label"] == 0]
    scores_1 = [r["consistency_score"] for r in neighbors if r["query_label"] == 1]

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0, 1, 21)
    ax.hist(scores_0, bins=bins, alpha=0.6, label="Uninfected (0)", color="#2166ac")
    ax.hist(scores_1, bins=bins, alpha=0.6, label="Parasitized (1)", color="#d6604d")
    ax.set_xlabel("Consistency score (fraction of k-NN with same label)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"CBR Consistency  mean={cbr_results['mean_consistency']:.3f} "
        f"± {cbr_results['std_consistency']:.3f}"
    )
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("CBR consistency plot → %s", out_path)


def save_gradcam_grid(
    model: nn.Module,
    loader: DataLoader,
    device: str | torch.device,
    out_dir: str | Path,
    n_images: int = 8,
) -> None:
    """Run GradCAM on the first n_images from loader and save individual overlays."""
    from src.interpretability.gradcam import GradCAMVisualizer

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    model.to(device)

    for imgs, _ in loader:
        batch = imgs[:n_images].to(device)
        viz = GradCAMVisualizer(model)
        viz.save_batch(batch, out_dir, prefix="gradcam")
        logger.info("GradCAM images saved to %s", out_dir)
        break
