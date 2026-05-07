from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


class GaborVisualizer:
    """Visualizes learned Gabor filter bank parameters and kernels."""

    def __init__(self, gabor_layer: nn.Module):
        self.layer = gabor_layer

    def get_kernels(self) -> np.ndarray:
        """Build and return all Gabor kernels as numpy array (n_filters, K, K)."""
        with torch.no_grad():
            kernels = self.layer._build_kernels()  # (n_filters, 1, K, K)
        return kernels.squeeze(1).cpu().numpy()

    def plot_filter_bank(self, out_dir: str | Path, cols: int = 8) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        kernels = self.get_kernels()
        n = len(kernels)
        rows = math.ceil(n / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        axes = np.array(axes).reshape(-1)

        for i, ax in enumerate(axes):
            if i < n:
                ax.imshow(kernels[i], cmap="RdBu_r", interpolation="nearest")
                ax.set_title(f"f{i}", fontsize=7)
            ax.axis("off")

        plt.suptitle("Learned Gabor Filter Bank", fontsize=14)
        plt.tight_layout()
        out_path = out_dir / "gabor_filter_bank.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def plot_parameter_distribution(self, out_dir: str | Path) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            thetas = self.layer.theta.cpu().numpy()
            lambdas = self.layer.lam.cpu().numpy()
            sigmas = self.layer.sigma.cpu().numpy()
            gammas = self.layer.gamma.cpu().numpy()

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        params = [
            (thetas, "θ (orientation, rad)", "forestgreen"),
            (lambdas, "λ (frequency)", "steelblue"),
            (sigmas, "σ (bandwidth)", "darkorange"),
            (gammas, "γ (aspect ratio)", "purple"),
        ]
        for ax, (vals, label, color) in zip(axes.flat, params):
            ax.hist(vals, bins=min(20, len(vals)), color=color, edgecolor="white", alpha=0.8)
            ax.set_title(label); ax.set_xlabel("Value"); ax.set_ylabel("Count")

        plt.suptitle("Learned Gabor Parameter Distributions", fontsize=13)
        plt.tight_layout()
        out_path = out_dir / "gabor_param_distributions.png"
        fig.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def plot_decomposition(
        self,
        image: torch.Tensor,
        out_dir: str | Path,
        n_show: int = 8,
    ) -> Path:
        """Show first n_show Gabor responses for a single input image."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.layer.eval()

        with torch.no_grad():
            device = next(self.layer.parameters()).device
            img_batch = image.unsqueeze(0).to(device)
            out = self.layer(img_batch)  # (1, out_channels, H, W)

        out_np = out[0].cpu().numpy()
        n_show = min(n_show, len(out_np))
        img_np = image.permute(1, 2, 0).numpy()
        img_np = np.clip(
            img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1
        )

        fig, axes = plt.subplots(1, n_show + 1, figsize=((n_show + 1) * 3, 3))
        axes[0].imshow(img_np); axes[0].set_title("Input"); axes[0].axis("off")
        for i in range(n_show):
            axes[i + 1].imshow(out_np[i % len(out_np)], cmap="gray")
            axes[i + 1].set_title(f"Gabor ch{i}"); axes[i + 1].axis("off")

        plt.tight_layout()
        out_path = out_dir / "gabor_decomposition.png"
        fig.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        return out_path
