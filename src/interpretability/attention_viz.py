from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class AttentionVisualizer:
    """Visualizes channel and spatial attention maps from DualAttention."""

    def __init__(self, model: nn.Module, device: str | torch.device | None = None):
        self.model = model
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

    @torch.no_grad()
    def collect_attention(
        self, loader: DataLoader, max_batches: int = 10
    ) -> dict[str, list[np.ndarray]]:
        self.model.eval()
        self.model.to(self.device)
        ch_atts, sp_atts, entropies = [], [], []

        for i, (imgs, _) in enumerate(tqdm(loader, desc="Attention", total=max_batches)):
            if i >= max_batches:
                break
            imgs = imgs.to(self.device)
            _, info = self.model(imgs)
            if "channel_attention" in info:
                ch_atts.append(info["channel_attention"].cpu().numpy())
            if "spatial_attention" in info:
                sp_atts.append(info["spatial_attention"].cpu().numpy())
            if "attention_entropy" in info:
                entropies.append(info["attention_entropy"].cpu().numpy())

        return {
            "channel_attention": ch_atts,
            "spatial_attention": sp_atts,
            "attention_entropy": entropies,
        }

    def plot_spatial_attention(
        self,
        images: torch.Tensor,
        out_dir: str | Path,
        prefix: str = "attention",
    ) -> list[Path]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.model.eval()
        self.model.to(self.device)
        saved = []

        with torch.no_grad():
            imgs = images.to(self.device)
            _, info = self.model(imgs)

        sp_att = info.get("spatial_attention")
        if sp_att is None:
            return saved

        for i, (img_t, att) in enumerate(zip(images, sp_att)):
            img_np = img_t.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
            img_np = np.clip(img_np, 0, 1)

            att_map = att.squeeze().cpu().numpy()

            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].imshow(img_np); axes[0].set_title("Image"); axes[0].axis("off")
            im = axes[1].imshow(att_map, cmap="hot"); axes[1].set_title("Spatial Attention"); axes[1].axis("off")
            plt.colorbar(im, ax=axes[1], fraction=0.046)
            plt.tight_layout()
            out_path = out_dir / f"{prefix}_{i:04d}.png"
            fig.savefig(out_path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            saved.append(out_path)

        return saved

    def plot_entropy_distribution(
        self,
        entropies: list[np.ndarray],
        out_dir: str | Path,
        prefix: str = "entropy",
    ) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        all_ent = np.concatenate(entropies)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(all_ent, bins=50, color="steelblue", edgecolor="white")
        ax.set_xlabel("Attention Entropy"); ax.set_ylabel("Count")
        ax.set_title("Distribution of Attention Entropy")
        out_path = out_dir / f"{prefix}_distribution.png"
        fig.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        return out_path
