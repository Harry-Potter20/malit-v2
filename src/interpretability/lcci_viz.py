from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class LCCIVisualizer:
    """
    Visualizes LCCI channel suppression patterns.

    Compares channel energies before and after LCCI to show
    which channels are amplified vs suppressed.
    """

    def __init__(self, model: nn.Module, device: str | torch.device | None = None):
        self.model = model
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

    @torch.no_grad()
    def collect_channel_energies(
        self, loader: DataLoader, max_batches: int = 20
    ) -> dict[str, np.ndarray]:
        """
        Captures channel energies before and after LCCI using hooks.
        E_c = (1/HW) Σ x²
        """
        self.model.eval()
        self.model.to(self.device)

        pre_energies, post_energies, labels_all = [], [], []
        pre_features: dict = {}
        post_features: dict = {}

        def pre_hook(module, inp, out):
            pre_features["x"] = inp[0].detach()

        def post_hook(module, inp, out):
            post_features["x"] = out.detach()

        lcci_module = getattr(self.model, "lcci", None)
        if lcci_module is None:
            raise ValueError("Model has no 'lcci' attribute.")

        h_pre = lcci_module.register_forward_pre_hook(pre_hook)
        h_post = lcci_module.register_forward_hook(post_hook)

        for i, (imgs, labels) in enumerate(tqdm(loader, desc="LCCI energies")):
            if i >= max_batches:
                break
            imgs = imgs.to(self.device)
            self.model(imgs)
            # E_c = (1/HW) Σ x²
            pre_e = pre_features["x"].pow(2).mean(dim=(-2, -1)).cpu().numpy()
            post_e = post_features["x"].pow(2).mean(dim=(-2, -1)).cpu().numpy()
            pre_energies.append(pre_e)
            post_energies.append(post_e)
            labels_all.extend(labels.tolist())

        h_pre.remove()
        h_post.remove()

        return {
            "pre_lcci": np.concatenate(pre_energies, axis=0),
            "post_lcci": np.concatenate(post_energies, axis=0),
            "labels": np.array(labels_all),
        }

    def plot_channel_suppression(
        self,
        energies: dict[str, np.ndarray],
        out_dir: str | Path,
        top_k: int = 20,
    ) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        pre = energies["pre_lcci"].mean(axis=0)   # (C,)
        post = energies["post_lcci"].mean(axis=0)  # (C,)
        ratio = post / (pre + 1e-8)

        top_idx = np.argsort(ratio)[-top_k:]
        bot_idx = np.argsort(ratio)[:top_k]
        viz_idx = np.concatenate([bot_idx, top_idx])

        fig, ax = plt.subplots(figsize=(14, 5))
        x = np.arange(len(viz_idx))
        ax.bar(x[:top_k], ratio[bot_idx], color="tomato", label="Suppressed")
        ax.bar(x[top_k:], ratio[top_idx], color="steelblue", label="Amplified")
        ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Channel Index"); ax.set_ylabel("Post/Pre Energy Ratio")
        ax.set_title(f"LCCI Channel Suppression (top-{top_k} each)")
        ax.legend()
        plt.tight_layout()
        out_path = out_dir / "lcci_channel_suppression.png"
        fig.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def dominance_ratio(self, energies: np.ndarray, top_pct: float = 0.15) -> float:
        """DR = top_15% energy / total energy (per sample, then averaged)."""
        n_top = max(1, int(energies.shape[1] * top_pct))
        sorted_e = np.sort(energies, axis=1)
        top_sum = sorted_e[:, -n_top:].sum(axis=1)
        total_sum = energies.sum(axis=1) + 1e-8
        return float((top_sum / total_sum).mean())
