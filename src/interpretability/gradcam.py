from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


class GradCAMVisualizer:
    """
    Gradient-weighted Class Activation Mapping for MALITV2.

    Hooks into the last convolutional layer of the MultiScale aggregator.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None):
        self.model = model
        self.target_layer = target_layer or self._find_target_layer(model)
        self._gradients: torch.Tensor | None = None
        self._activations: torch.Tensor | None = None
        self._handles: list = []

    def _find_target_layer(self, model: nn.Module) -> nn.Module:
        """Default to the last Conv2d in the multiscale aggregator."""
        if hasattr(model, "multiscale"):
            for m in reversed(list(model.multiscale.modules())):
                if isinstance(m, nn.Conv2d):
                    return m
        for m in reversed(list(model.modules())):
            if isinstance(m, nn.Conv2d):
                return m
        raise ValueError("No Conv2d layer found in model.")

    def _register_hooks(self) -> None:
        def fwd_hook(module, inp, out):
            self._activations = out.detach()

        def bwd_hook(module, grad_in, grad_out):
            self._gradients = grad_out[0].detach()

        self._handles = [
            self.target_layer.register_forward_hook(fwd_hook),
            self.target_layer.register_full_backward_hook(bwd_hook),
        ]

    def _remove_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def generate(
        self, x: torch.Tensor, class_idx: int | None = None
    ) -> np.ndarray:
        """
        Compute GradCAM heatmap for a single image or batch.

        Returns cam : (B, H, W) normalized heatmaps.
        """
        self._register_hooks()
        self.model.eval()
        x = x.requires_grad_(True)

        logits, _ = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1)[0].item())

        self.model.zero_grad()
        logits[:, class_idx].sum().backward()
        self._remove_hooks()

        weights = self._gradients.mean(dim=(-2, -1), keepdim=True)  # (B, C, 1, 1)
        cam = (weights * self._activations).sum(dim=1)              # (B, H, W)
        cam = torch.relu(cam)
        # Normalize per sample
        B, H, W = cam.shape
        cam_flat = cam.view(B, -1)
        cam_min = cam_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1)
        cam_max = cam_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam.detach().cpu().numpy()

    def overlay(
        self,
        image: np.ndarray,
        cam: np.ndarray,
        alpha: float = 0.4,
    ) -> np.ndarray:
        """Overlay heatmap on original image (H×W×3 uint8)."""
        from PIL import Image as PILImage
        import matplotlib.cm as cm

        h, w = image.shape[:2]
        cam_resized = np.array(
            PILImage.fromarray((cam * 255).astype(np.uint8)).resize((w, h), PILImage.BILINEAR)
        ) / 255.0
        colormap = cm.jet(cam_resized)[:, :, :3]  # (H, W, 3)
        overlay = (1 - alpha) * image / 255.0 + alpha * colormap
        return np.clip(overlay * 255, 0, 255).astype(np.uint8)

    def save_batch(
        self,
        images: torch.Tensor,
        out_dir: str | Path,
        prefix: str = "gradcam",
        class_idx: int | None = None,
    ) -> list[Path]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cams = self.generate(images, class_idx)
        saved = []
        for i, (img_t, cam) in enumerate(zip(images, cams)):
            # De-normalize image for display
            img_np = img_t.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
            img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)

            overlaid = self.overlay(img_np, cam)
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(img_np); axes[0].set_title("Original"); axes[0].axis("off")
            axes[1].imshow(cam, cmap="jet"); axes[1].set_title("GradCAM"); axes[1].axis("off")
            axes[2].imshow(overlaid); axes[2].set_title("Overlay"); axes[2].axis("off")
            plt.tight_layout()
            out_path = out_dir / f"{prefix}_{i:04d}.png"
            fig.savefig(out_path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            saved.append(out_path)
        return saved
