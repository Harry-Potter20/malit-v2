from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


class ArtifactSaver:
    def __init__(self, base_dir: str | Path = "."):
        self.base = Path(base_dir)

    def _ensure(self, *parts: str) -> Path:
        path = self.base.joinpath(*parts)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_model(self, model: torch.nn.Module, name: str, seed: int, epoch: int | None = None) -> Path:
        out_dir = self._ensure("results", "models")
        suffix = f"_seed{seed}" + (f"_epoch{epoch}" if epoch is not None else "")
        out_path = out_dir / f"{name}{suffix}.pt"
        torch.save({"model_state": model.state_dict(), "seed": seed, "epoch": epoch}, out_path)
        return out_path

    def save_metrics(self, metrics: dict[str, Any], name: str) -> Path:
        out_dir = self._ensure("results", "statistics")
        out_path = out_dir / f"{name}.json"
        with open(out_path, "w") as f:
            json.dump(_make_serializable(metrics), f, indent=2)
        return out_path

    def save_figure(self, fig: Any, name: str, subdir: str = "plots") -> Path:
        out_dir = self._ensure("results", subdir)
        out_path = out_dir / f"{name}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        return out_path

    def save_table(self, df: Any, name: str) -> Path:
        out_dir = self._ensure("results", "tables")
        csv_path = out_dir / f"{name}.csv"
        df.to_csv(csv_path, index=False)
        return csv_path

    def save_reproducibility(self, data: dict[str, Any], filename: str) -> Path:
        out_dir = self._ensure("results", "reproducibility")
        out_path = out_dir / filename
        with open(out_path, "w") as f:
            json.dump(_make_serializable(data), f, indent=2)
        return out_path

    def save_numpy(self, arr: np.ndarray, name: str, subdir: str = "results") -> Path:
        out_dir = self._ensure(subdir)
        out_path = out_dir / f"{name}.npy"
        np.save(out_path, arr)
        return out_path


def _make_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        v = obj.item()
        return None if not np.isfinite(v) else v
    if isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj
