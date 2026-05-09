from __future__ import annotations


class HISTracker:
    """
    Records delta values (depth_scale) at all three HIS depths after each training epoch.

    history structure:
        {"epoch_1": {"delta_1": float, "delta_2": float, "delta_3": float}, ...}
    """

    def __init__(self):
        self.history: dict[str, dict[str, float]] = {}

    def update(self, epoch: int, his_stats: dict) -> None:
        self.history[f"epoch_{epoch}"] = {
            "delta_1": his_stats["delta_1"],
            "delta_2": his_stats["delta_2"],
            "delta_3": his_stats["delta_3_mean"],
        }
