from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CalibrationAwareLoss(nn.Module):
    """
    Calibration-Aware Loss: L = CE + lambda * ECE

    ECE is approximated via differentiable soft Gaussian binning so it is
    back-propagatable. A warmup flag (`apply_calibration=False`) disables
    the ECE term during the first N epochs when predictions are unstable.
    """

    def __init__(self, lambda_cal: float = 0.1, n_bins: int = 10):
        super().__init__()
        self.lambda_cal = lambda_cal
        self.n_bins = n_bins
        self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        apply_calibration: bool = True,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        ce_loss = self.ce(logits, targets)

        if not apply_calibration:
            return ce_loss, {"ce": ce_loss.item(), "ece": 0.0, "total": ce_loss.item()}

        ece_loss = self._soft_ece(logits, targets)
        total = ce_loss + self.lambda_cal * ece_loss

        return total, {
            "ce": ce_loss.item(),
            "ece": ece_loss.item(),
            "total": total.item(),
        }

    def _soft_ece(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        confidence = probs.max(dim=1).values
        predictions = probs.argmax(dim=1)
        correct = (predictions == targets).float()

        bin_edges = torch.linspace(0.0, 1.0, self.n_bins + 1, device=logits.device)
        bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = 1.0 / self.n_bins
        sigma = bin_width / 2.0

        dist = (confidence.unsqueeze(1) - bin_centres.unsqueeze(0)) ** 2
        weights = torch.exp(-dist / (2 * sigma ** 2))
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)

        bin_acc = (weights * correct.unsqueeze(1)).sum(dim=0)
        bin_conf = (weights * confidence.unsqueeze(1)).sum(dim=0)
        bin_n = weights.sum(dim=0)
        bin_n_safe = bin_n + 1e-6

        bin_acc_norm = bin_acc / bin_n_safe
        bin_conf_norm = bin_conf / bin_n_safe

        ece = ((bin_n / bin_n.sum()) * (bin_acc_norm - bin_conf_norm).abs()).sum()
        return ece
