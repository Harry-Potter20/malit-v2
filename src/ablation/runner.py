"""
Ablation study runner for MALIT V2.

Tests the contribution of each architectural component by disabling it
one at a time and comparing against the full model via McNemar + TOST.

Variants
--------
full          Full MALIT V2 (uses results already saved by run_full_pipeline)
no_gabor      Remove learnable Gabor preprocessing
no_lcci       Remove LCCI lateral channel inhibition
no_attention  Remove DualAttention (channel + spatial)
no_multiscale Remove MultiScaleAggregator (single receptive field)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.malit import MALITV2
from src.training.trainer import Trainer
from src.utils.config import Config
from src.utils.save import ArtifactSaver, _make_serializable
from src.utils.seed import set_seed
from src.statistics.tests import McNemarTest, TOSTEquivalence

logger = logging.getLogger(__name__)


# Each entry is the set of flags passed to MALITV2 to disable that component.
ABLATION_VARIANTS: dict[str, dict[str, bool]] = {
    "no_gabor":      {"use_gabor": False},
    "no_lcci":       {"use_lcci": False},
    "no_attention":  {"use_attention": False},
    "no_multiscale": {"use_multiscale": False},
    "no_dais":       {"use_dais": False},
    "dais_equal":    {"dais_equal": True},
}


class AblationRunner:
    """
    Trains each ablation variant and compares it against the full MALIT V2
    using McNemar (significance) and TOST (equivalence) tests.

    The full-model results are loaded from disk (written by run_full_pipeline.py)
    so there is no re-training of the complete model.

    Parameters
    ----------
    cfg         : loaded Config object
    train_loader, val_loader, test_loader : same split as MALIT training
    saver       : ArtifactSaver (same base dir as pipeline)
    device      : torch device string or object
    seeds       : seeds to use per variant (default: tier-2 [42, 123])
    """

    def __init__(
        self,
        cfg: Config,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        saver: ArtifactSaver,
        device: str | torch.device | None = None,
        seeds: list[int] | None = None,
    ):
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.saver = saver
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.seeds = seeds or cfg.training.seeds_tier2  # [42, 123]
        self._results: dict[str, Any] = {}

    # ── Public ─────────────────────────────────────────────────────────────────

    def run(self) -> "AblationRunner":
        logger.info("Ablation study: %d variants × %d seeds",
                    len(ABLATION_VARIANTS), len(self.seeds))

        for variant_name, flags in ABLATION_VARIANTS.items():
            logger.info("── Ablation variant: %s ──", variant_name)
            self._train_variant(variant_name, flags)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return self

    @property
    def results(self) -> dict[str, Any]:
        return self._results

    # ── Private ────────────────────────────────────────────────────────────────

    def _build_model(self, ablation_flags: dict[str, bool]) -> MALITV2:
        mc = self.cfg.model
        return MALITV2(
            num_classes=mc.num_classes,
            gabor_n_orientations=mc.gabor.n_orientations,
            gabor_n_scales=mc.gabor.n_scales,
            gabor_kernel_size=mc.gabor.kernel_size,
            efficientnet_backbone=mc.efficientnet.backbone,
            efficientnet_pretrained=mc.efficientnet.pretrained,
            efficientnet_freeze_blocks=mc.efficientnet.freeze_blocks,
            lcci_reduction=mc.lcci_reduction,
            attention_reduction=mc.attention_channel_reduction,
            ms_rfs=mc.multiscale_rfs,
            **ablation_flags,
        )

    def _train_variant(self, name: str, flags: dict[str, bool]) -> None:
        tc = self.cfg.training
        seed_preds: dict[int, list[int]] = {}
        seed_metrics: dict[int, dict] = {}
        seed_labels: list[int] | None = None

        out_dir = Path(self.saver.base) / "results" / "ablation" / name
        out_dir.mkdir(parents=True, exist_ok=True)

        for seed in self.seeds:
            set_seed(seed)
            model = self._build_model(flags)

            trainer = Trainer(
                model=model,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                lr=tc.optimizer.lr,
                weight_decay=tc.optimizer.weight_decay,
                epochs=tc.epochs,
                T_max=tc.scheduler.T_max,
                eta_min=tc.scheduler.eta_min,
                amp=tc.amp,
                early_stopping_patience=tc.early_stopping.patience,
                grad_clip=tc.grad_clip,
                device=self.device,
                saver=self.saver,
                model_name=f"ablation_{name}",
                seed=seed,
            )

            trainer.fit()
            metrics, preds, labels, _ = trainer.evaluate(self.test_loader)

            seed_preds[seed] = preds
            seed_metrics[seed] = metrics
            seed_labels = labels

            with open(out_dir / f"seed_{seed}_metrics.json", "w") as f:
                json.dump(_make_serializable(
                    {"seed": seed, "metrics": metrics, "predictions": preds,
                     "labels": labels}
                ), f, indent=2)

            logger.info("  %s seed=%d  F1=%.4f  Acc=%.4f",
                        name, seed, metrics.get("f1", 0), metrics.get("accuracy", 0))

            del model, trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Aggregate across seeds
        f1_vals = [m.get("f1", float("nan")) for m in seed_metrics.values()]
        agg = {
            "mean_f1": float(np.nanmean(f1_vals)),
            "std_f1":  float(np.nanstd(f1_vals, ddof=1) if len(f1_vals) >= 2 else 0.0),
        }
        logger.info("%s aggregate  F1=%.4f ± %.4f", name, agg["mean_f1"], agg["std_f1"])

        self._results[name] = {
            "seed_preds":   seed_preds,
            "seed_metrics": seed_metrics,
            "seed_labels":  seed_labels,
            "aggregate":    agg,
        }

    def compare_against_full(
        self, full_seed_preds: dict[int, list[int]], full_seed_metrics: dict[int, dict]
    ) -> dict[str, Any]:
        """
        For each variant, run McNemar (significance) and TOST (equivalence)
        against the full MALIT V2 predictions.

        Returns a dict keyed by variant name with comparison results and
        interpretation text.
        """
        comparison: dict[str, Any] = {}

        full_f1_vals = [m.get("f1", float("nan")) for m in full_seed_metrics.values()]
        full_mean_f1 = float(np.nanmean(full_f1_vals))

        for name, data in self._results.items():
            variant_preds  = data["seed_preds"]
            variant_metrics = data["seed_metrics"]
            labels = data["seed_labels"] or []

            # Shared seeds
            shared_seeds = sorted(set(full_seed_preds) & set(variant_preds))
            if not shared_seeds or not labels:
                comparison[name] = {"error": "no shared seeds or labels"}
                continue

            # McNemar on majority-vote seed (pick seed with most data, default first)
            ref_seed = shared_seeds[0]
            fp = full_seed_preds[ref_seed]
            vp = variant_preds[ref_seed]

            if len(fp) == len(labels) and len(vp) == len(labels):
                mcnemar = McNemarTest.test(fp, vp, labels)
            else:
                mcnemar = {"error": "length mismatch", "significant": False}

            # Multi-seed McNemar
            multi = McNemarTest.multi_seed_mcnemar(
                full_seed_preds, variant_preds, labels
            )

            # F1 delta
            v_f1_vals = [m.get("f1", float("nan")) for m in variant_metrics.values()]
            v_mean_f1 = float(np.nanmean(v_f1_vals))
            delta_f1  = full_mean_f1 - v_mean_f1  # positive = full is better

            # TOST — use only shared seeds so arrays are equal length
            shared_for_tost = sorted(set(full_seed_metrics) & set(variant_metrics))
            full_f1_scores    = [full_seed_metrics[s].get("f1", float("nan")) for s in shared_for_tost]
            variant_f1_scores = [variant_metrics[s].get("f1", float("nan")) for s in shared_for_tost]
            tost = TOSTEquivalence.test(
                full_f1_scores, variant_f1_scores,
                margin=self.cfg.statistics.tost_margin,
                alpha=self.cfg.statistics.tost_alpha,
            )

            # Interpretation
            if delta_f1 > 0.005 and multi.get("majority_sig", False):
                verdict = "significant_contribution"
            elif delta_f1 > 0.002:
                verdict = "marginal_contribution"
            elif tost.get("equivalent") and not tost.get("indeterminate"):
                verdict = "no_contribution_equivalent"
            else:
                verdict = "inconclusive"

            comparison[name] = {
                "variant": name,
                "full_mean_f1": round(full_mean_f1, 4),
                "variant_mean_f1": round(v_mean_f1, 4),
                "delta_f1": round(delta_f1, 4),
                "mcnemar": mcnemar,
                "multi_seed_mcnemar": multi,
                "tost": tost,
                "verdict": verdict,
            }
            logger.info(
                "  %s: ΔF1=%.4f  McNemar_sig=%s  verdict=%s",
                name, delta_f1,
                multi.get("majority_sig", "?"), verdict,
            )

        return comparison

    def save_ablation_table(
        self,
        comparison: dict[str, Any],
        full_mean_f1: float,
        out_dir: str | Path,
    ) -> Path:
        """Save a CSV table: variant | mean_F1 | ΔF1 | McNemar_sig | verdict"""
        import csv
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "ablation_table.csv"

        rows = [
            {
                "variant": "full (MALIT V2)",
                "mean_F1": round(full_mean_f1, 4),
                "delta_F1": 0.0,
                "mcnemar_sig": "—",
                "verdict": "reference",
            }
        ]
        for name, res in comparison.items():
            if "error" in res:
                continue
            multi = res.get("multi_seed_mcnemar", {})
            rows.append({
                "variant": name,
                "mean_F1": res["variant_mean_f1"],
                "delta_F1": res["delta_f1"],
                "mcnemar_sig": str(multi.get("majority_sig", "?")),
                "verdict": res["verdict"],
            })

        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["variant", "mean_F1", "delta_F1", "mcnemar_sig", "verdict"]
            )
            writer.writeheader()
            writer.writerows(rows)

        logger.info("Ablation table → %s", out_path)
        return out_path
