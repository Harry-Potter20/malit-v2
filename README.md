# MALIT V2

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0](https://img.shields.io/badge/PyTorch-2.0-orange.svg)](https://pytorch.org/)
[![Kaggle](https://img.shields.io/badge/compute-Kaggle%20GPU-20BEFF.svg)](https://kaggle.com)
[![Paper](https://img.shields.io/badge/paper-under%20review-red.svg)]()

> **MALIT** — Multi-scale Attention with Learnable channel Inhibition and Texture-aware filters

Official implementation of *"MALIT: An Intelligent Malaria Screening System with Learnable Channel Competitive Inhibition, Interpretable Gabor Texture Filters, and Confidence-Gated Escalation"

---

## Overview

MALIT V2 is a biologically-inspired CNN for automated malaria cell classification from Giemsa-stained thin blood smear images. Beyond classification accuracy, the system is designed for **clinical deployment utility**: every architectural component produces directly auditable outputs, and a confidence-gated escalation protocol flags low-confidence predictions for human microscopist review.

The current version incorporates two major architectural additions:

- **HIS (Hierarchical Inhibition Scheduling)** — extends single-point LCCI into a three-depth inhibition hierarchy with positive-constrained learnable depth scaling (Softplus), allowing inhibition strength to emerge dynamically across architectural depth without hard-coded constraints.
- **CAL (Calibration-Aware Learning)** — augments the cross-entropy objective with a differentiable ECE term (`L = CE + λ·ECE`, λ=0.1) using soft Gaussian binning, with a 3-epoch warmup before ECE regularisation activates.

### Key contributions

- **HIS — Hierarchical Inhibition Scheduling** — three LCCI depths: after Gabor preprocessing (depth 1), after EfficientNet backbone (depth 2), and inside each multi-scale aggregator branch (depth 3). Each depth has an independent positive-constrained scaling factor δ_d = softplus(raw). The ordering δ₁ > δ₂ > δ₃ is not imposed but expected to emerge from optimisation.
- **LCCI (Lateral Competitive Cortical Inhibition)** — `LCCI(x) = ReLU(x + g(x) ⊙ (x ⊙ δ·w_inh))` where g is a squeeze-excite sigmoid gate and δ is the Softplus-constrained depth scaling factor; 79.4% dominance ratio on backbone features.
- **Learnable Gabor front-end** — 32 filters with learnable θ, λ, σ, γ; self-organises into broad spectral coverage without explicit constraints.
- **Dual channel-spatial attention** — independently extractable channel and spatial attention maps; channel entropy separates by class consistently across seeds.
- **Multi-scale dilated aggregator** — four parallel `AggregatorBranch` modules at receptive fields {1, 3, 5, 7} pixels, each containing an independent LCCI module (HIS depth 3).
- **CAL — Calibration-Aware Learning** — differentiable ECE via soft Gaussian binning back-propagated alongside cross-entropy; per-epoch tracking of ECE, Brier score, NLL, and confidence statistics.
- **Confidence-gated escalation** — combines ensemble CI width and k-NN embedding consistency to flag ambiguous cases for human review.

---

## Results

Evaluated on the [NIH Malaria Cell Images Dataset](https://lhncbc.nlm.nih.gov/LHC-research/LHC-projects/image-processing/malaria-screener.html) (27,558 cells, N = 4,134 test set), three-seed protocol (seeds 42, 123, 456).

### Performance comparison

| Model | n | Acc (%) ±std | F1 (%) ±std | AUROC (%) | ECE | Params (M) |
|---|---|---|---|---|---|---|
| **CBAM-ResNet-50** | 3 | **97.57 ±0.23** | **97.57 ±0.23** | **99.57 ±0.02** | 0.0119 | 23.5 |
| EfficientNet-B0 | 3 | 97.46 ±0.06 | 97.46 ±0.06 | 99.54 ±0.01 | **0.0060** | 4.0 |
| SE-Net-ResNet-50 | 3 | 97.21 ±0.44 | 97.21 ±0.44 | 99.54 ±0.01 | 0.0056 | 23.5 |
| MobileNetV3-Small | 3 | 96.86 ±0.46 | 96.86 ±0.46 | 99.35 ±0.01 | 0.0095 | 1.5 |
| ResNet-50 | 3 | 96.55 ±0.49 | 96.55 ±0.49 | 99.27 ±0.02 | 0.0101 | 23.5 |
| **MALIT V2 (ours)** | 3 | 96.78 ±0.49 | 96.77 ±0.55 | 99.21 ±0.08 | 0.0089 | **7.8** |

Statistical comparisons: CBAM significantly outperforms MALIT (McNemar p = 0.008, majority-seed criterion). All other comparisons are inconclusive (McNemar p > 0.05; TOST equivalence not confirmed). Results reported without selective framing.

### Per-seed MALIT metrics

| Seed | Acc (%) | F1 (%) | AUROC (%) | ECE | Sens (%) | Spec (%) |
|---|---|---|---|---|---|---|
| 42 | 97.07 | 97.07 | 99.22 | 0.0079 | 96.86 | 97.29 |
| 123 | 96.15 | 96.14 | 99.13 | 0.0108 | 95.84 | 96.47 |
| **456 ★** | **97.12** | **97.10** | **99.29** | 0.0080 | 96.52 | **97.73** |
| Mean ±std | 96.78 ±0.49 | 96.77 ±0.55 | 99.21 ±0.08 | 0.0089 | 96.40 ±0.52 | 97.16 ±0.65 |

★ = best seed.

### Interpretability metrics

| Property | Value |
|---|---|
| LCCI dominance ratio (backbone) | 79.4% |
| LCCI top-channel amplification | ×3.4–4.0 |
| Gabor orientation coverage | Self-organised 0–180° |
| Channel entropy — parasitised | ≈5.53 bits |
| Channel entropy — uninfected | ≈5.56 bits |
| k-NN embedding consistency (k=5) | 94.6% ±16.7% |
| Ensemble cross-seed agreement | 99.03% |
| CI width–error correlation r | 0.39 |

---

## Architecture

### HIS Forward Pass

```
Input (224×224×3)
         │
         ▼
┌────────────────────────────────┐
│   Learnable Gabor Front-End    │  32 filters (8 orient. × 4 scales), 15×15 kernels
│   θ, λ, σ, γ all learnable     │  dynamic kernel recomputation each forward pass
└──────────────┬─────────────────┘
               ▼
┌────────────────────────────────┐
│   LCCI — Depth 1  (gabor_lcci) │  channels=3
│   δ₁ = softplus(raw₁)         │  effective_w = δ₁ · w_inh
│   LCCI(x) = ReLU(x + g(x)     │  g: squeeze-excite sigmoid gate
│             ⊙ (x ⊙ δ₁·w_inh)) │
└──────────────┬─────────────────┘
               ▼
┌────────────────────────────────┐
│   EfficientNet-B0 Backbone     │  ImageNet pretrained
│   Stages 0–3 frozen            │  output: (B, 1280, 7, 7)
└──────────────┬─────────────────┘
               ▼
┌────────────────────────────────┐
│   LCCI — Depth 2 (backbone_lcci│  channels=1280
│   δ₂ = softplus(raw₂)         │  79.4% dominance ratio
│   LCCI(x) = ReLU(x + g(x)     │
│             ⊙ (x ⊙ δ₂·w_inh)) │
└──────────────┬─────────────────┘
               ▼
┌────────────────────────────────┐
│   Dual Channel-Spatial Attn    │  SE-Net channel + 1×1 spatial, in parallel
│   both maps individually       │  channel entropy H reported per prediction
│   extractable                  │
└──────────────┬─────────────────┘
               ▼
┌──────────────────────────────────────────────────────────┐
│            Multi-Scale Aggregator (4 × AggregatorBranch)  │
│                                                           │
│  RF=1px (d=0)  │  RF=3px (d=1)  │  RF=5px (d=2)  │  RF=7px (d=3) │
│  1×1 conv      │  3×3 dw-sep    │  3×3 dw-sep    │  3×3 dw-sep   │
│  BN + ReLU     │  BN + ReLU     │  BN + ReLU     │  BN + ReLU    │
│  LCCI depth 3  │  LCCI depth 3  │  LCCI depth 3  │  LCCI depth 3 │
│  δ₃ per branch │                │                │               │
└───────────────────────────────┬──────────────────────────────────┘
                                │  concat (4 × 1280) → 1×1 conv+BN+ReLU → 1280ch
                                ▼
                   ┌──────────────────────┐
                   │  Global Average Pool │  (B, 1280)
                   └──────────┬───────────┘
                               ▼
                   ┌──────────────────────┐
                   │  MLP Classifier      │  1280 → 640 → 2
                   │  BN + ReLU + Drop    │  7.8 M total params
                   └──────────┬───────────┘
                               │
                 ┌─────────────┴──────────────┐
           Prediction                 Calibration signals
         (CE + λ·ECE)         (reliability diagram, Brier, NLL,
                               confidence histogram, per-sample CSV)
```

### HIS Depth Scaling

Each `LCCIModule` carries an independent, positive-constrained depth scaling scalar:

```
δ_d = softplus(depth_scale_raw_d)   ← guarantees δ_d > 0
effective_w_inh = δ_d · w_inh
LCCI(x) = ReLU(x + g(x) ⊙ (x ⊙ effective_w_inh))
```

After training, `extract_his_stats(model)` reports δ₁, δ₂, δ₃ and dominance ratios at every depth. The hierarchy δ₁ > δ₂ > δ₃ is not enforced — it must emerge from optimisation.

### CAL Loss

```
L = L_CE  +  λ · ECE_soft        (λ = 0.1 default)

ECE_soft — differentiable approximation via soft Gaussian bin assignment:
  weights_ij = exp(-||conf_i - centre_j||² / 2σ²) / Z_i
  ECE = Σ_j (n_j/N) · |acc_j - conf_j|

Warmup: epochs 1–3 use CE only. CAL activates from epoch 4 onward.
```

---

## Repository Structure

```
malit-v2/
├── configs/
│   └── malit_v2.yaml               ← all hyperparameters incl. lambda_cal, cal_warmup_epochs
│
├── kaggle/
│   ├── run.py                      ← full pipeline (MALIT + baselines + ablation)
│   ├── run_baselines_only.py       ← baselines without MALIT retraining
│   ├── run_ablation_only.py        ← ablation without MALIT retraining
│   └── requirements.txt
│
├── scripts/
│   ├── run_full_pipeline.py        ← MALIT training + HIS analysis + visualisations
│   ├── run_baselines.py            ← 5 baselines + McNemar/TOST comparison tables
│   └── run_ablation.py             ← 4 ablation variants + component contribution table
│
├── src/
│   ├── models/
│   │   ├── malit.py                ← MALITV2: 3-depth HIS pipeline
│   │   ├── lcci.py                 ← LCCIModule with HIS depth scaling (Softplus)
│   │   ├── multiscale.py           ← MultiScaleAggregator + AggregatorBranch (LCCI depth 3)
│   │   ├── attention.py            ← DualAttention (channel + spatial)
│   │   └── gabor.py                ← LearnableGaborLayer
│   │
│   ├── training/
│   │   ├── trainer.py              ← Trainer: CAL warmup, calibration tracking, HIS dynamics
│   │   ├── runner.py               ← MultiSeedRunner: per-sample prediction CSV
│   │   └── cal_loss.py             ← CalibrationAwareLoss (CE + λ·ECE_soft)
│   │
│   ├── interpretability/
│   │   ├── his_analysis.py         ← extract_his_stats(): δ values + dominance ratios
│   │   ├── his_tracking.py         ← HISTracker: per-epoch δ₁/δ₂/δ₃ history
│   │   ├── gradcam.py
│   │   ├── gabor_viz.py
│   │   ├── lcci_viz.py
│   │   └── attention_viz.py
│   │
│   ├── visualization/
│   │   ├── reliability.py          ← plot_reliability_diagram()
│   │   └── confidence_histogram.py ← plot_confidence_histogram()
│   │
│   ├── ablation/
│   │   └── runner.py               ← AblationRunner: no_gabor/lcci/attention/multiscale
│   │
│   ├── baselines/                  ← EfficientNet-B0, ResNet-50, CBAM, SE-Net, MobileNetV3
│   ├── evaluation/                 ← EvaluationMetrics (F1, AUROC, Brier, NLL, ECE)
│   ├── explainability/             ← BayesianUncertainty, EnsembleConfidence, CBR
│   ├── statistics/                 ← McNemarTest, TOSTEquivalence, StatisticsReport
│   └── utils/
│       ├── config.py               ← Config dataclasses incl. lambda_cal, cal_warmup_epochs
│       ├── save.py                 ← ArtifactSaver
│       └── prediction_logging.py  ← PredictionLogger → per-sample CSV
│
└── tests/                          ← 325 tests
```

---

## Pipeline Outputs

After a full run, `results/` contains:

| File | Description |
|---|---|
| `statistics/seed_N_metrics.json` | Per-seed F1, AUROC, Brier, NLL, ECE, Sensitivity, Specificity |
| `statistics/aggregate_metrics.json` | Mean ± std across all seeds |
| `statistics/calibration_curves_seedN.json` | Per-epoch train ECE loss, val ECE, Brier, NLL, confidence mean/var |
| `statistics/his_stats_seedN.json` | Final δ₁, δ₂, δ₃ values and dominance ratios per depth |
| `statistics/his_dynamics_seedN.json` | δ₁/δ₂/δ₃ tracked every epoch — shows emergent inhibition hierarchy |
| `statistics/sample_predictions_seedN.csv` | Per-sample: image_id, true label, predicted label, confidence, entropy, correct |
| `statistics/full_statistics_report.json` | McNemar + TOST multi-seed report (MALIT vs baselines) |
| `plots/reliability_seedN.png` | Reliability diagram — model confidence vs empirical accuracy |
| `plots/confidence_hist_seedN.png` | Confidence distribution histogram |
| `plots/gabor_filter_bank.png` | All 32 learned Gabor kernels |
| `plots/gabor_param_distributions.png` | Distributions of θ, λ, σ, γ after training |
| `plots/gabor_decomposition.png` | Per-channel Gabor responses on a sample image |
| `plots/lcci_channel_suppression.png` | Pre/post LCCI energy ratio per channel |
| `gradcam/` | GradCAM overlays for test images |
| `attention_maps/` | Spatial attention map visualisations |
| `ablation/ablation_table.csv` | Variant, mean F1, ΔF1, McNemar sig, verdict |
| `tables/baselines_comparison.csv` | Baseline comparison table |

---

## Running on Kaggle (recommended)

This repo is built to run entirely on Kaggle free GPU. One command does everything.

### Step 1 — Add the dataset

In your Kaggle notebook go to **Add Data → Search** and add the  
[Cell Images for Detecting Malaria](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria) dataset.

### Step 2 — Add this repo

Go to **Add Data → GitHub** and link `Harry-Potter20/malit-v2`, or upload it as a Kaggle dataset.

### Step 3 — Enable GPU

Go to **Settings → Accelerator → GPU T4 x2** (or P100).

### Step 4 — Run

```python
!python kaggle/run.py
```

That is the complete workflow. `run.py` handles everything automatically:

| Step | What it does |
|---|---|
| Setup | Finds dataset (recursive search under `/kaggle/input`), validates GPU, installs packages, sets env vars |
| 1/3 MALIT | `run_full_pipeline.py` — 3-seed training with HIS+CAL, HIS analysis, uncertainty, CBR, all visualisations |
| 2/3 Baselines | `run_baselines.py` — 5 baselines under identical protocol + McNemar/TOST comparison tables |
| 3/3 Ablation | `run_ablation.py` — 4 variants (no_gabor/no_lcci/no_attention/no_multiscale) + component contribution table |
| Package | Zips everything in `/kaggle/working/results/` to `malit_results.zip` |

Download `malit_results.zip` from the **Output** tab when the run completes.

### Standalone entrypoints

If MALIT training has already been completed and you only need to re-run a specific stage:

```python
!python kaggle/run_baselines_only.py    # baselines only — does not retrain MALIT
!python kaggle/run_ablation_only.py     # ablation only  — requires existing seed_*_metrics.json
```

> If the dataset cannot be located, `run.py` prints the exact `find` command to help you identify the correct mount path:
> ```
> !find /kaggle/input -name '*.png' | head -3
> ```

---

## Running Locally

```bash
git clone https://github.com/Harry-Potter20/malit-v2.git
cd malit-v2

pip install -r requirements.txt

export DATASET_PATH=/path/to/cell_images   # must contain Parasitized/ and Uninfected/
export OUTPUT_DIR=/path/to/output

PYTHONPATH=. python scripts/run_full_pipeline.py
PYTHONPATH=. python scripts/run_baselines.py
PYTHONPATH=. python scripts/run_ablation.py
```

**Quick mode** (1 seed, 1 epoch, 10% data — smoke test):

```bash
PYTHONPATH=. python scripts/run_full_pipeline.py --quick
```

**Requirements:** Python 3.9+, PyTorch 2.0, CUDA 12.1, GPU with ≥8 GB VRAM recommended.

---

## Ablation Variants

| Variant | Disabled component | Flag |
|---|---|---|
| `no_gabor` | Learnable Gabor front-end + depth-1 LCCI | `use_gabor=False` |
| `no_lcci` | All LCCI modules (depths 1, 2, 3) | `use_lcci=False` |
| `no_attention` | Dual channel-spatial attention | `use_attention=False` |
| `no_multiscale` | Multi-scale aggregator (single-branch fallback) | `use_multiscale=False` |

Comparisons use McNemar (significance) + TOST (equivalence) against the full model, on shared seeds only to guarantee paired tests.

---

## Dataset

The [NIH Malaria Cell Images Dataset](https://lhncbc.nlm.nih.gov/LHC-research/LHC-projects/image-processing/malaria-screener.html) contains 27,558 cell images (13,779 parasitised, 13,779 uninfected) from Giemsa-stained thin blood smear preparations. Public domain — no access permissions required.

Expected directory structure (resolved automatically by `run.py` on Kaggle):

```
cell_images/
├── Parasitized/    # 13,779 PNG images
└── Uninfected/     # 13,779 PNG images
```

---

## Citation

```bibtex
@article{eke2026malit,
  title   = {{MALIT}: An Intelligent Malaria Screening System with Learnable
             Channel Competitive Inhibition, Interpretable {Gabor} Texture
             Filters, and Confidence-Gated Escalation},
  author  = {Eke, Chukwudi and Eke, Chinedum and Motajo, Oluwatosin},
  journal = {Expert Systems with Applications},
  year    = {2026},
  note    = {Under review}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

The NIH Malaria Cell Images Dataset is in the public domain, made available by the Lister Hill National Center for Biomedical Communications, National Library of Medicine.

---

## Contact

Correspondence: c3rb3rus1@proton.me

ORCID — Chukwudi Eke: [0009-0002-2294-673X](https://orcid.org/0009-0002-2294-673X) · Oluwatosin Motajo: [0000-0002-0342-1842](https://orcid.org/0000-0002-0342-1842)
