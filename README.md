# MALIT V2

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0](https://img.shields.io/badge/PyTorch-2.0-orange.svg)](https://pytorch.org/)
[![Kaggle](https://img.shields.io/badge/compute-Kaggle%20GPU-20BEFF.svg)](https://kaggle.com)
[![Paper](https://img.shields.io/badge/paper-under%20review-red.svg)]()

> **MALIT** — Multi-scale Attention with Learnable channel Inhibition and Texture-aware filters

Official implementation of *"MALIT: An Intelligent Malaria Screening System with Learnable Channel Competitive Inhibition, Interpretable Gabor Texture Filters, and Confidence-Gated Escalation"* — under review at *Expert Systems with Applications*.

---

## Overview

MALIT is a biologically-inspired CNN for automated malaria cell classification from Giemsa-stained thin blood smear images. Beyond classification accuracy, the system is designed for **deployment utility**: every architectural component produces directly auditable outputs, and a confidence-gated escalation protocol flags low-confidence predictions for human microscopist review without requiring threshold tuning.

### Key contributions

- **LCCI (Learnable Channel Competitive Inhibition)** — winner-take-more channel gain-control module with 79.4% dominance ratio; produces per-prediction visualisable feature compression maps
- **Learnable Gabor front-end** — 32 filters with learnable θ, λ, σ, γ; self-organises into broad spectral coverage without explicit constraints
- **Dual channel-spatial attention** — individually extractable maps; channel entropy separates by class consistently across seeds
- **Multi-scale dilated aggregator** — four parallel branches with receptive fields {1, 3, 5, 7} pixels at 7×7 feature resolution
- **Confidence-gated escalation** — combines ensemble CI width and k-NN embedding consistency to flag ambiguous cases

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
| **MALIT (ours)** | 3 | 96.78 ±0.49 | 96.77 ±0.55 | 99.21 ±0.08 | 0.0089 | **7.8** |

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
| LCCI dominance ratio | 79.4% |
| LCCI top-channel amplification | ×3.4–4.0 |
| Gabor orientation coverage | Self-organised 0–180° |
| Channel entropy — parasitised | ≈5.53 bits |
| Channel entropy — uninfected | ≈5.56 bits |
| k-NN embedding consistency (k=5) | 94.6% ±16.7% |
| Ensemble cross-seed agreement | 99.03% |
| CI width–error correlation r | 0.39 |

---

## Architecture

```
Input (224×224×3)
    │
    ▼
┌──────────────────────────────┐
│  Learnable Gabor Front-End   │  32 filters, 7×7 kernels
│  θ, λ, σ, γ all learnable    │  Dynamic kernel recomputation per forward pass
└─────────────┬────────────────┘
              ▼
┌──────────────────────────────┐
│  EfficientNet-B0 Backbone    │  ImageNet pretrained
│  Stages 1–2 frozen           │  Output: (B, 1280, 7, 7)
└─────────────┬────────────────┘
              ▼
┌──────────────────────────────┐
│  LCCI (1280 channels)        │  LCCI(x) = ReLU(x + g(x) ⊙ (x ⊙ w_inh))
│  w_inh ∈ ℝ¹²⁸⁰ learnable    │  79.4% dominance ratio
└─────────────┬────────────────┘
              ▼
┌──────────────────────────────┐
│  Dual Channel-Spatial Attn   │  SE-Net channel + 1×1 spatial, in parallel
│  Both maps individually      │  Channel entropy H reported per prediction
│  extractable                 │
└─────────────┬────────────────┘
              ▼
┌────────────────────────────────────────────────────────┐
│             Multi-Scale Dilated Aggregator              │
│  d=0 → RF=1px  │  d=1 → RF=3px  │  d=2 → RF=5px  │ d=3│
│  1280→128+LCCI │  1280→128+LCCI │  1280→128+LCCI │    │
└──────────────────────────┬─────────────────────────────┘
                           │  concat → 512ch → 1×1 conv+BN+ReLU+Drop
                           ▼
             ┌─────────────────────────┐
             │  MLP: 512→256→128→2     │  7.8 M total params
             └────────────┬────────────┘
                          │
                 ┌────────┴──────────┐
           Prediction          Escalation signals
                           (ensemble CI width,
                            k-NN consistency)
```

---

## Repository Structure

```
malit-v2/
├── kaggle/
│   └── run.py                  ← Single entrypoint — the ONLY file you execute
│
├── scripts/
│   ├── run_full_pipeline.py    ← MALIT V2 training + evaluation + visualisations
│   ├── run_baselines.py        ← 5 baselines training + comparison tables
│   └── run_ablation.py         ← Ablation study: 4 variants + component table
│
├── requirements.txt
└── README.md
```

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

| What it does | Detail |
|---|---|
| Finds the dataset | Searches all common Kaggle mount paths under `/kaggle/input`; falls back to recursive search for `Parasitized/` + `Uninfected/` sibling directories |
| Validates GPU | Prints device name and VRAM; aborts with a clear message if no GPU is available |
| Installs packages | Runs `pip install -r requirements.txt`; also individually verifies `imagehash`, `tqdm`, and `statsmodels` |
| Sets env vars | Exports `DATASET_PATH` and `OUTPUT_DIR`; all downstream scripts read these |
| Step 1/3 | Runs `scripts/run_full_pipeline.py` — MALIT V2 training across all three seeds + all evaluation metrics and visualisations |
| Step 2/3 | Runs `scripts/run_baselines.py` — trains all 5 baselines under identical protocols + McNemar, TOST, and bootstrap CI comparison tables |
| Step 3/3 | Runs `scripts/run_ablation.py` — trains 4 ablation variants + component contribution table |
| Packages results | Zips everything in `/kaggle/working/results/` to `/kaggle/working/malit_results.zip` |

Download `malit_results.zip` from the **Output** tab when the run completes.

> If the dataset cannot be located, `run.py` prints the exact `find` command to help you identify the correct mount path:
> ```
> !find /kaggle/input -name '*.png' | head -3
> ```

---

## Running Locally

Set the two environment variables and call each script directly:

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

**Requirements:** Python 3.9+, PyTorch 2.0, CUDA 12.1, GPU with ≥8 GB VRAM recommended.

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
