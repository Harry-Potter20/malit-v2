# MALIT: An Intelligent Malaria Screening System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0](https://img.shields.io/badge/PyTorch-2.0-orange.svg)](https://pytorch.org/)
[![Paper](https://img.shields.io/badge/paper-under%20review-red.svg)]()

> **MALIT** — Multi-scale Attention with Learnable channel Inhibition and Texture-aware filters

Official implementation of *"MALIT: An Intelligent Malaria Screening System with Learnable Channel Competitive Inhibition, Interpretable Gabor Texture Filters, and Confidence-Gated Escalation"* — under review at *Expert Systems with Applications*.

---

## Overview

MALIT is a biologically-inspired CNN for automated malaria cell classification from Giemsa-stained thin blood smear images. It is designed for **deployment utility**, not just benchmark accuracy: every architectural component produces directly auditable outputs, and the system includes a confidence-gated escalation protocol that flags low-confidence predictions for human microscopist review.

### Key contributions

- **LCCI (Learnable Channel Competitive Inhibition)** — a winner-take-more channel gain-control module with 79.4% dominance ratio, producing per-prediction visualisable feature compression maps
- **Learnable Gabor front-end** — 32 filters with learnable frequency, orientation, bandwidth, and aspect ratio; self-organises into broad spectral coverage without explicit constraints
- **Dual channel-spatial attention** — individually extractable maps; channel entropy separates by class consistently across seeds
- **Multi-scale dilated aggregator** — four parallel branches with receptive fields {1, 3, 5, 7} pixels, covering sub-nuclear inclusions to whole-cell morphology
- **Confidence-gated escalation** — combines ensemble CI width and k-NN embedding consistency to flag ambiguous cases without threshold tuning

---

## Results

Evaluated on the [NIH Malaria Cell Images Dataset](https://lhncbc.nlm.nih.gov/LHC-research/LHC-projects/image-processing/malaria-screener.html) (27,558 cells, N = 4,134 test set), three-seed protocol (seeds 42, 123, 456).

### Performance comparison

| Model | n seeds | Acc (%) ±std | F1 (%) ±std | AUROC (%) | ECE | Params (M) |
|---|---|---|---|---|---|---|
| **CBAM-ResNet-50** | 3 | **97.57 ±0.23** | **97.57 ±0.23** | **99.57 ±0.02** | 0.0119 | 23.5 |
| EfficientNet-B0 | 3 | 97.46 ±0.06 | 97.46 ±0.06 | 99.54 ±0.01 | **0.0060** | 4.0 |
| SE-Net-ResNet-50 | 3 | 97.21 ±0.44 | 97.21 ±0.44 | 99.54 ±0.01 | 0.0056 | 23.5 |
| MobileNetV3-Small | 3 | 96.86 ±0.46 | 96.86 ±0.46 | 99.35 ±0.01 | 0.0095 | 1.5 |
| ResNet-50 | 3 | 96.55 ±0.49 | 96.55 ±0.49 | 99.27 ±0.02 | 0.0101 | 23.5 |
| **MALIT (ours)** | 3 | 96.78 ±0.49 | 96.77 ±0.55 | 99.21 ±0.08 | 0.0089 | **7.8** |

### Per-seed MALIT metrics

| Seed | Acc (%) | F1 (%) | AUROC (%) | ECE | Sens (%) | Spec (%) |
|---|---|---|---|---|---|---|
| 42 | 97.07 | 97.07 | 99.22 | 0.0079 | 96.86 | 97.29 |
| 123 | 96.15 | 96.14 | 99.13 | 0.0108 | 95.84 | 96.47 |
| **456 ★** | **97.12** | **97.10** | **99.29** | 0.0080 | 96.52 | **97.73** |
| **Mean ±std** | 96.78 ±0.49 | 96.77 ±0.55 | 99.21 ±0.08 | 0.0089 | 96.40 ±0.52 | 97.16 ±0.65 |

★ = best seed. Statistical comparisons: CBAM significantly outperforms MALIT (McNemar p = 0.008). All other comparisons inconclusive (p > 0.05, TOST equivalence not confirmed).

### Interpretability metrics

| Component | Finding | Value |
|---|---|---|
| LCCI | Channel dominance ratio | 79.4% |
| LCCI | Top-20 channel amplification | ×3.4–4.0 |
| Gabor | Orientation coverage | Self-organised, 0–180° |
| Gabor | Frequency modes | 4 discrete (4, 8, 12, 16 cycles/patch) |
| Attention | Channel entropy — parasitised | ≈5.53 bits |
| Attention | Channel entropy — uninfected | ≈5.56 bits |
| Embedding | k-NN consistency (k=5) | 94.6% ±16.7% |
| Ensemble | Cross-seed agreement | 99.03% |
| Ensemble | CI width–error correlation r | 0.39 |

---

## Architecture

```
Input (224×224×3)
    │
    ▼
┌─────────────────────────────┐
│  Learnable Gabor Front-End  │  32 filters, 7×7 kernels
│  f, θ, σ, γ all learnable   │  Dynamic kernel recomputation
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│   EfficientNet-B0 Backbone  │  ImageNet pretrained
│   Stages 1–2 frozen         │  Output: (B, 1280, 7, 7)
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│   LCCI (1280 channels)      │  Winner-take-more gate
│   wᵢₙₕ ∈ ℝ¹²⁸⁰             │  79.4% dominance ratio
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│   Dual Channel-Spatial      │  SE-Net channel attention
│   Attention                 │  + 1×1 spatial attention
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────┐
│            Multi-Scale Dilated Aggregator            │
│  d=0 (RF=1px) │ d=1 (RF=3px) │ d=2 (RF=5px) │ d=3  │
│  1280→128     │ 1280→128     │ 1280→128     │ (7px)│
│  +LCCI        │ +LCCI        │ +LCCI        │+LCCI │
└───────────────────────────┬─────────────────────────┘
                            │  concat → 512ch → 1×1 conv
                            ▼
              ┌─────────────────────────┐
              │   MLP Classifier        │
              │   512→256→128→2         │
              └─────────────┬───────────┘
                            │
                    ┌───────┴────────┐
                    │                │
              Prediction      Escalation signals
              (parasitised /  (CI width, k-NN
               uninfected)     consistency)
```

**Total trainable parameters: 7.8 M**

---

## Repository Structure

```
MALIT/
├── model/
│   ├── malit.py              # Full MALIT architecture
│   ├── lcci.py               # Learnable Channel Competitive Inhibition module
│   ├── gabor.py              # Learnable Gabor filter front-end
│   ├── attention.py          # Dual channel-spatial attention
│   └── aggregator.py         # Multi-scale dilated feature aggregator
├── train.py                  # Training script (all seeds)
├── evaluate.py               # Evaluation: metrics, McNemar, TOST, bootstrap CI
├── inference.py              # Single-image inference with escalation output
├── escalation.py             # Confidence-gated escalation protocol
├── interpretability/
│   ├── gradcam.py            # GradCAM++ visualisation
│   ├── lcci_analysis.py      # LCCI energy map analysis
│   ├── gabor_analysis.py     # Gabor parameter and decomposition analysis
│   ├── attention_entropy.py  # Channel attention entropy computation
│   └── embedding_cbr.py      # k-NN embedding consistency (CBR)
├── baselines/
│   └── train_baselines.py    # Train all 5 baselines under identical protocol
├── weights/
│   ├── malit_seed42.pth
│   ├── malit_seed123.pth
│   └── malit_seed456.pth
├── results/
│   ├── full_statistics_report.json
│   ├── ensemble_summary.json
│   ├── cbr_summary.json
│   └── lcci_dominance_ratio.json
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/Harry-Potter20/MALIT.git
cd MALIT

# Create environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
torch==2.0.0
torchvision==0.15.0
numpy==1.24.3
pillow==9.5.0
scikit-learn==1.2.2
scipy==1.10.1
matplotlib==3.7.1
imagehash==4.3.1
tqdm==4.65.0
```

---

## Dataset

Download the [NIH Malaria Cell Images Dataset](https://lhncbc.nlm.nih.gov/LHC-research/LHC-projects/image-processing/malaria-screener.html):

```
data/
└── cell_images/
    ├── Parasitized/    # 13,779 images
    └── Uninfected/     # 13,779 images
```

The training script handles stratified splitting, deduplication via perceptual hashing, and fixed random seeds automatically.

---

## Training

### Train MALIT (all three seeds)

```bash
python train.py \
    --data_dir data/cell_images \
    --seeds 42 123 456 \
    --epochs 50 \
    --batch_size 32 \
    --lr 3e-4 \
    --weight_decay 1e-4 \
    --output_dir weights/
```

### Train a single seed

```bash
python train.py \
    --data_dir data/cell_images \
    --seeds 42 \
    --output_dir weights/
```

### Train all baselines

```bash
python baselines/train_baselines.py \
    --data_dir data/cell_images \
    --seeds 42 123 456 \
    --models efficientnet_b0 senet cbam mobilenetv3 resnet50
```

---

## Evaluation

### Full statistical evaluation (McNemar, TOST, bootstrap CI)

```bash
python evaluate.py \
    --data_dir data/cell_images \
    --weights_dir weights/ \
    --seeds 42 123 456 \
    --output_dir results/
```

This reproduces all tables in the paper including per-seed metrics, McNemar's exact test between MALIT and all baselines, TOST equivalence testing (margin ±0.5% F1), and 95% bootstrap CIs (5,000 resamples).

---

## Inference

### Single image

```bash
python inference.py \
    --image path/to/cell.png \
    --weights weights/malit_seed456.pth \
    --output_dir outputs/
```

Output includes:
- Predicted class and softmax probability
- GradCAM++ activation overlay
- LCCI channel energy map
- Spatial attention map
- Channel attention entropy (interpretability index)

### Ensemble inference with escalation

```bash
python inference.py \
    --image path/to/cell.png \
    --weights weights/malit_seed42.pth weights/malit_seed123.pth weights/malit_seed456.pth \
    --ensemble \
    --escalation_threshold 0.75 \
    --output_dir outputs/
```

The escalation output flags the prediction if ensemble CI width exceeds the threshold percentile, indicating the case should be reviewed by a human microscopist.

---

## Interpretability Analysis

### LCCI channel energy maps

```bash
python interpretability/lcci_analysis.py \
    --weights weights/malit_seed456.pth \
    --data_dir data/cell_images \
    --output_dir outputs/lcci/
```

Produces per-channel post/pre-LCCI energy ratios and the dominance ratio statistic.

### Gabor filter bank and parameter distributions

```bash
python interpretability/gabor_analysis.py \
    --weights weights/malit_seed456.pth \
    --output_dir outputs/gabor/
```

Produces filter bank visualisations and learned parameter (θ, λ, σ, γ) distribution plots.

### Attention entropy analysis

```bash
python interpretability/attention_entropy.py \
    --weights_dir weights/ \
    --seeds 42 123 456 \
    --data_dir data/cell_images \
    --output_dir outputs/entropy/
```

Computes per-prediction channel attention entropy across the test set, separated by class.

### k-NN embedding consistency (CBR)

```bash
python interpretability/embedding_cbr.py \
    --weights weights/malit_seed456.pth \
    --data_dir data/cell_images \
    --k 5 \
    --output_dir outputs/cbr/
```

Computes the k-NN consistency score distribution and identifies low-consistency escalation candidates.

---

## Reproducing Paper Results

All results in the paper can be reproduced exactly using fixed seeds:

```bash
# 1. Train MALIT (all seeds)
python train.py --data_dir data/cell_images --seeds 42 123 456 --output_dir weights/

# 2. Train all baselines
python baselines/train_baselines.py --data_dir data/cell_images --seeds 42 123 456

# 3. Full evaluation — reproduces all tables
python evaluate.py --data_dir data/cell_images --weights_dir weights/ --seeds 42 123 456

# 4. Interpretability figures — reproduces all figures
python interpretability/lcci_analysis.py --weights weights/malit_seed456.pth --data_dir data/cell_images
python interpretability/gabor_analysis.py --weights weights/malit_seed456.pth
python interpretability/attention_entropy.py --weights_dir weights/ --seeds 42 123 456 --data_dir data/cell_images
python interpretability/embedding_cbr.py --weights weights/malit_seed456.pth --data_dir data/cell_images
```

Pre-trained weights for all three seeds are available in `weights/`. The NIH dataset split is fixed at `random_state=42`; per-seed variance reflects model initialisation only.

---

## Citation

If you use this code or the MALIT architecture in your research, please cite:

```bibtex
@article{eke2026malit,
  title     = {{MALIT}: An Intelligent Malaria Screening System with Learnable
               Channel Competitive Inhibition, Interpretable Gabor Texture
               Filters, and Confidence-Gated Escalation},
  author    = {Eke, Chukwudi and Eke, Chinedum and Motajo, Oluwatosin},
  journal   = {Expert Systems with Applications},
  year      = {2026},
  note      = {Under review}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

The NIH Malaria Cell Images Dataset is made available by the Lister Hill National Center for Biomedical Communications, National Library of Medicine, and is in the public domain.

---

## Contact

Correspondence: c3rb3rus1@proton.me

ORCID — Chukwudi Eke: [0009-0002-2294-673X](https://orcid.org/0009-0002-2294-673X) | Oluwatosin Motajo: [0000-0002-0342-1842](https://orcid.org/0000-0002-0342-1842)
