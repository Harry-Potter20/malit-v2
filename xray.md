# MALIT-H: Chest X-Ray Second Dataset Validation
## Agent Implementation Guide

This document tells the agent exactly what to do to run the chest X-ray
validation pipeline and collect the results needed for the Pattern Recognition
paper. Follow each step in order. Do not skip verification steps.

---

## Context

MALIT-H has been trained and evaluated on the NIH Malaria Cell Images Dataset.
To strengthen the Pattern Recognition submission, we need to validate that the
DAIS depth-adaptive inhibition gradient (δ₁ > δ₂) emerges on a completely
different imaging modality. The chosen second dataset is the Chest X-Ray
Pneumonia dataset (Kaggle, 5,863 images, binary: PNEUMONIA vs NORMAL).

The key question is: does training on chest X-ray images also produce
δ₁ > δ₂ without ordering constraint? If yes, the claim becomes cross-modal.

---

## Step 1 — Add the Dataset to Your Kaggle Notebook

In the Kaggle notebook interface:

1. Click **Add Data** (top right)
2. Search for: `chest-xray-pneumonia`
3. Select the dataset by **Paul Mooney** (5,863 images, 2 categories)
4. Click **Add**

The dataset will mount at:
```
/kaggle/input/chest-xray-pneumonia/chest_xray/
├── train/
│   ├── NORMAL/      # 1,341 images
│   └── PNEUMONIA/   # 3,875 images
├── val/
│   ├── NORMAL/      # 8 images  (too small — pipeline merges with train)
│   └── PNEUMONIA/   # 8 images
└── test/
    ├── NORMAL/      # 234 images
    └── PNEUMONIA/   # 390 images
```

**Verify the mount:**
```python
!find /kaggle/input -name "PNEUMONIA" -type d | head -5
```
You should see paths containing `chest-xray-pneumonia`.

---

## Step 2 — Place the Pipeline Script

Copy `run_xray_pipeline.py` into your `scripts/` directory:

```python
import shutil
shutil.copy('/kaggle/working/run_xray_pipeline.py',
            'scripts/run_xray_pipeline.py')
```

Or if you have uploaded it already to the repo, it is already at
`scripts/run_xray_pipeline.py`.

---

## Step 3 — Verify the Model Import Path

**This is the most important step.** The pipeline imports the MALIT-H model.
Open `run_xray_pipeline.py` and find this line (around line 114):

```python
from src.models.malit import MALITModel
```

Check your actual repo structure:
```python
!find . -name "*.py" | xargs grep -l "class MALIT" 2>/dev/null
```

Update the import to match. Common variants:
- `from src.models.malit import MALITModel`
- `from src.models.malit_v2 import MALITv2`
- `from src.model import MALIT`

The model must have LCCI modules with `depth_scale` attributes for the HIS
extraction to work. Verify:
```python
!python -c "
import sys; sys.path.insert(0, '.')
from src.models.malit import MALITModel   # adjust import
import torch
m = MALITModel(num_classes=2)
# Check for depth_scale
has_dais = False
for name, p in m.named_parameters():
    if 'depth_scale' in name:
        print(f'DAIS found: {name}')
        has_dais = True
if not has_dais:
    print('WARNING: No depth_scale parameters found — HIS will not be extracted')
"
```

If `depth_scale` is not found, check your LCCI module implementation and
ensure DAIS was actually added per the implementation guide.

---

## Step 4 — Set Environment Variables

```python
import os
os.environ['OUTPUT_DIR'] = '/kaggle/working/results'
# XRAY_DATASET_PATH is auto-detected — no need to set unless auto-detection fails
```

If auto-detection fails, set manually:
```python
os.environ['XRAY_DATASET_PATH'] = '/kaggle/input/chest-xray-pneumonia/chest_xray'
```

---

## Step 5 — Run the Pipeline

```python
!PYTHONPATH=. python scripts/run_xray_pipeline.py
```

Expected console output:
```
━━━ MALIT-H Chest X-Ray Validation ━━━
GPU: Tesla T4 (15.8 GB)
Dataset: /kaggle/input/chest-xray-pneumonia/chest_xray
Train+val pool: 5216 images  |  Test: 624 images
Class counts — NORMAL: 1349  PNEUMONIA: 3867
Train: 4172  Val: 1044  Test: 624
══ Seed 42 ══
Epoch  1 | val_loss=0.xxxx val_f1=0.xxxx
...
HIS: delta_1=0.xxxx delta_2=0.xxxx delta_1>delta_2=True gap=0.xxxx
Seed 42 | F1=0.xxxx AUROC=0.xxxx
══ Seed 123 ══
...
══ Seed 456 ══
...
━━━ X-RAY SUMMARY ━━━
F1:    0.xxxx ± 0.xxxx
AUROC: 0.xxxx ± 0.xxxx
DAIS:  mean_delta_1=0.xxxx  mean_delta_2=0.xxxx  delta_1>delta_2 in all seeds: True/False
Results saved to /kaggle/working/results/xray/xray_summary.json
━━━ Done in xxx.xs ━━━
```

Expected runtime: **25–40 minutes** on T4 for all three seeds.

---

## Step 6 — Collect and Verify Results

After the run completes, read the summary:

```python
import json

with open('/kaggle/working/results/xray/xray_summary.json') as f:
    summary = json.load(f)

print('=== CHEST X-RAY RESULTS ===')
print(f"Mean F1:    {summary['mean_f1']*100:.2f}% ± {summary['std_f1']*100:.2f}%")
print(f"Mean AUROC: {summary['mean_auroc']*100:.2f}% ± {summary['std_auroc']*100:.2f}%")
print()
print('=== DAIS GRADIENT (critical for PR paper) ===')
his = summary['his_summary']
print(f"delta_1 values: {his['delta_1_values']}")
print(f"delta_2 values: {his['delta_2_values']}")
print(f"mean delta_1:   {his['mean_delta_1']:.4f}")
print(f"mean delta_2:   {his['mean_delta_2']:.4f}")
print(f"delta_1 > delta_2 in ALL seeds: {his['delta_1_gt_delta_2_all_seeds']}")
print()
print('=== PER-SEED ===')
for m in summary['per_seed']:
    his_s = m['his']
    d1 = his_s.get('delta_1', 'N/A')
    d2 = his_s.get('delta_2', 'N/A')
    print(f"Seed {m['seed']}: F1={m['f1']*100:.2f}%  AUROC={m['auroc']*100:.2f}%  "
          f"delta_1={d1:.4f}  delta_2={d2:.4f}  d1>d2={his_s.get('delta_1_gt_delta_2','?')}")
```

---

## Step 7 — Download the Results

```python
import zipfile, os

zip_path = '/kaggle/working/xray_results.zip'
results_dir = '/kaggle/working/results/xray'

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            fp = os.path.join(root, file)
            zf.write(fp, os.path.relpath(fp, '/kaggle/working/results'))

print(f'Saved to {zip_path} ({os.path.getsize(zip_path)/1e6:.1f} MB)')
```

Download `xray_results.zip` from the Kaggle **Output** tab.

---

## Step 8 — Interpret the Critical Result

### If delta_1 > delta_2 in ALL THREE seeds (best case):

The paper gains the following sentence in Section 5.2 (DAIS Depth Analysis):

> "Cross-modal validation on the Chest X-Ray Pneumonia dataset (5,863 images,
> N = 624 test set) confirms that the DAIS depth gradient (δ₁ > δ₂) is not
> specific to microscopy imaging. Training on chest X-ray images (PNEUMONIA vs
> NORMAL) independently produces δ₁ = [value] ± [std] > δ₂ = [value] ± [std]
> across all three seeds without ordering constraint (Table 7). This
> cross-modal consistency strengthens the claim that depth-adaptive competitive
> inhibition is a general learned property of DAIS-equipped CNNs."

### If delta_1 > delta_2 in 2 of 3 seeds:

> "Cross-modal validation on the Chest X-Ray Pneumonia dataset shows δ₁ > δ₂
> in 2 of 3 seeds (mean δ₁ = [value], mean δ₂ = [value]), consistent with
> the malaria result (2 of 3 seeds showing full ordering). The depth gradient
> is present but not universal across initialisations."

### If delta_1 > delta_2 in 0 or 1 seeds:

The cross-modal DAIS claim does not hold. Report honestly:

> "Cross-modal validation on the Chest X-Ray Pneumonia dataset does not confirm
> the δ₁ > δ₂ gradient (mean δ₁ = [value], mean δ₂ = [value]). The DAIS
> depth gradient observed on the malaria benchmark may be specific to
> microscopy texture classification tasks. Further cross-task investigation
> is required."

In this case the paper still has value but the DAIS claim is scoped to
microscopy tasks. This is honest and publishable.

---

## Step 9 — Send Results to Claude

Upload the following files in your next message to Claude:

1. `xray_summary.json` — the main summary file
2. Per-seed JSONs: `seed_42/metrics.json`, `seed_123/metrics.json`,
   `seed_456/metrics.json`

Claude will then:
- Update the manuscript with the cross-dataset results
- Add Table 7 (chest X-ray per-seed metrics)
- Update Section 5.2 (DAIS analysis) with cross-modal finding
- Update Section 6.3 (scope and generalisability)
- Update the abstract and conclusion
- Rebuild the cover letter with the stronger cross-modal claim

---

## Troubleshooting

### "src.models.malit not found"
The script falls back to plain EfficientNet-B0 without LCCI+DAIS. This gives
F1/AUROC results but no HIS stats. Fix the import path (Step 3).

### "XRAY_DATASET_PATH not found"
Run:
```python
!find /kaggle/input -name "PNEUMONIA" -type d
```
Use the parent of the parent directory as XRAY_DATASET_PATH.

### Out of memory (OOM)
Reduce batch size in `run_xray_pipeline.py` line 19:
```python
BATCH_SIZE = 16   # reduce from 32
```

### Very low F1 (< 0.70) after seed 1
Check that the weighted sampler is being applied correctly. The dataset is
imbalanced (74% PNEUMONIA). Without weighting, the model may collapse to
predicting all-PNEUMONIA and achieve high accuracy but low macro F1.

### Training is very slow (> 2 hours)
Enable GPU in your notebook: Settings → Accelerator → GPU T4 x2.

---

## Expected Results (approximate, for sanity check)

Based on EfficientNet-B0 published results on this dataset:
- F1: 0.85 – 0.92 (macro, accounting for class imbalance)
- AUROC: 0.96 – 0.99

MALIT-H should achieve similar or slightly lower F1 given the smaller model
and CAL regularisation. If F1 < 0.75, something is wrong — check the weighted
sampler and class imbalance handling.

---

## Files Produced

```
/kaggle/working/results/xray/
├── xray_summary.json          ← main summary — upload this to Claude
├── seed_42/
│   ├── metrics.json           ← upload this to Claude
│   └── weights.pt
├── seed_123/
│   ├── metrics.json           ← upload this to Claude
│   └── weights.pt
└── seed_456/
    ├── metrics.json           ← upload this to Claude
    └── weights.pt
```