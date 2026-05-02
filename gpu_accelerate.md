# ☁️ kaggle_execution.md — Remote GPU Execution via Kaggle API

---

# 🎯 OBJECTIVE

Enable the MALIT V2 repository to run **entirely on Kaggle GPU infrastructure** using the Kaggle API.

The system must:

- Push code (not data) to Kaggle
- Execute full pipeline remotely
- Automatically mount dataset
- Save results to Kaggle workspace
- Pull only results back to local machine
- Maintain strict reproducibility and security

---

# ⚠️ CORE PRINCIPLE

> Code moves. Data stays. Results return.

---

# 🏗️ 1. KAGGLE EXECUTION DIRECTORY

Agent MUST create a dedicated execution folder:

kaggle/
├── run.py
├── kernel-metadata.json
└── requirements.txt (optional fallback)

---

# ⚙️ 2. KERNEL CONFIGURATION

## kernel-metadata.json

```json
{
  "id": "<your-username>/malit-v2",
  "title": "MALIT V2 Full Pipeline",
  "code_file": "run.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": false,
  "dataset_sources": [
    "iarunava/cell-images-for-detecting-malaria"
  ]
}
🚀 3. ENTRYPOINT SCRIPT
run.py (MANDATORY)

This is the ONLY script Kaggle executes.

Responsibilities
Detect Kaggle runtime
Set dataset path
Call full pipeline
Implementation
import os
import subprocess

# Kaggle dataset mount path
DATA_PATH = "/kaggle/input/cell-images-for-detecting-malaria"

# Set env variable for entire repo
os.environ["DATASET_PATH"] = DATA_PATH

# Ensure outputs go to Kaggle working dir
os.environ["OUTPUT_DIR"] = "/kaggle/working/results"

# Execute full pipeline
subprocess.run(
    ["python", "scripts/run_full_pipeline.py"],
    check=True
)
📂 4. DATA ACCESS STANDARD
RULE

Agent MUST enforce:

import os
DATA_PATH = os.getenv("DATASET_PATH")
❌ FORBIDDEN
Hardcoded local paths
Relative paths tied to developer machine
💾 5. OUTPUT MANAGEMENT
Kaggle Constraint

All outputs MUST be written to:

/kaggle/working/

Required Structure

/kaggle/working/results/
├── models/
├── uncertainty/
├── cbr/
├── explainability/
├── plots/
├── statistics/
└── reproducibility/

🔄 6. EXECUTION FLOW
Push to Kaggle
kaggle kernels push -p kaggle/
Monitor Run
Kaggle UI OR CLI
Pull Results
kaggle kernels output <username>/malit-v2 -p ./results
🧠 7. FULL PIPELINE CONTRACT

scripts/run_full_pipeline.py MUST:

Load dataset from DATASET_PATH
Perform deduplication (pHash)
Create fixed splits
Train multi-seed models
Run uncertainty modules:
MC Dropout
Ensemble CI
Build CBR index
Run statistical tests (McNemar, TOST)
Generate explainability outputs
Save ALL results
⚡ 8. PERFORMANCE OPTIMIZATION

Agent SHOULD enforce:

torch.cuda.amp enabled
DataLoader:
num_workers=4
pin_memory=True
Batch size auto-scaled to GPU memory
📦 9. DEPENDENCY HANDLING
Preferred

Use Kaggle preinstalled packages.

Optional

requirements.txt only if necessary:

torch
timm
scikit-learn
faiss-cpu
imagehash
🔐 10. SECURITY RULES

Agent MUST:

Never include .env in kaggle/ folder
Never push API keys
Use .gitignore to exclude secrets
🔁 11. REPRODUCIBILITY

Agent MUST ensure:

Fixed seeds: [42, 123, 456]
Deterministic data splits
Versioned kernel metadata
Saved configs in:
results/reproducibility/
🚨 12. FAILURE CONDITIONS

Agent must detect and halt if:

Dataset path not found
GPU not enabled
Output directory not writable
Results not saved to /kaggle/working
Kernel crashes due to memory overflow
🧪 13. VALIDATION CHECKS

After execution:

Models saved
Metrics generated
Uncertainty outputs exist
CBR index created
Reproducibility artifacts present
📊 14. MINIMAL TEST RUN

Agent SHOULD support quick test mode:

1 seed
10% dataset
1 epoch

For debugging Kaggle execution.

🧠 FINAL PRINCIPLE

This system must allow:

Running the full MALIT pipeline on GPU without downloading the dataset locally.

✅ SUCCESS CRITERIA
Full pipeline executes on Kaggle GPU
Dataset never leaves Kaggle
Only results downloaded locally
Outputs are reproducible and complete