# 🧠 MALIT V2 — Unified Research Orchestration & Clinical Explainability Agent

---

# 🎯 OBJECTIVE

Build a **clinically interpretable, uncertainty-aware, fully reproducible AI system** for malaria detection that:

- Matches MALIT V2 architecture exactly
- Produces **quantified clinical confidence (not just probability)**
- Integrates **multi-layer uncertainty (Bayesian + ensemble + structural)**
- Supports **case-based reasoning (human-aligned interpretability)**
- Runs on Kaggle GPU
- Saves ALL artifacts locally for paper reproducibility

---

# 🏗️ 1. REPOSITORY STRUCTURE

malit-v2/
├── src/
│   ├── models/
│   ├── data/
│   ├── training/
│   ├── evaluation/
│   ├── explainability/
│   │   ├── clinical_confidence.py
│   │   ├── bayesian_uncertainty.py
│   │   ├── ensemble_confidence.py
│   │   └── case_based_reasoning.py
│   ├── statistics/
│   └── utils/
├── configs/
├── scripts/
├── results/
│   ├── models/
│   ├── uncertainty/
│   │   ├── mc_dropout/
│   │   ├── ensemble/
│   ├── cbr/
│   ├── explainability/
│   ├── plots/
│   ├── gradcam/
│   ├── attention_maps/
│   ├── statistics/
│   └── reproducibility/
├── .env
├── pyproject.toml
├── requirements.txt
├── uv.lock
└── agents.md

---

# ⚙️ 2. ENVIRONMENT SETUP

## Virtual Environment

```bash
uv venv
source .venv/bin/activate
Dependencies
torch, torchvision, timm
scikit-learn
imagehash
faiss-cpu
matplotlib, seaborn
kaggle
python-dotenv
Kaggle API

.env:

KAGGLE_USERNAME=...
KAGGLE_KEY=...

🚨 Agent MUST:

NEVER log or print API key
ONLY access via environment variables
☁️ 3. KAGGLE GPU EXECUTION
Upload code via Kaggle API
Attach malaria dataset
Enable GPU (T4/V100)
Run training remotely
Sync results locally
📦 4. DATA PIPELINE
Dataset: NIH Malaria (27,558 images)
Deduplication: pHash (Hamming ≤ 4)
Stratified split:
Train 70%
Val 15%
Test 15%
Fixed seed: 42
🧠 5. ARCHITECTURE (STRICT)

Pipeline:

Input → Gabor Filters → EfficientNet-B0 → LCCI → Dual Attention → MultiScale → MLP

Constraints:

LCCI is channel-wise only
RF = {1,3,5,7}
freeze_backbone configurable
🔁 6. TRAINING
Optimizer: AdamW
LR: 3e-4
AMP enabled
Cosine schedule
Early stopping (patience=3)
Gradient clipping: 1.0
🌱 7. MULTI-SEED STRATEGY

Tier 1:
[42, 123, 456]

Agent MUST:

Store predictions per seed
Store logits per seed
📊 8. STATISTICAL TESTING

Implement:

McNemar (per-seed + aggregate)
TOST equivalence (±0.5% F1)
🧠 9. CLINICAL EXPLAINABILITY CORE
9.1 Clinical Confidence Score (CCS)

CCS = P_calibrated × R_model

9.2 Reliability

R_model = 1 − normalized_entropy

9.3 Required Signals
Attention entropy
Channel energy
LCCI dominance
Seed agreement
🔬 10. BAYESIAN UNCERTAINTY (MC DROPOUT)
Requirements
Enable dropout during inference
Perform T=30 forward passes
Outputs
Predictive mean
Predictive variance
Storage

results/uncertainty/mc_dropout/

📊 11. ENSEMBLE CONFIDENCE

Using multi-seed models:

Mean prediction
Std deviation
95% CI
Agreement score
Storage

results/uncertainty/ensemble/

🧠 12. CASE-BASED REASONING (CBR)
Pipeline
Extract embeddings (512-d)
Build FAISS index
Retrieve top-k neighbors
Outputs
Similar images
Labels
Similarity scores
Consistency score
Storage

results/cbr/

🧩 13. UNIFIED CONFIDENCE ENGINE

Agent MUST combine:

CCS
MC variance
Ensemble CI width
Seed agreement
CBR consistency
Final Confidence Logic

High confidence if:

CCS high
Variance low
CI narrow
Agreement high
CBR consistent
🧾 14. FINAL CLINICAL REPORT

For EACH sample:

Prediction: {class}

Clinical Confidence: XX%

Breakdown:

Calibrated Probability
Reliability (entropy-based)
MC Dropout Variance
Ensemble CI
Seed Agreement
CBR Consistency

Uncertainty Type:

Structural / Epistemic / Data ambiguity

Clinical Decision:

Screening-safe (≥99% sensitivity)
Review required
Reject
🔬 15. INTERPRETABILITY OUTPUTS

Generate:

GradCAM
Attention maps
LCCI visualizations
Gabor filters
💾 16. AUTO-SAVE SYSTEM

Everything MUST be saved locally:

Models
Metrics
Uncertainty outputs
CBR results
Plots
📈 17. VALIDATION

Agent MUST verify:

Higher entropy → higher error
Higher variance → higher error
Wider CI → higher error
Lower CBR consistency → higher error
🔁 18. REPRODUCIBILITY

Save:

seed_manifest.json
hyperparameters.json
calibration.json
model_registry.json
🔐 19. SECURITY
Never expose .env
No external logging
All outputs local
⚡ 20. EXECUTION PIPELINE
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

python scripts/download_dataset.py
python scripts/run_training.py
python scripts/run_uncertainty.py
python scripts/run_cbr.py
python scripts/run_statistics.py
python scripts/generate_reports.py
🧪 21. FAILURE CONDITIONS

Agent must flag:

High confidence + wrong prediction
Low agreement + high confidence
Zero MC variance
CI width = 0
Random CBR neighbors
🧠 FINAL PRINCIPLE

The system must answer:

“How confident is this prediction, how stable is it, and has the model seen similar cases before?”

NOT:

“Where did the model look?”

✅ SUCCESS CRITERIA
Clinician can interpret output without ML knowledge
Confidence correlates with correctness
Uncertainty is actionable
Results are fully reproducible