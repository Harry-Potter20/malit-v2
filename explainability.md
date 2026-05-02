# 🧠 explainability.md — MALIT V2 Clinical Explainability Framework

## 🎯 Objective

Provide **clinically meaningful, decision-aligned interpretability** for MALIT V2 that goes beyond visualization.

The system must:
- Quantify confidence (not just probability)
- Explain *why* a prediction was made
- Express uncertainty in clinically understandable terms
- Support safe decision-making (WHO-aligned)

---

# ⚠️ Core Principle

> Visualizations (GradCAM, attention maps) are **supporting evidence**, not explanations.

Primary explanation must be:
- Quantitative
- Calibrated
- Decision-aware

---

# 🧬 1. CLINICAL CONFIDENCE FRAMEWORK

## 1.1 Clinical Confidence Score (CCS)

### Definition

CCS = P_calibrated × R_model

Where:

- P_calibrated → probability after calibration
- R_model → reliability factor derived from internal model behavior

---

## 1.2 Probability Calibration

### Required Method

- Temperature Scaling (default)
- Optional: Isotonic Regression

### Constraint

- Must be fitted on validation set only
- Stored in:
  results/reproducibility/calibration.json

---

## 1.3 Reliability Factor

### Definition

R_model = 1 − H_norm

Where:

- H_norm = normalized attention entropy

---

### Attention Entropy

H = −Σ p_c log₂(p_c)

Where:

p_c = softmax(channel_attention)

---

### Normalization

H_norm = H / log₂(C)

Where:
- C = number of channels (e.g., 1280)

---

### Interpretation

| H_norm | Meaning |
|--------|--------|
| Low    | Focused decision (high confidence) |
| High   | Diffuse attention (low confidence) |

---

# 🔬 2. FEATURE-LEVEL EXPLANATION SIGNALS

## 2.1 Channel Energy (LCCI Analysis)

E_c = (1 / HW) Σ x²

Used to measure:

- Channel dominance
- Suppression vs amplification

---

### Derived Metric

Dominance Ratio:

DR = top_15% energy / total energy

---

### Interpretation

| DR | Meaning |
|----|--------|
| High | Strong feature selection (reliable) |
| Low | Weak signal separation |

---

## 2.2 Attention Entropy (Uncertainty Signal)

Already defined above.

---

## 2.3 Seed Agreement Score

Agreement = (# seeds predicting same class) / total seeds

---

### Interpretation

| Agreement | Meaning |
|----------|--------|
| 1.0 | Fully stable |
| 0.66 | Moderate uncertainty |
| <0.5 | Unstable prediction |

---

# 🧪 3. UNCERTAINTY CLASSIFICATION

Agent must classify uncertainty into:

## Types

### 1. Structural Uncertainty
- High entropy
- Low dominance ratio

Meaning:
→ Weak or diffuse features

---

### 2. Epistemic Uncertainty
- Low seed agreement

Meaning:
→ Model instability

---

### 3. Data Ambiguity
- Moderate entropy
- High agreement

Meaning:
→ Difficult biological case

---

# 🏥 4. CLINICAL DECISION LAYER

## WHO Constraint

≥99% sensitivity must be maintained

---

## Threshold Logic

If:

prob ≥ sensitivity_threshold

→ Classification: **Screening-safe**

Else:

→ **Requires human verification**

---

## Output Categories

| Category | Meaning |
|--------|--------|
| High Confidence | Safe for automated decision |
| Moderate | Review recommended |
| Low | Manual inspection required |

---

# 📊 5. FINAL EXPLANATION STRUCTURE

## Required Output Format

Each prediction MUST produce:

---

### 🧾 Clinical Report

Cell ID: <id>

Prediction: PARASITIZED / UNINFECTED

Clinical Confidence: XX%

---

### Breakdown

- Calibrated Probability: XX%
- Reliability Factor: XX%
- Seed Agreement: X/X

---

### Model Reasoning

- Feature focus: High / Medium / Low
- Channel dominance: Strong / Weak
- Attention entropy: Value + interpretation

---

### Uncertainty Type

- Structural / Epistemic / Data ambiguity

---

### Clinical Safety

- Meets WHO sensitivity threshold: YES / NO

---

### Recommendation

- Accept
- Review
- Reject

---

# 🔁 6. PIPELINE INTEGRATION

## Required Module

src/explainability/clinical_confidence.py

---

## Required Functions

### 1. Calibration

```python
def calibrate_probabilities(logits, method="temperature"):
2. Entropy
def compute_entropy(attn_weights):
3. Reliability
def compute_reliability(entropy, num_channels):
4. Seed Agreement
def compute_seed_agreement(predictions):
5. CCS
def compute_ccs(prob, reliability):
6. Uncertainty Type
def classify_uncertainty(entropy, agreement, dominance):
7. Report Generator
def generate_clinical_report(sample):
💾 7. STORAGE REQUIREMENTS

All outputs must be saved:

results/explainability/

Files
clinical_reports.json
entropy_values.csv
reliability_scores.csv
seed_agreement.csv
📈 8. VALIDATION REQUIREMENTS

Agent must verify:

Lower CCS correlates with higher error rate
High entropy → higher misclassification probability
Calibration curve is near diagonal
Required Plots
Reliability vs Accuracy
Entropy vs Error Rate
CCS vs F1 Score
🔐 9. REPRODUCIBILITY
Calibration parameters must be saved
Entropy computations must be deterministic
Seed agreement must use fixed splits
🚨 10. FAILURE CONDITIONS

Agent must flag:

High confidence + wrong prediction
Low agreement + high confidence
Entropy near maximum
🧠 FINAL PRINCIPLE

The system must answer:

“How confident is the model, and should a clinician trust this prediction?”

NOT:

“Where did the model look?”

🧪 EXTENSION (OPTIONAL)

Future upgrades:

Bayesian uncertainty (MC Dropout)
Ensemble confidence intervals
Case-based reasoning (similar past cells)
✅ SUCCESS CRITERIA

Explainability is successful if:

Clinicians can interpret output without ML knowledge
Confidence correlates with correctness
Uncertainty is actionable