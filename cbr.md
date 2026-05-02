# 🧠 bayesian_uncertainty.md — MC Dropout Uncertainty Module

## 🎯 Objective

Implement **approximate Bayesian uncertainty estimation** using MC Dropout for MALIT.

This provides:
- Predictive mean
- Predictive variance
- Confidence intervals

---

# ⚙️ 1. CORE PRINCIPLE

Enable dropout at inference time and perform multiple stochastic forward passes.

Approximation:

E[p(y|x)] ≈ (1/T) Σ p(y|x, w_t)

Var[p(y|x)] ≈ (1/T) Σ (p_t − mean)²

---

# 🧪 2. MODEL MODIFICATIONS

## Requirement

- Dropout layers MUST remain active during inference

---

## Implementation

```python
def enable_mc_dropout(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()
🔁 3. SAMPLING
Parameters
T = 20–50 forward passes
Function
def mc_dropout_predict(model, x, T=30):
    preds = []
    for _ in range(T):
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        preds.append(probs)

    preds = torch.stack(preds)

    mean = preds.mean(dim=0)
    var = preds.var(dim=0)

    return mean, var
📊 4. OUTPUTS

For each sample:

mean probability
variance
entropy (optional)
🧾 5. CLINICAL INTERPRETATION
Variance	Meaning
Low	High confidence
High	Model uncertainty
💾 6. STORAGE

results/uncertainty/mc_dropout/

mean_probs.csv
variance.csv
predictive_entropy.csv
📈 7. VALIDATION

Agent must verify:

Higher variance correlates with misclassification
Variance increases near decision boundary
🔐 8. REPRODUCIBILITY
Fix seed before sampling
Store T value in config
🚨 9. FAILURE CONDITIONS
Variance near zero for all samples → dropout not active
Identical outputs across passes → bug

---

# 🧠 `ensemble_confidence.md`

```markdown id="88421"
# 🧠 ensemble_confidence.md — Multi-Seed Ensemble Confidence

## 🎯 Objective

Leverage multi-seed models to compute:
- Mean prediction
- Confidence intervals
- Agreement score

---

# ⚙️ 1. INPUT

From MultiSeedRunner:

- all_seed_preds
- all_seed_probs

---

# 📊 2. COMPUTATION

## Mean

mean_p = mean(probabilities across seeds)

---

## Variance

var_p = variance across seeds

---

## Confidence Interval (95%)

CI = mean ± 1.96 * std / sqrt(N)

---

# 🧪 3. IMPLEMENTATION

```python
def ensemble_stats(probs):
    mean = probs.mean(axis=0)
    std = probs.std(axis=0)

    ci_low = mean - 1.96 * std / np.sqrt(len(probs))
    ci_high = mean + 1.96 * std / np.sqrt(len(probs))

    return mean, std, ci_low, ci_high
🧠 4. AGREEMENT SCORE

agreement = mode(predictions) frequency / N

📊 5. INTERPRETATION
CI Width	Meaning
Narrow	Stable prediction
Wide	Uncertain
🧾 6. OUTPUT

For each sample:

mean probability
CI lower/upper
agreement score
💾 7. STORAGE

results/uncertainty/ensemble/

mean_probs.csv
ci_bounds.csv
agreement.csv
📈 8. VALIDATION

Agent must verify:

Wider CI correlates with errors
Low agreement correlates with misclassification
🔐 9. REPRODUCIBILITY
Same data split across seeds
Fixed seed list [42, 123, 456]
🚨 10. FAILURE CONDITIONS
CI width = 0 → seeds not diverse
Agreement always 1 → bug or leakage

---

# 🧠 `case_based_reasoning.md`

```markdown id="55913"
# 🧠 case_based_reasoning.md — Similar Case Retrieval System

## 🎯 Objective

Provide **human-aligned interpretability** via similar past cases.

---

# 🧠 1. CORE IDEA

Retrieve nearest neighbors in feature space.

---

# ⚙️ 2. FEATURE EXTRACTION

Use:

- 512-d vector AFTER global average pooling

---

## Function

```python
def extract_embedding(model, x):
    features = model.forward_features(x)
    pooled = torch.mean(features, dim=(2,3))
    return pooled
🗄️ 3. INDEX BUILDING
Store:
embeddings
labels
image paths
Use
FAISS (preferred)
or sklearn NearestNeighbors
🔍 4. RETRIEVAL
def retrieve_neighbors(query, index, k=5):
    distances, indices = index.search(query, k)
    return indices, distances
📊 5. OUTPUT

For each sample:

top-k similar images
their labels
similarity scores
🧾 6. CLINICAL INTERPRETATION
Scenario	Meaning
Similar infected cases	Supports prediction
Mixed neighbors	Ambiguous
Similar uninfected	Possible error
🧠 7. CONSISTENCY SCORE

consistency = (# neighbors with same label) / k

📊 8. OUTPUT FORMAT
neighbor images
similarity scores
consistency score
💾 9. STORAGE

results/cbr/

embeddings.npy
index.faiss
neighbors.json
📈 10. VALIDATION

Agent must verify:

High consistency → higher accuracy
Low consistency → higher error rate
🔐 11. REPRODUCIBILITY
Fixed embedding extraction pipeline
Store index version
🚨 12. FAILURE CONDITIONS
Random neighbors → embedding broken
All distances equal → normalization issue

---

# 🧠 Final Integration Layer (IMPORTANT)

You now have **3 independent uncertainty systems**.

Your agent MUST combine them:

---

## 🧩 Unified Confidence Schema

```python
final_confidence = combine(
    CCS,
    mc_variance,
    ensemble_ci_width,
    cbr_consistency
)
🧾 Final Output Should Include
Clinical Confidence (CCS)
Bayesian Uncertainty (variance)
Ensemble Stability (CI + agreement)
Case Similarity (CBR)