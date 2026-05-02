# 🧠 baselines.md — Baseline Model Suite

---

# 🎯 OBJECTIVE

Define a **minimal, representative, and fair baseline set** to evaluate MALIT V2 against existing approaches.

Baselines must:
- Cover different architectural families
- Use identical training conditions
- Be reproducible across seeds

---

# 🧠 1. BASELINE CATEGORIES

Agent MUST include:

---

## 1.1 Backbone Baseline

Purpose:
Test whether MALIT improves over its own backbone

Model:
- EfficientNet-B0 (no Gabor, no LCCI, no attention)

---

## 1.2 Classical CNN

Purpose:
Compare against widely used architectures

Models:
- ResNet50
- MobileNetV3

---

## 1.3 Attention Baselines

Purpose:
Test whether MALIT attention is superior

Models:
- EfficientNet + SE block
- EfficientNet + CBAM

---

# ⚙️ 2. IMPLEMENTATION RULES

Agent MUST:

- Use SAME input resolution (224×224)
- Use SAME preprocessing
- Use SAME dataset split
- Use SAME augmentation

---

# 🔒 3. TRAINING PARITY (CRITICAL)

All baselines MUST use:

- Optimizer: AdamW
- LR: same as MALIT (or tuned but documented)
- Scheduler: Cosine
- AMP: enabled
- Early stopping: identical

---

# 🧠 4. FREEZING POLICY

All baselines must be:

- Fully fine-tuned (no frozen layers)

---

# 🌱 5. SEED STRATEGY

Baselines:

- Tier 2 seeds: [42, 123]

MALIT:

- Tier 1 seeds: [42, 123, 456]

---

# 📊 6. OUTPUT REQUIREMENTS

For each baseline:

- Predictions
- Probabilities
- Metrics (F1, Accuracy, AUC)
- Per-seed outputs

---

# 💾 7. STORAGE

results/baselines/

- resnet50/
- mobilenetv3/
- efficientnet/
- cbam/
- senet/

---

# 🚨 8. FAILURE CONDITIONS

Agent must flag:

- Different data splits
- Different augmentations
- Missing seeds
- Missing outputs

---

# 🧠 FINAL PRINCIPLE

Baselines must isolate:

> “Is MALIT better because of its components, or just because of training differences?”

---

# ✅ SUCCESS CRITERIA

- All baselines trained under identical conditions
- Results comparable across models
- Outputs ready for statistical testing
🧠 comparison_protocol.md
# 🧠 comparison_protocol.md — Statistical Comparison Framework

---

# 🎯 OBJECTIVE

Ensure **fair, statistically valid comparison** between MALIT and baselines.

---

# ⚖️ 1. FAIRNESS CONSTRAINTS

Agent MUST guarantee:

- Same dataset split
- Same preprocessing
- Same evaluation set
- No data leakage

---

# 📊 2. PRIMARY METRICS

Agent MUST compute:

- F1 Score (primary)
- Sensitivity (critical for medical)
- Specificity
- AUC
- ECE (calibration)

---

# 🧪 3. STATISTICAL TESTS

---

## 3.1 McNemar Test

Compare:

MALIT vs each baseline

---

## 3.2 TOST Equivalence

Test:

Is MALIT significantly better OR equivalent?

Margin:

±0.5% F1

---

## 3.3 Effect Size

Agent MUST compute:

- ΔF1
- ΔSensitivity

---

# 🌱 4. MULTI-SEED AGGREGATION

Agent MUST:

- Aggregate per-seed results
- Store all predictions

---

# 📊 5. CONFIDENCE INTERVALS

Compute:

- Mean ± 95% CI for each metric

---

# 🧠 6. SIGNIFICANCE RULES

---

## Improvement

If:

- p < 0.05 (McNemar)
- AND ΔF1 > 0

→ MALIT superior

---

## Equivalent

If:

- TOST passes

→ models equivalent

---

## Inconclusive

Otherwise

---

# 📈 7. ADDITIONAL ANALYSIS

Agent SHOULD compute:

- Error overlap between models
- Cases where MALIT corrects baseline errors

---

# 💾 8. STORAGE

results/statistics/

- mcnemar_results.json
- tost_results.json
- effect_sizes.json

---

# 🚨 9. FAILURE CONDITIONS

Agent must flag:

- Missing predictions
- Mismatched sample counts
- NaN metrics

---

# 🧠 FINAL PRINCIPLE

Comparison must answer:

> “Is MALIT meaningfully better, not just numerically higher?”

---

# ✅ SUCCESS CRITERIA

- All models compared statistically
- Results reproducible across seeds
- Clear significance conclusions
🧠 reporting.md
# 🧠 reporting.md — Paper Reporting & Tables

---

# 🎯 OBJECTIVE

Present results in a **clear, reviewer-proof format**.

---

# 📊 1. MAIN PERFORMANCE TABLE

Agent MUST generate:

| Model | F1 | Sensitivity | Specificity | AUC | ECE |
|------|----|-------------|-------------|-----|-----|

Include:

- Mean ± CI
- Best value highlighted

---

# 📊 2. STATISTICAL TABLE

| Comparison | ΔF1 | McNemar p | TOST | Conclusion |
|-----------|-----|-----------|------|-----------|

---

# 📊 3. ABLATION TABLE

| Model Variant | F1 | ΔF1 |
|--------------|----|-----|
| Full MALIT | X | — |
| - Gabor | X | Δ |
| - LCCI | X | Δ |
| - Attention | X | Δ |

---

# 📈 4. REQUIRED FIGURES

Agent MUST generate:

---

## 4.1 ROC Curves

All models on same plot

---

## 4.2 Confidence vs Error

- CCS vs error rate
- MC variance vs error

---

## 4.3 Calibration Curve

- Reliability diagram

---

## 4.4 CI Width vs Error

- Ensemble uncertainty validation

---

# 🧠 5. INTERPRETABILITY TABLE

| Metric | Infected | Uninfected |
|-------|----------|-----------|
| Entropy | X | X |
| Channel Energy | X | X |

---

# 🧾 6. TEXT TEMPLATE

Agent MUST generate:

---

## Comparison Statement

“MALIT outperformed all baseline models, achieving an F1 score of X compared to Y for the best baseline (p < 0.05, McNemar test).”

---

## Statistical Statement

“Differences were statistically significant and not attributable to random initialization.”

---

# 🚨 7. FAILURE CONDITIONS

Agent must flag:

- Missing CI
- Missing statistical tests
- Inconsistent tables

---

# 🧠 FINAL PRINCIPLE

Results must be:

- Clear
- Statistically justified
- Easy to verify

---

# ✅ SUCCESS CRITERIA

- Paper-ready tables generated
- All claims supported by statistics
- Visualizations match numerical results