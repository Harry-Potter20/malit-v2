# 🧠 comparison_fixes.md — Statistical & Baseline Evaluation Fixes

---

# 🎯 OBJECTIVE

Strengthen statistical validity, baseline fairness, and reporting rigor by:

- Extending statistical tests to multi-seed evaluation
- Handling edge cases in equivalence testing
- Ensuring fair baseline comparison
- Adding confidence–error validation
- Preventing subtle training and memory bugs

---

# ⚠️ 1. MULTI-SEED MCNEMAR TEST (CRITICAL)

## Problem

Current implementation uses:
- Single seed (seed=42)

This contradicts:
- Multi-seed evaluation strategy

---

## REQUIRED FIX

Agent MUST compute McNemar test:

- Per seed
- Aggregate results

---

## Implementation

```python
def mcnemar_multi_seed(preds_a, preds_b, labels):
    results = []

    for seed in preds_a.keys():
        result = mcnemar_test(
            preds_a[seed],
            preds_b[seed],
            labels[seed]
        )
        results.append(result)

    return aggregate_mcnemar(results)
Aggregation Strategy
Mean p-value OR
Majority vote (significant vs not)
VALIDATION
All seeds processed
No missing predictions
Aggregated result stored
⚠️ 2. TOST EQUIVALENCE EDGE CASE
Problem

Low variance across seeds leads to:

Artificial equivalence
Invalid statistical conclusion
REQUIRED FIX

Agent MUST detect near-zero variance:

def safe_tost(mean_diff, std, n):
    if std < 1e-6:
        return "Indeterminate (zero variance)"
    return run_tost(mean_diff, std, n)
RULE
Log condition explicitly
Do NOT report equivalence in this case
⚠️ 3. BASELINE FAIRNESS (HYPERPARAMETERS)
Problem

Different architectures may require different optimal LR.

Using identical LR:

Ensures fairness
BUT may disadvantage some baselines
REQUIRED ACTION

Agent MUST:

Option A (default)
Use identical hyperparameters
Document explicitly
Option B (optional)
Perform minimal LR sweep:
{3e-4, 1e-4}
Select best for each baseline
Log chosen values
LOGGING

results/baselines/hparams.json

📊 4. CONFIDENCE VS ERROR ANALYSIS (MANDATORY)
Objective

Validate that:

Confidence correlates with correctness

REQUIRED IMPLEMENTATION

Bin predictions by confidence:

def compute_confidence_bins(confidences, errors, bins=10):
    # returns error rate per bin
OUTPUT TABLE
Confidence Bin	Error Rate
0.9–1.0	X
0.8–0.9	X
...	...
VALIDATION

Agent MUST verify:

Monotonic trend:
higher confidence → lower error
🧪 5. NEW INTEGRATION TEST
Objective

Ensure MALIT does not regress below baselines

TEST
assert malit_f1 >= best_baseline_f1 - tolerance

Where:

tolerance = 0.01 (configurable)
FAILURE CONDITION
MALIT underperforms baseline significantly
⚙️ 6. GPU PERFORMANCE OPTIMIZATION
REQUIRED ADDITION
torch.backends.cudnn.benchmark = True
CONDITION
Only enable for fixed input size
⚠️ 7. SAFE DETACH PATTERN
Problem

Incorrect use of .detach() can:

Break gradient flow
Corrupt training
REQUIRED PATTERN
loss = criterion(logits, labels)
loss.backward()

stored_logits = logits.detach()
FORBIDDEN
logits = logits.detach()  # ❌ breaks gradients
📊 8. REPORTING UPDATES

Agent MUST include:

New Table

Confidence vs Error

Updated Statistical Reporting
Multi-seed McNemar results
TOST with edge-case handling
🔁 9. REPRODUCIBILITY

Agent MUST save:

results/statistics/

mcnemar_per_seed.json
mcnemar_aggregated.json
tost_results.json
🚨 10. FAILURE CONDITIONS

Agent MUST flag:

Missing seeds in McNemar
Zero variance in TOST
Non-monotonic confidence-error relationship
MALIT underperforming baseline
Incorrect detach usage (if detectable)
🧠 FINAL PRINCIPLE

The system must ensure:

Statistical conclusions are robust, multi-seed validated, and not artifacts of implementation shortcuts.

✅ SUCCESS CRITERIA
Multi-seed statistical tests implemented
Edge cases handled safely
Confidence validated against error
Baseline comparison is fair and defensible
No silent training bugs