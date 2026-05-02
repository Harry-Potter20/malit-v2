# 🧠 final_polish.md — Final Statistical & Reporting Refinements

---

# 🎯 OBJECTIVE

Apply final refinements to:

- Ensure statistical correctness in reporting
- Improve robustness of validation checks
- Enhance clarity of outputs for reviewers
- Add one high-impact visualization

---

# ⚠️ 1. MCNEMAR AGGREGATION POLICY

## Problem

Mean p-values are not statistically rigorous for hypothesis testing.

---

## REQUIRED FIX

Agent MUST:

- Use **majority vote across seeds** for final significance decision
- Treat mean p-value as **descriptive only**

---

## Implementation Rule

```python
verdict = "Significant" if majority_vote(p_values) else "Not Significant"
mean_p = np.mean(p_values)  # descriptive only
REPORTING

Agent MUST state:

“Significance determined via majority vote across seeds; mean p-values reported for reference only.”

⚠️ 2. CONFIDENCE MONOTONICITY (ROBUST VERSION)
Problem

Strict monotonicity is too brittle for real data.

REQUIRED FIX

Allow small violations with tolerance.

Implementation
def check_monotonicity(error_rates, tolerance=0.02):
    violations = 0
    for i in range(1, len(error_rates)):
        if error_rates[i] > error_rates[i-1] + tolerance:
            violations += 1
    return violations <= 1  # allow small noise
RULE
Do NOT fail pipeline on minor violations
Log as warning instead
⚠️ 3. NON-REGRESSION CHECK (LOGGING)
Problem

Current assertion is binary and silent.

REQUIRED FIX

Convert into:

Soft constraint
Logged warning
Implementation
if malit_f1 < best_baseline_f1:
    if malit_f1 >= best_baseline_f1 - tolerance:
        log.warning("MALIT slightly below baseline within tolerance")
    else:
        raise ValueError("MALIT significantly underperforms baseline")
⚠️ 4. TOST INDETERMINATE REPORTING
Problem

Indeterminate results may be hidden or misinterpreted.

REQUIRED FIX

Agent MUST explicitly surface:

indeterminate=True cases
Reporting Format
Comparison	TOST Result
MALIT vs CBAM	Indeterminate (low variance)
RULE
NEVER convert indeterminate → equivalent
Always explain reason
📊 5. CONFIDENCE HISTOGRAM (NEW PLOT)
Objective

Show separation between correct and incorrect predictions.

REQUIRED PLOT
X-axis: confidence
Y-axis: frequency
Two distributions:
Correct predictions
Incorrect predictions
Implementation
def plot_confidence_histogram(confidence, correct_mask):
    correct = confidence[correct_mask]
    incorrect = confidence[~correct_mask]

    plt.hist(correct, bins=20, alpha=0.6, label="Correct")
    plt.hist(incorrect, bins=20, alpha=0.6, label="Incorrect")
    plt.legend()
OUTPUT

results/plots/confidence_histogram.png

📊 6. REPORTING ENHANCEMENTS

Agent MUST include:

Statistical Section
Majority vote decision
Mean p-value (annotated as descriptive)
TOST Section
Explicit indeterminate cases
Explanation for low variance
Confidence Analysis
Monotonicity result (pass/warn)
Histogram visualization
💾 7. STORAGE UPDATES

results/statistics/

mcnemar_summary.json (with majority vote)
tost_results.json (with indeterminate flag)

results/plots/

confidence_histogram.png
🚨 8. FAILURE CONDITIONS

Agent MUST flag:

Majority vote missing
Indeterminate TOST hidden
Histogram not generated
Monotonicity check too strict (no tolerance)
🧠 FINAL PRINCIPLE

The system must ensure:

Statistical conclusions are not only correct, but clearly communicated and robust to real-world noise.

✅ SUCCESS CRITERIA
Majority-vote McNemar used for decisions
Confidence checks robust to noise
TOST indeterminate cases visible
Confidence histogram generated
Reporting is reviewer-proof