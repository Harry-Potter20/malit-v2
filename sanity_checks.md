# 🧠 post_validation_fixes.md — Final Corrections & Validation Hardening

---

# 🎯 OBJECTIVE

Resolve all warnings and strengthen the system to:

- Eliminate silent numerical issues
- Ensure statistical correctness
- Validate uncertainty and confidence metrics
- Prepare system for publication-grade reproducibility

---

# ⚠️ 1. DEVICE-SPECIFIC WARNINGS (MPS / CUDA)

## Problem

MPS (Apple Silicon) does not support `pin_memory=True`, causing warnings.

---

## REQUIRED FIX

Agent MUST conditionally enable pin_memory:

```python
def get_dataloader_config(device):
    return {
        "pin_memory": device.type == "cuda"
    }
VALIDATION
No warnings during DataLoader initialization
Works across:
CPU
CUDA
MPS
⚠️ 2. NUMERICAL STABILITY (DIVISION BY ZERO)
Problem

Single-seed runs cause:

std = 0
division by zero in CI / statistics
REQUIRED FIX
Safe Standard Deviation
def safe_std(arr):
    if len(arr) < 2:
        return 0.0
    return np.std(arr, ddof=1)
Safe Confidence Interval
def safe_ci(mean, std, n):
    if n < 2 or std == 0:
        return mean, mean
    margin = 1.96 * std / np.sqrt(n)
    return mean - margin, mean + margin
VALIDATION
No runtime warnings
CI collapses correctly for single-seed runs
🧠 3. STATISTICAL CORRECTNESS HARDENING

Agent MUST ensure:

3.1 McNemar Test Safety
Only run if disagreement exists
Handle edge case:
if b + c == 0:
    return "No disagreement; test not applicable"
3.2 TOST Equivalence
Skip if variance = 0
Log condition explicitly
3.3 ECE Stability
Avoid empty bins
Add epsilon:
eps = 1e-8
🧪 4. NEW VALIDATION TEST SUITES (MANDATORY)
4.1 Confidence Calibration Test
Goal

Verify that confidence correlates with correctness

Test
assert error_rate(low_confidence) > error_rate(high_confidence)
4.2 Uncertainty Monotonicity
Tests
assert error_rate(high_entropy) > error_rate(low_entropy)
assert error_rate(high_variance) > error_rate(low_variance)
4.3 Ensemble Stability Test
assert error_rate(wide_ci) > error_rate(narrow_ci)
4.4 CBR Consistency Test
assert error_rate(low_consistency) > error_rate(high_consistency)
📊 5. LOGGING & WARNING CONTROL

Agent MUST:

Convert warnings → structured logs
Suppress known-safe warnings explicitly
Example
import warnings

warnings.filterwarnings(
    "ignore",
    message=".*pin_memory.*MPS.*"
)
RULE
Do NOT globally suppress all warnings
Only suppress known, documented ones
💾 6. RESULT VALIDATION CHECKS

After every run, agent MUST verify:

No NaNs in metrics
No infinite values
All CI bounds valid (low ≤ high)
Variance ≥ 0
🔁 7. REPRODUCIBILITY HARDENING

Agent MUST:

Log all seeds used
Save:
statistical outputs
CI values
calibration parameters
📁 Required Files

results/reproducibility/

seed_manifest.json
stats_validation.json
calibration.json
🧠 8. FAILURE DETECTION RULES

Agent MUST flag:

Critical
High confidence + incorrect prediction
Zero variance across ensemble
CI width = 0 for multi-seed runs
Warning
High entropy + high confidence
Low agreement + high CCS
📈 9. OPTIONAL (RECOMMENDED) METRICS

Add:

Brier Score
Negative Log Likelihood (NLL)
🧠 10. FINAL VALIDATION PRINCIPLE

The system must guarantee:

Confidence, uncertainty, and stability metrics are not only computed, but meaningfully correlated with model error.

✅ SUCCESS CRITERIA
Zero runtime warnings
Stable statistics under all seed conditions
Confidence and uncertainty validated against error
Fully reproducible outputs