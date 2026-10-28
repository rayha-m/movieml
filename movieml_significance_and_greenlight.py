"""
MovieML: statistical significance, error analysis, and greenlight simulation.

Input:  model_predictions_export_v2.csv (one row per held-out test movie)
        Required columns: title, budget, genre_names, is_franchise, success,
                          logreg_proba, rf_untuned_proba, xgb_proba,
                          nn_proba, rf_final_proba
Output: printed results plus CSV/JSON files in the output directory.

Usage:
    python movieml_significance_and_greenlight.py
    python movieml_significance_and_greenlight.py path/to/predictions.csv out_dir
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score

INPUT_CSV = sys.argv[1] if len(sys.argv) > 1 else "model_predictions_export_v2.csv"
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "analysis_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = {
    "Logistic Regression": "logreg_proba",
    "Random Forest (untuned)": "rf_untuned_proba",
    "XGBoost": "xgb_proba",
    "Neural Network": "nn_proba",
    "Random Forest (final)": "rf_final_proba",
}
FINAL = "Random Forest (final)"

N_BOOT = 2000
SEED = 42
GREENLIGHT_THRESHOLDS = [0.5, 0.6, 0.7]

df = pd.read_csv(INPUT_CSV)
y = df["success"].values
n = len(df)
print(f"Loaded {n} test movies from {INPUT_CSV} (base success rate {y.mean():.3f})\n")

# ---------------------------------------------------------------------------
# 1. Point-estimate ROC-AUC per model
# ---------------------------------------------------------------------------
print("=== ROC-AUC (point estimates) ===")
for name, col in MODELS.items():
    print(f"{name}: {roc_auc_score(y, df[col].values):.4f}")

# ---------------------------------------------------------------------------
# 2. Paired bootstrap: confidence intervals and pairwise win rates.
#    The SAME resampled rows are used for every model in each iteration, so
#    comparisons between models are paired.
# ---------------------------------------------------------------------------
rng = np.random.RandomState(SEED)
boot = {name: [] for name in MODELS}
for _ in range(N_BOOT):
    idx = rng.randint(0, n, n)
    y_b = y[idx]
    if len(np.unique(y_b)) < 2:
        continue
    for name, col in MODELS.items():
        boot[name].append(roc_auc_score(y_b, df[col].values[idx]))

print(f"\n=== Bootstrap 95% CIs ({N_BOOT} resamples) ===")
ci_results = {}
for name in MODELS:
    arr = np.array(boot[name])
    lo, hi = np.percentile(arr, [2.5, 97.5])
    ci_results[name] = {"mean": float(arr.mean()), "lower": float(lo), "upper": float(hi)}
    print(f"{name}: {arr.mean():.4f} [{lo:.4f}, {hi:.4f}]")

print(f"\n=== Pairwise: % of resamples where {FINAL} beats each model ===")
final_arr = np.array(boot[FINAL])
pairwise = {}
for name in MODELS:
    if name == FINAL:
        continue
    win = float((final_arr > np.array(boot[name])).mean())
    pairwise[name] = win
    print(f"{FINAL} > {name}: {win * 100:.1f}%")

with open(os.path.join(OUT_DIR, "bootstrap_results.json"), "w") as f:
    json.dump({"ci": ci_results, "pairwise_win_rate": pairwise}, f, indent=2)

# ---------------------------------------------------------------------------
# 3. Confusion matrix and error analysis for the final model (threshold 0.5)
# ---------------------------------------------------------------------------
df["pred_final"] = (df["rf_final_proba"] >= 0.5).astype(int)
df["error"] = df["pred_final"] != df["success"]
cm = confusion_matrix(y, df["pred_final"])
print("\n=== Confusion matrix (final model, threshold 0.5) ===")
print("rows = actual (0, 1), cols = predicted (0, 1)")
print(cm)
print(f"Error rate: {df['error'].mean() * 100:.1f}%")

errors = df[df["error"]].copy()
errors["confidence"] = (errors["rf_final_proba"] - 0.5).abs()
worst = errors.sort_values("confidence", ascending=False)
print("\nMost confidently wrong predictions:")
print(worst[["title", "budget", "genre_names", "is_franchise", "success", "rf_final_proba"]]
      .head(10).to_string(index=False))

print("\nError rate by franchise status:")
print(df.groupby("is_franchise")["error"].mean().round(3))

df["budget_tier"] = pd.qcut(df["budget"], q=3, labels=["Low", "Mid", "High"])
print("\nError rate by budget tier:")
print(df.groupby("budget_tier", observed=True)["error"].mean().round(3))

worst.head(25).to_csv(os.path.join(OUT_DIR, "most_confident_errors.csv"), index=False)

# ---------------------------------------------------------------------------
# 4. Greenlight simulation: only "greenlight" movies the final model rates at
#    or above a confidence threshold, then check the realized success rate.
#    Illustrative only: it uses the same held-out outcomes the model was
#    evaluated on, not a validated forecast.
# ---------------------------------------------------------------------------
base_rate = float(y.mean())
print(f"\n=== Greenlight simulation (baseline success rate {base_rate * 100:.1f}%) ===")
rows = [{"threshold": "No model (full slate)", "greenlit": n,
         "pct_of_slate": 100.0, "realized_success_rate": round(base_rate * 100, 1)}]
for t in GREENLIGHT_THRESHOLDS:
    picked = df[df["rf_final_proba"] >= t]
    if len(picked) == 0:
        continue
    rate = float(picked["success"].mean())
    rows.append({"threshold": f">= {t:.2f}", "greenlit": len(picked),
                 "pct_of_slate": round(len(picked) / n * 100, 1),
                 "realized_success_rate": round(rate * 100, 1)})
    print(f"Threshold {t:.2f}: greenlights {len(picked)}/{n} "
          f"({len(picked) / n * 100:.1f}% of slate), "
          f"realized success rate {rate * 100:.1f}%")

pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "greenlight_simulation.csv"), index=False)
df.to_csv(os.path.join(OUT_DIR, "predictions_with_errors.csv"), index=False)
print(f"\nSaved outputs to ./{OUT_DIR}/")
