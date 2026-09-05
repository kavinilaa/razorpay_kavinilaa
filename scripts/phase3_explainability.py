import json
import os
import time

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TXN_PARQUET = os.path.join(ROOT, "data", "processed", "engineered_transactions.parquet")
MODELS_DIR = os.path.join(ROOT, "models")
REPORTS_DIR = os.path.join(ROOT, "reports")
RANDOM_SEED = 42

with open(os.path.join(MODELS_DIR, "model_metadata.json")) as f:
    meta = json.load(f)

best_name = meta["selected_model"]
feature_list = meta["feature_list"]
train_steps = meta["train_steps"]
val_steps = meta["val_steps"]
test_steps = meta["test_steps"]

print(f"Selected model: {best_name}")
model = joblib.load(os.path.join(MODELS_DIR, f"{best_name}.joblib"))

cols = feature_list + ["isFraud", "step", "eligible_for_fraud_model"]
df = pd.read_parquet(TXN_PARQUET, columns=cols)
df = df.loc[df["eligible_for_fraud_model"] == 1]

val = df.loc[(df["step"] >= val_steps[0]) & (df["step"] <= val_steps[1])]
test = df.loc[(df["step"] >= test_steps[0]) & (df["step"] <= test_steps[1])]

X_val, y_val = val[feature_list], val["isFraud"]
X_test, y_test = test[feature_list], test["isFraud"]

results = {"model": best_name}

# --- 1. Built-in gain-based feature importance ---
print("Computing built-in feature importance...")
if best_name == "xgboost":
    booster = model.get_booster()
    gain_dict = booster.get_score(importance_type="gain")
    importances = {f: gain_dict.get(f, 0.0) for f in feature_list}
elif best_name == "lightgbm":
    importances = dict(zip(feature_list, model.feature_importances_.tolist()))
else:
    coefs = model.named_steps["clf"].coef_[0]
    importances = dict(zip(feature_list, np.abs(coefs).tolist()))

sorted_importance = dict(sorted(importances.items(), key=lambda x: -x[1]))
results["builtin_feature_importance"] = sorted_importance

# --- 2. Permutation importance (on a subsample of validation set for speed) ---
print("Computing permutation importance (validation subsample)...")
rng = np.random.RandomState(RANDOM_SEED)
sample_size = min(20000, len(X_val))
sample_idx = rng.choice(len(X_val), size=sample_size, replace=False)
X_val_sample = X_val.iloc[sample_idx]
y_val_sample = y_val.iloc[sample_idx]

t0 = time.time()
if best_name == "xgboost" or best_name == "lightgbm":
    predict_obj = model
else:
    predict_obj = model
perm = permutation_importance(
    predict_obj, X_val_sample, y_val_sample, n_repeats=5, random_state=RANDOM_SEED,
    scoring="average_precision", n_jobs=-1,
)
perm_importance = dict(sorted(
    zip(feature_list, perm.importances_mean.tolist()), key=lambda x: -x[1]
))
results["permutation_importance_pr_auc_drop"] = perm_importance
results["permutation_importance_sample_size"] = sample_size
print(f"Permutation importance computed in {time.time()-t0:.1f}s")

# --- 3. SHAP values ---
print("Computing SHAP values (TreeExplainer on a sample)...")
shap_sample_size = min(3000, len(X_test))
shap_sample = X_test.sample(n=shap_sample_size, random_state=RANDOM_SEED)

if best_name in ("xgboost", "lightgbm"):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(shap_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
else:
    background = X_val.sample(n=min(200, len(X_val)), random_state=RANDOM_SEED)
    explainer = shap.LinearExplainer(model.named_steps["clf"], model.named_steps["scaler"].transform(background))
    shap_values = explainer.shap_values(model.named_steps["scaler"].transform(shap_sample))

mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_importance = dict(sorted(zip(feature_list, mean_abs_shap.tolist()), key=lambda x: -x[1]))
results["shap_mean_abs_importance"] = shap_importance
results["shap_sample_size"] = shap_sample_size

# --- 4. Explain one genuine high-risk TEST transaction (true positive, high predicted prob) ---
print("Finding a genuine high-confidence true-positive fraud case in TEST set for explainability example...")
test_probs = model.predict_proba(X_test)[:, 1]
test_df = test.copy()
test_df["pred_prob"] = test_probs
tp_candidates = test_df.loc[(test_df["isFraud"] == 1) & (test_df["pred_prob"] >= 0.9)]
if len(tp_candidates) > 0:
    example_row = tp_candidates.iloc[0]
    example_idx_pos = X_test.index.get_loc(example_row.name)
    example_features = X_test.iloc[[example_idx_pos]]

    if best_name in ("xgboost", "lightgbm"):
        example_shap = explainer.shap_values(example_features)
        if isinstance(example_shap, list):
            example_shap = example_shap[1]
        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[1] if len(np.atleast_1d(base_value)) > 1 else float(np.atleast_1d(base_value)[0])
    else:
        example_shap = explainer.shap_values(model.named_steps["scaler"].transform(example_features))
        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = float(np.atleast_1d(base_value)[0])

    shap_row = np.asarray(example_shap).flatten()
    contrib = sorted(zip(feature_list, shap_row.tolist(), example_features.iloc[0].tolist()),
                      key=lambda x: -abs(x[1]))

    results["example_high_risk_transaction"] = {
        "step": int(example_row["step"]),
        "true_label": int(example_row["isFraud"]),
        "predicted_probability": round(float(example_row["pred_prob"]), 6),
        "shap_base_value": round(float(base_value), 6),
        "top_contributing_features": [
            {"feature": f, "shap_value": round(float(v), 6), "feature_value": float(fv)}
            for f, v, fv in contrib[:8]
        ],
    }
else:
    results["example_high_risk_transaction"] = "No true-positive fraud case found with predicted_probability >= 0.9 in test set."

with open(os.path.join(REPORTS_DIR, "phase3_explainability_stats.json"), "w") as f:
    json.dump(results, f, indent=2, default=str)

print("Wrote reports/phase3_explainability_stats.json")
print(json.dumps(results.get("example_high_risk_transaction", {}), indent=2, default=str))
