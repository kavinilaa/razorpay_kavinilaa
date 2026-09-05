import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TXN_PARQUET = os.path.join(ROOT, "data", "processed", "engineered_transactions.parquet")
TIME_PARQUET = os.path.join(ROOT, "data", "processed", "time_window_features.parquet")
RAW_CSV = os.path.join(ROOT, "data", "raw", "PS_20174392719_1491204439457_log.csv")
OUT_PATH = os.path.join(ROOT, "reports", "phase2_validation_stats.json")

results = {}

print("Loading processed transaction parquet...")
df = pd.read_parquet(TXN_PARQUET)
print("Loading time-window parquet...")
twf = pd.read_parquet(TIME_PARQUET)

# raw row count for comparison (cheap: count lines)
with open(RAW_CSV, "rb") as f:
    raw_row_count = sum(1 for _ in f) - 1  # minus header

results["row_counts"] = {
    "raw_csv_data_rows": raw_row_count,
    "engineered_transactions_rows": len(df),
    "match": raw_row_count == len(df),
}

results["columns"] = list(df.columns)
results["n_columns_total"] = len(df.columns)

raw_cols = {"step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
            "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud"}
engineered_cols = [c for c in df.columns if c not in raw_cols]
results["engineered_feature_count"] = len(engineered_cols)
results["engineered_feature_names"] = engineered_cols


dupe_subset = ["step", "type", "amount", "nameOrig", "nameDest", "oldbalanceOrg", "newbalanceOrig"]
results["duplicate_rows_on_key_subset"] = int(df.duplicated(subset=dupe_subset).sum())
raw_dupe_check = pd.read_csv(RAW_CSV, dtype={"step": "int32"}, usecols=["step"])
results["step_min_raw"] = int(raw_dupe_check["step"].min())
results["step_max_raw"] = int(raw_dupe_check["step"].max())
del raw_dupe_check

# NaN check
na_counts = df.isna().sum()
results["nan_counts_nonzero"] = {c: int(v) for c, v in na_counts.items() if v > 0}

# infinite value check (numeric columns only) - per-column to avoid
# interleaving 33 mixed-dtype columns into one huge float64 array (OOM risk)
numeric_cols = df.select_dtypes(include=[np.number]).columns
inf_nonzero = {}
for c in numeric_cols:
    n_inf = int(np.isinf(df[c].to_numpy()).sum())
    if n_inf > 0:
        inf_nonzero[c] = n_inf
results["inf_counts_nonzero"] = inf_nonzero

# impossible balance checks
results["impossible_balances"] = {
    "negative_oldbalanceOrg": int((df["oldbalanceOrg"] < 0).sum()),
    "negative_newbalanceOrig": int((df["newbalanceOrig"] < 0).sum()),
    "negative_oldbalanceDest": int((df["oldbalanceDest"] < 0).sum()),
    "negative_newbalanceDest": int((df["newbalanceDest"] < 0).sum()),
    "negative_amount": int((df["amount"] < 0).sum()),
}

# categorical consistency
results["categorical_checks"] = {
    "type_categories": sorted(df["type"].astype(str).unique().tolist()),
    "hour_of_day_range": [int(df["hour_of_day"].min()), int(df["hour_of_day"].max())],
    "day_index_range": [int(df["day_index"].min()), int(df["day_index"].max())],
    "binary_flags_are_0_1": {
        c: bool(set(df[c].unique().tolist()) <= {0, 1})
        for c in ["drained_to_zero", "origin_balance_zero_before", "dest_balance_artifact_flag",
                  "is_transfer", "is_cash_out", "eligible_for_fraud_model", "destination_is_new",
                  "high_amount_indicator", "isFraud", "isFlaggedFraud"]
    },
}

# feature range sanity
range_cols = [
    "log_amount", "balance_change_orig", "balance_error_orig", "amount_to_origin_balance_ratio",
    "balance_change_dest", "balance_error_dest", "hour_sin", "hour_cos",
    "hist_dest_txn_count", "hist_dest_unique_origin_count", "hist_dest_total_amount",
    "hist_dest_avg_amount", "hist_dest_max_amount", "amount_to_dest_avg_ratio",
    "amount_to_type_avg_ratio",
]
results["feature_ranges"] = {
    c: {"min": float(df[c].min()), "max": float(df[c].max()), "mean": float(df[c].mean())}
    for c in range_cols
}

# ratio sentinel cap respected
results["ratio_sentinel_cap_respected"] = {
    "amount_to_origin_balance_ratio_max_le_1000": bool(df["amount_to_origin_balance_ratio"].max() <= 1000.0),
    "amount_to_dest_avg_ratio_max_le_1000": bool(df["amount_to_dest_avg_ratio"].max() <= 1000.0),
    "amount_to_type_avg_ratio_max_le_1000": bool(df["amount_to_type_avg_ratio"].max() <= 1000.0),
}

# leakage sanity: single-feature separation of isFraud (should be strong but not deterministic
# for legitimate signal features; should be exactly deterministic ONLY for isFlaggedFraud which
# we explicitly exclude from features)
def single_feature_auc(col):
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(df["isFraud"], df[col]))
    except Exception as e:
        return str(e)

try:
    import sklearn  # noqa
    has_sklearn = True
except ImportError:
    has_sklearn = False

if has_sklearn:
    leakage_check_cols = [
        "amount", "drained_to_zero", "amount_to_origin_balance_ratio", "is_transfer",
        "is_cash_out", "high_amount_indicator", "hist_dest_txn_count", "amount_to_dest_avg_ratio",
        "dest_balance_artifact_flag", "isFlaggedFraud",
    ]
    results["single_feature_auc_vs_isFraud"] = {c: single_feature_auc(c) for c in leakage_check_cols}
else:
    results["single_feature_auc_vs_isFraud"] = "sklearn not available, skipped"

# restriction effect: all types vs TRANSFER+CASH_OUT only
n_all = len(df)
n_restricted = int(df["eligible_for_fraud_model"].sum())
fraud_all = int(df["isFraud"].sum())
fraud_restricted = int(df.loc[df["eligible_for_fraud_model"] == 1, "isFraud"].sum())
results["type_restriction_effect"] = {
    "all_types_rows": n_all,
    "all_types_fraud_rows": fraud_all,
    "all_types_fraud_pct": round(fraud_all / n_all * 100, 6),
    "restricted_rows": n_restricted,
    "restricted_fraud_rows": fraud_restricted,
    "restricted_fraud_pct": round(fraud_restricted / n_restricted * 100, 6),
    "pct_of_all_fraud_retained_in_restricted": round(fraud_restricted / fraud_all * 100, 4) if fraud_all else None,
    "row_reduction_pct": round((1 - n_restricted / n_all) * 100, 2),
}

# time window table checks
results["time_window"] = {
    "rows": len(twf),
    "columns": list(twf.columns),
    "nan_counts_nonzero": {c: int(v) for c, v in twf.isna().sum().items() if v > 0},
    "gt_fraud_count_total": int(twf["gt_fraud_count"].sum()),
    "gt_fraud_count_matches_txn_level": int(twf["gt_fraud_count"].sum()) == int(df["isFraud"].sum()),
}

with open(OUT_PATH, "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"Validation stats written to {OUT_PATH}")
print(json.dumps(results, indent=2, default=str)[:3000])
