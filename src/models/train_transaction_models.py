"""
Phase 3, Part A - Transaction-level fraud risk model training & evaluation.
AI Fraud-Spike & Risk Detection System (Razorpay Buildathon, Track 02)

Trains three models (Logistic Regression, LightGBM, XGBoost) on a
leakage-safe, time-split, TRANSFER+CASH_OUT-restricted slice of the Phase 2
engineered dataset. No hyperparameter tuning is done against the test set;
the test set is scored exactly once, at the end, for final reporting.
"""
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_score, recall_score,
    f1_score, confusion_matrix, precision_recall_curve,
)

import lightgbm as lgb
import xgboost as xgb

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TXN_PARQUET = os.path.join(ROOT, "data", "processed", "engineered_transactions.parquet")
MODELS_DIR = os.path.join(ROOT, "models")
REPORTS_DIR = os.path.join(ROOT, "reports")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

RANDOM_SEED = 42

TRAIN_STEPS = (1, 520)
VAL_STEPS = (521, 631)
TEST_STEPS = (632, 743)

# Final feature list for the transaction-risk model. Excludes: isFraud
# (target), isFlaggedFraud (Phase 1/2 decision), raw nameOrig/nameDest
# (Phase 2 decision), raw step and day_index (ordinal time position -
# excluded from the CLASSIFIER to avoid the model keying on "which chunk of
# the 743-step timeline this row is from" rather than a generalizable risk
# pattern; hour_of_day + cyclic encoding still capture the legitimate
# time-of-day signal), raw `type` string (redundant with is_transfer once
# restricted to 2 types), raw balance columns (Phase 2 recommended the
# engineered discrepancy features over these noisy raw fields), and
# eligible_for_fraud_model (constant 1 after filtering, no information).
FEATURE_LIST = [
    "amount", "log_amount",
    "balance_change_orig", "balance_error_orig", "drained_to_zero",
    "origin_balance_zero_before", "amount_to_origin_balance_ratio",
    "balance_change_dest", "balance_error_dest", "dest_balance_artifact_flag",
    "hour_of_day", "hour_sin", "hour_cos",
    "is_transfer",
    "hist_dest_txn_count", "hist_dest_unique_origin_count",
    "hist_dest_total_amount", "hist_dest_avg_amount", "hist_dest_max_amount",
    "destination_is_new",
    "amount_to_dest_avg_ratio", "amount_to_type_avg_ratio",
    "high_amount_indicator",
]
TARGET = "isFraud"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def load_modeling_data():
    log("Loading engineered_transactions.parquet ...")
    cols = FEATURE_LIST + [TARGET, "step", "eligible_for_fraud_model", "type"]
    df = pd.read_parquet(TXN_PARQUET, columns=cols)
    n_all = len(df)
    n_all_fraud = int(df[TARGET].sum())

    restricted = df.loc[df["eligible_for_fraud_model"] == 1].drop(columns=["eligible_for_fraud_model"]).copy()
    n_restricted = len(restricted)
    n_restricted_fraud = int(restricted[TARGET].sum())

    audit = {
        "all_types_rows": n_all,
        "all_types_fraud_rows": n_all_fraud,
        "restricted_rows": n_restricted,
        "restricted_fraud_rows": n_restricted_fraud,
        "pct_of_fraud_retained": round(n_restricted_fraud / n_all_fraud * 100, 4) if n_all_fraud else None,
        "restricted_fraud_rate_pct": round(n_restricted_fraud / n_restricted * 100, 6),
        "restricted_type_counts": {str(k): int(v) for k, v in restricted["type"].value_counts().items()},
    }
    assert n_restricted_fraud == n_all_fraud, "Restriction must retain 100% of fraud!"
    log(f"Restriction check: {n_restricted:,} rows ({n_restricted/n_all*100:.1f}% of full), "
        f"fraud retained {n_restricted_fraud}/{n_all_fraud} = {audit['pct_of_fraud_retained']}%")

    restricted = restricted.drop(columns=["type"])
    del df
    return restricted, audit


def time_split(df):
    train = df.loc[(df["step"] >= TRAIN_STEPS[0]) & (df["step"] <= TRAIN_STEPS[1])]
    val = df.loc[(df["step"] >= VAL_STEPS[0]) & (df["step"] <= VAL_STEPS[1])]
    test = df.loc[(df["step"] >= TEST_STEPS[0]) & (df["step"] <= TEST_STEPS[1])]

    def split_stats(name, part):
        n = len(part)
        nf = int(part[TARGET].sum())
        return {
            "name": name, "rows": n, "fraud_rows": nf,
            "fraud_rate_pct": round(nf / n * 100, 6) if n else None,
            "imbalance_ratio": round((n - nf) / nf, 1) if nf else None,
        }

    split_report = [split_stats("train", train), split_stats("val", val), split_stats("test", test)]
    for s in split_report:
        log(f"{s['name']:5s}: {s['rows']:>10,} rows, {s['fraud_rows']:>5,} fraud "
            f"({s['fraud_rate_pct']}%), imbalance {s['imbalance_ratio']}:1")

    X_train, y_train = train[FEATURE_LIST], train[TARGET]
    X_val, y_val = val[FEATURE_LIST], val[TARGET]
    X_test, y_test = test[FEATURE_LIST], test[TARGET]
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), split_report


def evaluate(model_name, y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    pr_auc = average_precision_score(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_prob)
    precision_at_recall = {}
    for target_recall in [0.5, 0.7, 0.8, 0.9]:
        idx = np.where(rec_curve >= target_recall)[0]
        if len(idx) > 0:
            best_idx = idx[np.argmax(prec_curve[idx])]
            precision_at_recall[f"recall_{int(target_recall*100)}"] = {
                "achievable": True,
                "precision": round(float(prec_curve[best_idx]), 6),
                "actual_recall": round(float(rec_curve[best_idx]), 6),
            }
        else:
            precision_at_recall[f"recall_{int(target_recall*100)}"] = {"achievable": False}

    return {
        "model": model_name, "threshold": threshold,
        "pr_auc": round(float(pr_auc), 6), "roc_auc": round(float(roc_auc), 6),
        "precision": round(float(precision), 6), "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
        "confusion_matrix": {"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)},
        "precision_at_recall": precision_at_recall,
        "n_samples": int(len(y_true)), "n_fraud": int(y_true.sum()),
    }


def threshold_sweep(y_true, y_prob, thresholds=None):
    if thresholds is None:
        thresholds = np.round(np.arange(0.01, 1.00, 0.02), 3)
    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        alert_rate = (tp + fp) / len(y_true)
        rows.append({
            "threshold": float(t), "precision": round(float(precision), 6),
            "recall": round(float(recall), 6), "f1": round(float(f1), 6),
            "fp": int(fp), "fn": int(fn), "tp": int(tp), "tn": int(tn),
            "alert_rate": round(float(alert_rate), 8),
        })
    return rows


def train_logistic_regression(X_train, y_train):
    log("Training Logistic Regression ...")
    t0 = time.time()
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_SEED)),
    ])
    pipe.fit(X_train, y_train)
    train_time = time.time() - t0
    log(f"Logistic Regression trained in {train_time:.1f}s")
    return pipe, train_time


def train_lightgbm(X_train, y_train, X_val, y_val):
    log("Training LightGBM ...")
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos
    log(f"LightGBM scale_pos_weight (from TRAIN only) = {scale_pos_weight:.2f}")
    t0 = time.time()
    model = lgb.LGBMClassifier(
        objective="binary", n_estimators=500, learning_rate=0.05, num_leaves=63,
        scale_pos_weight=scale_pos_weight, random_state=RANDOM_SEED, n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="average_precision",
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )
    train_time = time.time() - t0
    log(f"LightGBM trained in {train_time:.1f}s, best_iteration={model.best_iteration_}")
    return model, train_time, scale_pos_weight


def train_xgboost(X_train, y_train, X_val, y_val):
    log("Training XGBoost ...")
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos
    log(f"XGBoost scale_pos_weight (from TRAIN only) = {scale_pos_weight:.2f}")
    t0 = time.time()
    model = xgb.XGBClassifier(
        objective="binary:logistic", n_estimators=500, learning_rate=0.05, max_depth=6,
        scale_pos_weight=scale_pos_weight, random_state=RANDOM_SEED, n_jobs=-1,
        eval_metric="aucpr", early_stopping_rounds=30,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    train_time = time.time() - t0
    log(f"XGBoost trained in {train_time:.1f}s, best_iteration={model.best_iteration}")
    return model, train_time, scale_pos_weight


def measure_inference_speed(model, X, predict_fn=None, n_repeats=5, inner_loops=5):
    # predict_proba on a few tens of thousands of rows can complete faster
    # than clock resolution for some models; repeat each timed block
    # `inner_loops` times so elapsed time is reliably measurable.
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        for _ in range(inner_loops):
            if predict_fn:
                predict_fn(X)
            else:
                model.predict_proba(X)
        times.append((time.perf_counter() - t0) / inner_loops)
    best = max(min(times), 1e-6)
    return {"total_sec": round(best, 6), "rows_per_sec": round(len(X) / best, 1)}


def main():
    results = {"random_seed": RANDOM_SEED, "train_steps": TRAIN_STEPS, "val_steps": VAL_STEPS, "test_steps": TEST_STEPS,
               "feature_list": FEATURE_LIST, "target": TARGET}

    df, restriction_audit = load_modeling_data()
    results["restriction_audit"] = restriction_audit

    (X_train, y_train), (X_val, y_val), (X_test, y_test), split_report = time_split(df)
    results["split_report"] = split_report
    del df

    model_results = {}

    # --- Logistic Regression ---
    lr_model, lr_train_time = train_logistic_regression(X_train, y_train)
    lr_val_prob = lr_model.predict_proba(X_val)[:, 1]
    lr_test_prob = lr_model.predict_proba(X_test)[:, 1]
    lr_speed = measure_inference_speed(lr_model, X_test)
    model_results["logistic_regression"] = {
        "train_time_sec": round(lr_train_time, 2),
        "inference_speed": lr_speed,
        "val_metrics": evaluate("logistic_regression", y_val, lr_val_prob),
        "test_metrics": evaluate("logistic_regression", y_test, lr_test_prob),
    }
    joblib.dump(lr_model, os.path.join(MODELS_DIR, "logistic_regression.joblib"))

    # --- LightGBM ---
    lgb_model, lgb_train_time, lgb_spw = train_lightgbm(X_train, y_train, X_val, y_val)
    lgb_val_prob = lgb_model.predict_proba(X_val)[:, 1]
    lgb_test_prob = lgb_model.predict_proba(X_test)[:, 1]
    lgb_speed = measure_inference_speed(lgb_model, X_test)
    model_results["lightgbm"] = {
        "train_time_sec": round(lgb_train_time, 2),
        "inference_speed": lgb_speed,
        "scale_pos_weight": round(lgb_spw, 2),
        "best_iteration": int(lgb_model.best_iteration_),
        "val_metrics": evaluate("lightgbm", y_val, lgb_val_prob),
        "test_metrics": evaluate("lightgbm", y_test, lgb_test_prob),
    }
    joblib.dump(lgb_model, os.path.join(MODELS_DIR, "lightgbm.joblib"))

    # --- XGBoost ---
    xgb_model, xgb_train_time, xgb_spw = train_xgboost(X_train, y_train, X_val, y_val)
    xgb_val_prob = xgb_model.predict_proba(X_val)[:, 1]
    xgb_test_prob = xgb_model.predict_proba(X_test)[:, 1]
    xgb_speed = measure_inference_speed(xgb_model, X_test)
    model_results["xgboost"] = {
        "train_time_sec": round(xgb_train_time, 2),
        "inference_speed": xgb_speed,
        "scale_pos_weight": round(xgb_spw, 2),
        "best_iteration": int(xgb_model.best_iteration),
        "val_metrics": evaluate("xgboost", y_val, xgb_val_prob),
        "test_metrics": evaluate("xgboost", y_test, xgb_test_prob),
    }
    joblib.dump(xgb_model, os.path.join(MODELS_DIR, "xgboost.joblib"))

    results["model_results"] = model_results

    # --- Select best model by validation PR-AUC ---
    best_name = max(model_results, key=lambda k: model_results[k]["val_metrics"]["pr_auc"])
    results["selected_model"] = best_name
    log(f"Selected model by validation PR-AUC: {best_name} "
        f"(val PR-AUC={model_results[best_name]['val_metrics']['pr_auc']})")

    # --- Threshold sweep on VALIDATION for the selected model ---
    val_probs = {"logistic_regression": lr_val_prob, "lightgbm": lgb_val_prob, "xgboost": xgb_val_prob}
    test_probs = {"logistic_regression": lr_test_prob, "lightgbm": lgb_test_prob, "xgboost": xgb_test_prob}
    sweep = threshold_sweep(y_val, val_probs[best_name])
    results["threshold_sweep_validation"] = {"model": best_name, "rows": sweep}

    # pick 3 operating points from the validation sweep
    def closest(rows, key, target, mode="ge"):
        candidates = [r for r in rows if (r[key] >= target if mode == "ge" else r[key] <= target)]
        if not candidates:
            return None
        return max(candidates, key=lambda r: r["threshold"]) if mode == "ge" else min(candidates, key=lambda r: r["threshold"])

    high_recall_candidates = [r for r in sweep if r["recall"] >= 0.90]
    high_recall_point = max(high_recall_candidates, key=lambda r: r["precision"]) if high_recall_candidates else \
        max(sweep, key=lambda r: r["recall"])
    balanced_point = max(sweep, key=lambda r: r["f1"])
    high_precision_candidates = [r for r in sweep if r["precision"] >= 0.80]
    high_precision_point = max(high_precision_candidates, key=lambda r: r["recall"]) if high_precision_candidates else \
        max(sweep, key=lambda r: r["precision"])

    operating_modes = {
        "high_recall": high_recall_point,
        "balanced": balanced_point,
        "high_precision": high_precision_point,
    }
    results["operating_modes_validation"] = operating_modes

    # --- Apply the 3 chosen thresholds to the UNTOUCHED test set, exactly once ---
    final_test_eval = {}
    for mode_name, point in operating_modes.items():
        t = point["threshold"]
        final_test_eval[mode_name] = evaluate(best_name, y_test, test_probs[best_name], threshold=t)
    results["final_test_evaluation_by_mode"] = final_test_eval

    with open(os.path.join(REPORTS_DIR, "phase3_model_stats.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    log("Wrote reports/phase3_model_stats.json")

    # save probability arrays + y for downstream SHAP/importance/notebook work
    np.savez(
        os.path.join(MODELS_DIR, "predictions_cache.npz"),
        y_val=y_val.to_numpy(), y_test=y_test.to_numpy(),
        lr_val=lr_val_prob, lr_test=lr_test_prob,
        lgb_val=lgb_val_prob, lgb_test=lgb_test_prob,
        xgb_val=xgb_val_prob, xgb_test=xgb_test_prob,
    )

    metadata = {
        "random_seed": RANDOM_SEED,
        "feature_list": FEATURE_LIST,
        "target": TARGET,
        "train_steps": TRAIN_STEPS, "val_steps": VAL_STEPS, "test_steps": TEST_STEPS,
        "modeling_universe": "TRANSFER + CASH_OUT only (eligible_for_fraud_model == 1)",
        "preprocessing": {
            "logistic_regression": "StandardScaler + class_weight='balanced'",
            "lightgbm": f"scale_pos_weight={model_results['lightgbm']['scale_pos_weight']} (from TRAIN only)",
            "xgboost": f"scale_pos_weight={model_results['xgboost']['scale_pos_weight']} (from TRAIN only)",
        },
        "model_params": {
            "logistic_regression": {"class_weight": "balanced", "max_iter": 2000, "random_state": RANDOM_SEED},
            "lightgbm": {"n_estimators": 500, "learning_rate": 0.05, "num_leaves": 63,
                         "best_iteration": model_results["lightgbm"]["best_iteration"]},
            "xgboost": {"n_estimators": 500, "learning_rate": 0.05, "max_depth": 6,
                        "best_iteration": model_results["xgboost"]["best_iteration"]},
        },
        "selected_model": best_name,
        "selection_criterion": "highest validation PR-AUC",
        "operating_modes_validation_thresholds": {k: v["threshold"] for k, v in operating_modes.items()},
        "model_files": {
            "logistic_regression": "models/logistic_regression.joblib",
            "lightgbm": "models/lightgbm.joblib",
            "xgboost": "models/xgboost.joblib",
        },
    }
    with open(os.path.join(MODELS_DIR, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    log("Wrote models/model_metadata.json")

    log("Part A complete.")
    return results


if __name__ == "__main__":
    main()
