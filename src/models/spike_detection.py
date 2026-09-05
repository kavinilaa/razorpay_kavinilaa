"""
Phase 3, Part B - Fraud-spike detection.
AI Fraud-Spike & Risk Detection System (Razorpay Buildathon, Track 02)

Builds a volume-conditioned, leakage-safe spike-detection layer on top of
the Phase 2 per-step aggregates, adds a predictive signal derived from the
Part A transaction-risk model (applying an already-trained, fixed model to
generate a feature is not leakage -- it uses no future information), defines
an explicit ground-truth spike rule for evaluation, and tests two detection
methods: a statistical z-score baseline and an Isolation Forest.

GROUND TRUTH vs PREDICTIVE is enforced by naming convention throughout:
any column/quantity built from isFraud is prefixed `gt_` and is used only
for defining ground truth and for evaluation -- never as a detector input.
"""
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TXN_PARQUET = os.path.join(ROOT, "data", "processed", "engineered_transactions.parquet")
TIME_PARQUET = os.path.join(ROOT, "data", "processed", "time_window_features.parquet")
MODELS_DIR = os.path.join(ROOT, "models")
REPORTS_DIR = os.path.join(ROOT, "reports")

RANDOM_SEED = 42
TRAIN_STEPS = (1, 520)
TEST_STEPS = (632, 743)

# Ground-truth spike rule (documented, not hidden):
# A step is a ground-truth fraud spike if BOTH:
#   (a) actual fraud count that step >= MIN_ABS_FRAUD_COUNT (avoids Phase 1's
#       small-sample-noise failure mode, where low-volume steps hit 100%
#       fraud "rate" on a handful of transactions)
#   (b) actual fraud count >= SPIKE_MULTIPLIER x the expected fraud count for
#       that hour-of-day, where "expected" = historical (strictly prior
#       days) average fraud RATE for that hour-of-day, applied to this
#       step's OWN transaction count (so a volume change alone doesn't
#       trigger a spike -- fraud has to be elevated beyond what the current
#       volume would already explain).
MIN_ABS_FRAUD_COUNT = 3
SPIKE_MULTIPLIER = 2.0

# Predictive-side spike flag thresholds (statistical baseline method)
Z_SCORE_THRESHOLD = 2.0

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def build_extended_time_window_features():
    """Extends Phase 2's time_window_features.parquet with:
    - per-type-count hour-of-day baselines (transfer_count, cash_out_count, high_amount_count)
    - a predicted-high-risk-count signal from the trained Part A model, and its own
      hour-of-day baseline
    All using the same strictly-prior-day expanding-baseline method as Phase 2.
    """
    log("Loading Phase 2 time_window_features.parquet ...")
    twf = pd.read_parquet(TIME_PARQUET)

    def add_hour_baseline(df, col, prefix):
        g = df.groupby("hour_of_day")
        shifted = g[col].shift(1)
        df[f"_shifted_{prefix}"] = shifted
        mean_col = f"hist_avg_{prefix}_same_hour"
        std_col = f"hist_std_{prefix}_same_hour"
        df[mean_col] = df.groupby("hour_of_day")[f"_shifted_{prefix}"].expanding().mean().reset_index(level=0, drop=True)
        df[std_col] = df.groupby("hour_of_day")[f"_shifted_{prefix}"].expanding().std().reset_index(level=0, drop=True)
        global_mean = df[col].shift(1).expanding().mean()
        global_std = df[col].shift(1).expanding().std()
        df[mean_col] = df[mean_col].fillna(global_mean).fillna(df[col])
        df[std_col] = df[std_col].fillna(global_std).fillna(0.0)
        ratio_col = f"{prefix}_vs_hour_baseline_ratio"
        df[ratio_col] = (df[col] / df[mean_col].replace(0, np.nan)).fillna(1.0)
        zscore_col = f"{prefix}_zscore_vs_hour_baseline"
        safe_std = df[std_col].replace(0, np.nan)
        df[zscore_col] = ((df[col] - df[mean_col]) / safe_std).fillna(0.0)
        df.drop(columns=[f"_shifted_{prefix}"], inplace=True)
        return df

    twf = twf.sort_values("step").reset_index(drop=True)
    for col, prefix in [("transfer_count", "transfer_count"), ("cash_out_count", "cash_out_count"),
                        ("high_amount_count", "high_amount_count")]:
        twf = add_hour_baseline(twf, col, prefix)

    # --- ground-truth expected fraud count, for the spike DEFINITION only ---
    twf["gt_expected_fraud_count_same_hour"] = twf["gt_hist_avg_fraud_rate_same_hour"] * twf["transaction_count"]
    twf["gt_expected_fraud_count_same_hour"] = twf["gt_expected_fraud_count_same_hour"].fillna(0.0)

    twf["gt_fraud_spike"] = (
        (twf["gt_fraud_count"] >= MIN_ABS_FRAUD_COUNT) &
        (twf["gt_fraud_count"] >= SPIKE_MULTIPLIER * twf["gt_expected_fraud_count_same_hour"])
    ).astype(int)

    # --- predictive signal: score all TRANSFER/CASH_OUT transactions with the
    # already-trained (fixed) Part A model, aggregate per step ---
    log("Scoring all eligible transactions with the trained transaction model...")
    with open(os.path.join(MODELS_DIR, "model_metadata.json")) as f:
        meta = json.load(f)
    best_model_name = meta["selected_model"]
    feature_list = meta["feature_list"]
    model = joblib.load(os.path.join(MODELS_DIR, f"{best_model_name}.joblib"))

    cols = feature_list + ["step", "eligible_for_fraud_model"]
    txn = pd.read_parquet(TXN_PARQUET, columns=cols)
    txn = txn.loc[txn["eligible_for_fraud_model"] == 1]
    probs = model.predict_proba(txn[feature_list])[:, 1]
    txn = txn.assign(pred_prob=probs)

    # threshold: use the "balanced" operating point chosen in Part A
    balanced_threshold = meta["operating_modes_validation_thresholds"]["balanced"]
    txn["pred_high_risk"] = (txn["pred_prob"] >= balanced_threshold).astype(int)

    per_step_pred = txn.groupby("step").agg(
        predicted_high_risk_count=("pred_high_risk", "sum"),
        predicted_risk_score_sum=("pred_prob", "sum"),
    ).reset_index()
    del txn

    twf = twf.merge(per_step_pred, on="step", how="left")
    twf["predicted_high_risk_count"] = twf["predicted_high_risk_count"].fillna(0).astype(int)
    twf["predicted_risk_score_sum"] = twf["predicted_risk_score_sum"].fillna(0.0)

    twf = add_hour_baseline(twf, "predicted_high_risk_count", "predicted_high_risk_count")

    log(f"Extended time-window feature table: {twf.shape}")
    return twf, best_model_name, balanced_threshold


PREDICTIVE_SPIKE_FEATURES = [
    "txn_count_vs_hour_baseline_ratio", "amount_vs_hour_baseline_ratio",
    "transfer_count_vs_hour_baseline_ratio", "cash_out_count_vs_hour_baseline_ratio",
    "high_amount_count_vs_hour_baseline_ratio",
    "predicted_high_risk_count_vs_hour_baseline_ratio",
]


def method_statistical(twf):
    """Method A: flag a step if the predicted-high-risk-count z-score vs. its
    hour-of-day historical baseline exceeds Z_SCORE_THRESHOLD. Purely
    predictive (no isFraud used)."""
    pred = (twf["predicted_high_risk_count_zscore_vs_hour_baseline"] >= Z_SCORE_THRESHOLD).astype(int)
    return pred


def method_isolation_forest(twf):
    """Method B: Isolation Forest fit ONLY on TRAIN-period steps (1-520) over
    the predictive spike-feature vector, then scored on all steps."""
    train_mask = (twf["step"] >= TRAIN_STEPS[0]) & (twf["step"] <= TRAIN_STEPS[1])
    X = twf[PREDICTIVE_SPIKE_FEATURES].fillna(1.0)
    iso = IsolationForest(n_estimators=200, contamination="auto", random_state=RANDOM_SEED)
    iso.fit(X.loc[train_mask])
    raw_pred = iso.predict(X)  # -1 = anomaly, 1 = normal
    pred = (raw_pred == -1).astype(int)
    scores = -iso.decision_function(X)  # higher = more anomalous
    return pred, scores, iso


def evaluate_spike_method(name, twf, y_pred, restrict_to_test=False):
    mask = pd.Series(True, index=twf.index)
    if restrict_to_test:
        mask = (twf["step"] >= TEST_STEPS[0]) & (twf["step"] <= TEST_STEPS[1])
    y_true = twf.loc[mask, "gt_fraud_spike"].to_numpy()
    y_p = np.asarray(y_pred)[mask.to_numpy()]

    tp = int(((y_true == 1) & (y_p == 1)).sum())
    fp = int(((y_true == 0) & (y_p == 1)).sum())
    fn = int(((y_true == 1) & (y_p == 0)).sum())
    tn = int(((y_true == 0) & (y_p == 0)).sum())
    precision = precision_score(y_true, y_p, zero_division=0)
    recall = recall_score(y_true, y_p, zero_division=0)
    false_alert_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    alert_rate = (tp + fp) / len(y_true) if len(y_true) else 0.0

    # detection delay: by design, this is same-step (nowcasting) monitoring,
    # not multi-step-ahead forecasting -- delay is 0 steps for every alert
    # that fires on the same step as the ground-truth spike.
    return {
        "method": name, "scope": "test_only" if restrict_to_test else "all_steps",
        "n_steps": int(len(y_true)), "n_ground_truth_spikes": int(y_true.sum()),
        "n_predicted_spikes": int(y_p.sum()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(float(precision), 4), "recall": round(float(recall), 4),
        "false_alert_rate": round(float(false_alert_rate), 6),
        "alert_rate": round(float(alert_rate), 6),
        "detection_delay_steps": 0,
    }


def root_cause_analysis(twf, spike_steps, top_n=5):
    """For each flagged spike step, compare its per-type counts and amount
    behavior against the hour-of-day historical baseline to build a
    data-supported explanation."""
    explanations = []
    for step in spike_steps[:top_n]:
        row = twf.loc[twf["step"] == step].iloc[0]
        drivers = []
        for count_col, baseline_col, label in [
            ("transfer_count", "hist_avg_transfer_count_same_hour", "TRANSFER"),
            ("cash_out_count", "hist_avg_cash_out_count_same_hour", "CASH_OUT"),
            ("high_amount_count", "hist_avg_high_amount_count_same_hour", "high-amount"),
        ]:
            baseline = row[baseline_col]
            actual = row[count_col]
            if baseline and baseline > 0:
                ratio = actual / baseline
                if ratio >= 1.5:
                    drivers.append((label, ratio, actual, baseline))
        drivers.sort(key=lambda d: -d[1])
        explanation = None
        if drivers:
            label, ratio, actual, baseline = drivers[0]
            explanation = (
                f"Step {int(step)} (hour_of_day={int(row['hour_of_day'])}, day_index={int(row['day_index'])}): "
                f"risk activity increased primarily because {label} transactions were {ratio:.1f}x above "
                f"the historical hour-of-day baseline ({actual:.0f} actual vs. {baseline:.1f} expected)."
            )
        explanations.append({
            "step": int(step),
            "hour_of_day": int(row["hour_of_day"]),
            "day_index": int(row["day_index"]),
            "transaction_count": int(row["transaction_count"]),
            "txn_count_vs_baseline": round(float(row["txn_count_vs_hour_baseline_ratio"]), 3),
            "transfer_count": int(row["transfer_count"]),
            "transfer_count_vs_baseline": round(float(row["transfer_count_vs_hour_baseline_ratio"]), 3),
            "cash_out_count": int(row["cash_out_count"]),
            "cash_out_count_vs_baseline": round(float(row["cash_out_count_vs_hour_baseline_ratio"]), 3),
            "high_amount_count": int(row["high_amount_count"]),
            "high_amount_count_vs_baseline": round(float(row["high_amount_count_vs_hour_baseline_ratio"]), 3),
            "predicted_high_risk_count": int(row["predicted_high_risk_count"]),
            "predicted_high_risk_vs_baseline": round(float(row["predicted_high_risk_count_vs_hour_baseline_ratio"]), 3),
            "gt_fraud_count": int(row["gt_fraud_count"]),
            "gt_fraud_rate_pct": round(float(row["gt_fraud_rate"]) * 100, 4),
            "explanation": explanation or "No single transaction-type driver exceeded 1.5x its hour-of-day baseline.",
        })
    return explanations


def main():
    t0 = time.time()
    twf, best_model_name, balanced_threshold = build_extended_time_window_features()

    stat_pred = method_statistical(twf)
    iso_pred, iso_scores, iso_model = method_isolation_forest(twf)
    twf["stat_method_pred"] = stat_pred
    twf["iso_method_pred"] = iso_pred
    twf["iso_method_score"] = iso_scores

    results = {
        "spike_definition": {
            "min_abs_fraud_count": MIN_ABS_FRAUD_COUNT,
            "spike_multiplier": SPIKE_MULTIPLIER,
            "rule": "gt_fraud_count >= MIN_ABS_FRAUD_COUNT AND gt_fraud_count >= SPIKE_MULTIPLIER * gt_expected_fraud_count_same_hour",
            "gt_expected_fraud_count_same_hour": "gt_hist_avg_fraud_rate_same_hour (strictly prior days, same hour-of-day) * this step's own transaction_count",
        },
        "predictive_features_used": PREDICTIVE_SPIKE_FEATURES,
        "transaction_model_used_for_predicted_risk_feature": best_model_name,
        "transaction_model_threshold_used": balanced_threshold,
        "statistical_method": {"z_score_threshold": Z_SCORE_THRESHOLD},
        "isolation_forest_method": {"n_estimators": 200, "contamination": "auto", "trained_on": f"steps {TRAIN_STEPS[0]}-{TRAIN_STEPS[1]} only"},
        "n_ground_truth_spikes_total": int(twf["gt_fraud_spike"].sum()),
        "ground_truth_spike_steps": twf.loc[twf["gt_fraud_spike"] == 1, "step"].tolist(),
    }

    eval_rows = []
    for scope in [False, True]:
        eval_rows.append(evaluate_spike_method("statistical_zscore", twf, stat_pred, restrict_to_test=scope))
        eval_rows.append(evaluate_spike_method("isolation_forest", twf, iso_pred, restrict_to_test=scope))
    results["evaluation"] = eval_rows

    # root-cause analysis on ground-truth spike steps found by either method within the test window
    gt_test_spikes = twf.loc[(twf["gt_fraud_spike"] == 1) & (twf["step"] >= TEST_STEPS[0]) & (twf["step"] <= TEST_STEPS[1]), "step"].tolist()
    results["root_cause_analysis_test_period_gt_spikes"] = root_cause_analysis(twf, gt_test_spikes, top_n=10)

    # also show root-cause for a couple of statistically-flagged spikes for illustration
    stat_flagged_steps = twf.loc[twf["stat_method_pred"] == 1, "step"].tolist()
    results["root_cause_analysis_statistical_method_flags"] = root_cause_analysis(twf, stat_flagged_steps, top_n=5)

    with open(os.path.join(REPORTS_DIR, "phase3_spike_stats.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"Wrote reports/phase3_spike_stats.json")

    twf.to_parquet(os.path.join(ROOT, "data", "processed", "time_window_features_extended.parquet"), index=False)
    log(f"Wrote data/processed/time_window_features_extended.parquet")

    joblib.dump(iso_model, os.path.join(MODELS_DIR, "isolation_forest.joblib"))
    log("Wrote models/isolation_forest.joblib")

    log(f"Part B complete in {time.time()-t0:.1f}s")
    return results, twf


if __name__ == "__main__":
    main()
