"""
Unit tests for the pure, testable functions in src/models/spike_detection.py
(method_statistical, method_isolation_forest, evaluate_spike_method,
root_cause_analysis) using a small synthetic per-step table - NOT the full
743-step / 6.36M-row pipeline (that requires the raw PaySim CSV and is
exercised by running the script directly, per reports/provenance.md).
"""
import numpy as np
import pandas as pd
import pytest

from models.spike_detection import (
    PREDICTIVE_SPIKE_FEATURES,
    Z_SCORE_THRESHOLD,
    evaluate_spike_method,
    method_isolation_forest,
    method_statistical,
    root_cause_analysis,
)


def _make_synthetic_twf(n_steps=20, spike_steps=(5, 12)):
    rng = np.random.RandomState(0)
    steps = np.arange(1, n_steps + 1)
    df = pd.DataFrame({"step": steps})
    df["hour_of_day"] = df["step"] % 24
    df["day_index"] = df["step"] // 24
    df["transaction_count"] = rng.randint(50, 100, size=n_steps)
    df["transfer_count"] = df["transaction_count"] // 2
    df["cash_out_count"] = df["transaction_count"] // 2
    df["high_amount_count"] = rng.randint(1, 5, size=n_steps)
    df["predicted_high_risk_count"] = rng.randint(1, 5, size=n_steps)
    df["gt_fraud_count"] = rng.randint(0, 2, size=n_steps)

    for col, prefix in [("transfer_count", "transfer_count"), ("cash_out_count", "cash_out_count"),
                        ("high_amount_count", "high_amount_count"),
                        ("predicted_high_risk_count", "predicted_high_risk_count"),
                        ("transaction_count", "txn_count")]:
        df[f"hist_avg_{prefix}_same_hour"] = df[col].mean()
        df[f"hist_std_{prefix}_same_hour"] = max(df[col].std(), 1.0)
        df[f"{prefix}_vs_hour_baseline_ratio"] = df[col] / df[f"hist_avg_{prefix}_same_hour"]
        df[f"{prefix}_zscore_vs_hour_baseline"] = (
            (df[col] - df[f"hist_avg_{prefix}_same_hour"]) / df[f"hist_std_{prefix}_same_hour"]
        )

    df["average_amount"] = rng.uniform(1000, 5000, size=n_steps)
    df["hist_avg_amount_same_hour"] = df["average_amount"].mean()
    df["amount_vs_hour_baseline_ratio"] = df["average_amount"] / df["hist_avg_amount_same_hour"]
    df["gt_fraud_rate"] = df["gt_fraud_count"] / df["transaction_count"]

    # inject an obvious spike: predicted_high_risk_count z-score way above threshold
    for s in spike_steps:
        idx = df.index[df["step"] == s][0]
        df.loc[idx, "predicted_high_risk_count_zscore_vs_hour_baseline"] = Z_SCORE_THRESHOLD + 3.0
        df.loc[idx, "gt_fraud_spike"] = 1
        for feat in PREDICTIVE_SPIKE_FEATURES:
            df.loc[idx, feat] = df.loc[idx, feat] * 3 if feat in df.columns else 3.0

    df["gt_fraud_spike"] = df.get("gt_fraud_spike", 0)
    df["gt_fraud_spike"] = df["gt_fraud_spike"].fillna(0).astype(int)
    return df


def test_method_statistical_flags_injected_spike():
    twf = _make_synthetic_twf(spike_steps=(5,))
    pred = method_statistical(twf)
    flagged_steps = twf.loc[pred == 1, "step"].tolist()
    assert 5 in flagged_steps


def test_method_isolation_forest_returns_predictions_and_scores(monkeypatch):
    import models.spike_detection as sd
    twf = _make_synthetic_twf(n_steps=30, spike_steps=(10, 20))
    monkeypatch.setattr(sd, "TRAIN_STEPS", (1, 30))  # whole synthetic range is "train" for this unit test
    pred, scores, model = method_isolation_forest(twf)
    assert len(pred) == len(twf)
    assert len(scores) == len(twf)
    assert set(np.unique(pred)).issubset({0, 1})


def test_evaluate_spike_method_confusion_counts():
    twf = _make_synthetic_twf(spike_steps=(5, 12))
    y_pred = (twf["step"].isin([5, 12, 7])).astype(int)  # 2 correct, 1 false positive
    result = evaluate_spike_method("unit_test_method", twf, y_pred, restrict_to_test=False)
    assert result["tp"] == 2
    assert result["fp"] == 1
    assert result["fn"] == 0
    assert result["n_ground_truth_spikes"] == 2
    assert 0.0 <= result["precision"] <= 1.0
    assert 0.0 <= result["recall"] <= 1.0


def test_root_cause_analysis_produces_explanation_per_step():
    twf = _make_synthetic_twf(spike_steps=(5,))
    # exaggerate step 5's transfer_count so root_cause finds a clear driver
    idx = twf.index[twf["step"] == 5][0]
    twf["transfer_count"] = twf["transfer_count"].astype("float64")
    twf.loc[idx, "transfer_count"] = twf.loc[idx, "hist_avg_transfer_count_same_hour"] * 5
    twf.loc[idx, "transfer_count_vs_hour_baseline_ratio"] = 5.0

    explanations = root_cause_analysis(twf, [5], top_n=1)
    assert len(explanations) == 1
    exp = explanations[0]
    assert exp["step"] == 5
    assert "TRANSFER" in exp["explanation"]
