"""
Phase 4 - Root-cause engine for a flagged fraud-spike step.
AI Fraud-Spike & Risk Detection System (Razorpay Buildathon, Track 02)

Extends the root-cause logic already in src/models/spike_detection.py
(which only compares transfer_count / cash_out_count / high_amount_count
against their hour-of-day baselines) with two additional signals identified
as gaps in reports/phase3_spike_detection.md #16-17:

  1. `predicted_high_risk_count` vs. its own hour-of-day baseline (the spike
     layer's own risk-model-derived signal - previously computed but never
     checked by the root-cause function).
  2. Destination concentration within the step, when raw per-transaction
     data for that step is available - because reports/phase3_spike_detection.md
     found every inspected ground-truth spike was a small number of
     concentrated fraud transactions in an otherwise quiet step, a pattern
     the type-count-ratio-only rule could not explain.

Every number in the output is computed from whatever step-aggregate row /
per-transaction data is passed in - nothing here is hard-coded per spike.
"""
from typing import Optional

import pandas as pd

DRIVER_RATIO_THRESHOLD = 1.5  # matches the threshold used in spike_detection.py's root_cause_analysis()
DEST_CONCENTRATION_SHARE_THRESHOLD = 0.30  # a single destination receiving >=30% of a step's transactions/amount

# (row_count_col, row_baseline_col, label)
TYPE_COUNT_FACTORS = [
    ("transfer_count", "hist_avg_transfer_count_same_hour", "TRANSFER volume"),
    ("cash_out_count", "hist_avg_cash_out_count_same_hour", "CASH_OUT volume"),
    ("high_amount_count", "hist_avg_high_amount_count_same_hour", "high-amount transaction volume"),
    ("predicted_high_risk_count", "hist_avg_predicted_high_risk_count_same_hour", "model-flagged high-risk transaction volume"),
]

# also always report overall volume/amount deviation for context, even
# though they are control variables (Phase 1 finding: volume spikes alone
# are not fraud) rather than drivers on their own.
CONTEXT_FACTORS = [
    ("transaction_count", "hist_avg_txn_count_same_hour", "overall transaction volume"),
    ("average_amount", "hist_avg_amount_same_hour", "average transaction amount"),
]


def _ratio(actual, baseline):
    if baseline is None or baseline == 0:
        return None
    return actual / baseline


def _destination_concentration(transactions_in_step: pd.DataFrame) -> Optional[dict]:
    if transactions_in_step is None or len(transactions_in_step) == 0:
        return None
    if "nameDest" not in transactions_in_step.columns:
        return None

    total_txns = len(transactions_in_step)
    counts = transactions_in_step["nameDest"].value_counts()
    top_dest = counts.index[0]
    top_dest_count = int(counts.iloc[0])
    count_share = top_dest_count / total_txns

    amount_share = None
    if "amount" in transactions_in_step.columns:
        total_amount = transactions_in_step["amount"].sum()
        top_dest_amount = transactions_in_step.loc[transactions_in_step["nameDest"] == top_dest, "amount"].sum()
        if total_amount > 0:
            amount_share = float(top_dest_amount / total_amount)

    return {
        "top_destination": str(top_dest),
        "top_destination_txn_count": top_dest_count,
        "total_txns_in_step": total_txns,
        "txn_count_share": round(float(count_share), 4),
        "amount_share": round(amount_share, 4) if amount_share is not None else None,
    }


def analyze_spike(step_row: dict, transactions_in_step: Optional[pd.DataFrame] = None) -> dict:
    """Build a structured, evidence-backed root-cause explanation for one
    flagged (or ground-truth) spike step.

    Parameters
    ----------
    step_row : dict-like (a row from data/processed/time_window_features_extended.parquet,
        or any dict carrying the same keys). Missing keys are handled gracefully
        (that factor is simply skipped, never fabricated).
    transactions_in_step : optional DataFrame of the raw transactions belonging to
        this step (needs at least a `nameDest` column, ideally also `amount`).
        When omitted, destination concentration is reported as "not evaluated"
        rather than guessed.
    """
    def g(key):
        return step_row.get(key) if hasattr(step_row, "get") else step_row[key] if key in step_row else None

    contributors = []
    baseline_comparison = {}
    evidence = []

    for count_col, baseline_col, label in TYPE_COUNT_FACTORS:
        actual = g(count_col)
        baseline = g(baseline_col)
        if actual is None or baseline is None:
            continue
        ratio = _ratio(actual, baseline)
        baseline_comparison[label] = {"actual": actual, "baseline": baseline, "ratio": round(ratio, 3) if ratio is not None else None}
        if ratio is not None and ratio >= DRIVER_RATIO_THRESHOLD:
            contributors.append({"factor": label, "ratio_vs_baseline": round(ratio, 3), "actual": actual, "baseline": baseline})
            evidence.append(f"{label} was {ratio:.1f}x its hour-of-day baseline ({actual} actual vs. {baseline:.1f} expected).")

    for count_col, baseline_col, label in CONTEXT_FACTORS:
        actual = g(count_col)
        baseline = g(baseline_col)
        if actual is None or baseline is None:
            continue
        ratio = _ratio(actual, baseline)
        baseline_comparison[label] = {"actual": actual, "baseline": baseline, "ratio": round(ratio, 3) if ratio is not None else None}

    dest_concentration = _destination_concentration(transactions_in_step)
    if dest_concentration is not None:
        baseline_comparison["destination_concentration"] = dest_concentration
        if dest_concentration["txn_count_share"] >= DEST_CONCENTRATION_SHARE_THRESHOLD:
            contributors.append({
                "factor": "destination concentration",
                "ratio_vs_baseline": None,
                "actual": dest_concentration["top_destination_txn_count"],
                "baseline": None,
            })
            evidence.append(
                f"A single destination ({dest_concentration['top_destination']}) received "
                f"{dest_concentration['top_destination_txn_count']}/{dest_concentration['total_txns_in_step']} "
                f"({dest_concentration['txn_count_share']*100:.0f}%) of this step's transactions."
            )
    else:
        evidence.append("Destination concentration was not evaluated (no per-transaction data supplied for this step).")

    hour_of_day = g("hour_of_day")
    if hour_of_day is not None:
        evidence.append(f"Step occurred at hour_of_day={hour_of_day} of the simulated 24-hour cycle.")

    contributors.sort(key=lambda c: (c["ratio_vs_baseline"] is None, -(c["ratio_vs_baseline"] or 0)))

    if contributors:
        primary = contributors[0]
        primary_driver = primary["factor"]
    else:
        primary_driver = "no single driver identified"
        evidence.append(
            "No individual factor exceeded the driver threshold "
            f"({DRIVER_RATIO_THRESHOLD}x baseline / {DEST_CONCENTRATION_SHARE_THRESHOLD*100:.0f}% destination share). "
            "Consistent with reports/phase3_spike_detection.md #16: this pattern typically means fraud was "
            "concentrated in a small number of transactions during an otherwise low-volume step, rather than "
            "riding on an unusually large batch of a specific transaction type."
        )

    return {
        "step": g("step"),
        "hour_of_day": hour_of_day,
        "primary_driver": primary_driver,
        "contributors": contributors,
        "baseline_comparison": baseline_comparison,
        "evidence": evidence,
    }
