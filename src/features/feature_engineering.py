"""
Phase 2 - Feature Engineering Pipeline
AI Fraud-Spike & Risk Detection System (Razorpay Buildathon, Track 02)

Reads the raw PaySim CSV (never modified) and produces two leakage-safe,
model-ready datasets:

  1. data/processed/engineered_transactions.parquet
     One row per transaction. Predictive features only use information
     available AT or BEFORE the current transaction's step. Destination
     behavioral features are computed with a strict "steps < T" expanding
     window (see `compute_destination_history`) — never full-dataset stats
     attached retroactively to early rows.

  2. data/processed/time_window_features.parquet
     One row per `step`, aggregated for spike-detection. Columns are split
     into PREDICTIVE (safe as model input) and columns prefixed `gt_`
     (ground-truth/evaluation only — built from isFraud, never a model
     input).

Design decision on chunking: the full dataset fits in ~1GB RAM (confirmed in
Phase 1) against several GB of headroom, and the leakage-safe expanding
aggregations below require a single global time ordering to be correct —
splitting into independent chunks would silently break that ordering at
chunk boundaries. So this pipeline loads the full CSV once with
memory-efficient dtypes rather than chunked streaming. If dataset size grows
by an order of magnitude, the per-destination and per-type aggregation
tables (which are small — see functions below) could be computed
incrementally from chunks instead; the row-level feature attachment is a
simple merge and would still work.

isFraud and isFlaggedFraud are carried through as reference/evaluation
columns only. No engineered PREDICTIVE feature is derived from either.
"""
import os
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_CSV = os.path.join(ROOT, "data", "raw", "PS_20174392719_1491204439457_log.csv")
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
TXN_PARQUET = os.path.join(PROCESSED_DIR, "engineered_transactions.parquet")
TIME_PARQUET = os.path.join(PROCESSED_DIR, "time_window_features.parquet")

# Fraction of steps used to freeze the "training-period" amount thresholds
# used by `high_amount_indicator`. This is NOT the final train/val/test split
# (that is a Phase 3 modeling decision) -- it only defines which slice of
# data is allowed to inform this one frozen statistic, consistent with
# Phase 1's recommendation to split by time.
TRAIN_FRACTION_FOR_FROZEN_STATS = 0.8

RAW_DTYPES = {
    "step": "int32",
    "type": "category",
    "amount": "float64",
    "nameOrig": "string",
    "oldbalanceOrg": "float64",
    "newbalanceOrig": "float64",
    "nameDest": "string",
    "oldbalanceDest": "float64",
    "newbalanceDest": "float64",
    "isFraud": "int8",
    "isFlaggedFraud": "int8",
}

# Sentinel values used instead of inf/NaN so downstream models never see
# non-finite numbers. Each is documented in reports/phase2_feature_dictionary.md.
RATIO_SENTINEL_ZERO_DENOM = 1000.0   # amount / (origin balance == 0) -> "essentially infinite" ratio, capped
NEUTRAL_RATIO_NO_HISTORY = 1.0       # "no deviation info yet" for brand-new destinations/types


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_raw() -> pd.DataFrame:
    log(f"Loading raw CSV from {RAW_CSV} ...")
    df = pd.read_csv(RAW_CSV, dtype=RAW_DTYPES)
    log(f"Loaded {len(df):,} rows.")
    return df


# ---------------------------------------------------------------------------
# 2A. Basic transaction features
# ---------------------------------------------------------------------------
def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    df["log_amount"] = np.log1p(df["amount"]).astype("float32")
    return df


# ---------------------------------------------------------------------------
# 3. Origin balance features
# ---------------------------------------------------------------------------
def add_origin_balance_features(df: pd.DataFrame) -> pd.DataFrame:
    df["balance_change_orig"] = (df["oldbalanceOrg"] - df["newbalanceOrig"]).astype("float32")
    df["balance_error_orig"] = (df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]).astype("float32")
    df["drained_to_zero"] = (df["newbalanceOrig"] == 0).astype("int8")
    df["origin_balance_zero_before"] = (df["oldbalanceOrg"] == 0).astype("int8")

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(
            df["oldbalanceOrg"].to_numpy() > 0,
            df["amount"].to_numpy() / df["oldbalanceOrg"].to_numpy(),
            RATIO_SENTINEL_ZERO_DENOM,
        )
    # cap extreme ratios (e.g. tiny non-zero balances) so the feature stays bounded
    ratio = np.minimum(ratio, RATIO_SENTINEL_ZERO_DENOM)
    df["amount_to_origin_balance_ratio"] = ratio.astype("float32")
    return df


# ---------------------------------------------------------------------------
# 4. Destination balance features
# ---------------------------------------------------------------------------
def add_destination_balance_features(df: pd.DataFrame) -> pd.DataFrame:
    df["balance_change_dest"] = (df["newbalanceDest"] - df["oldbalanceDest"]).astype("float32")
    df["balance_error_dest"] = (df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]).astype("float32")

    # PaySim limitation (confirmed in Phase 1): merchant ('M') destinations
    # never have tracked balances (always 0/0), and mule accounts receiving
    # fraudulent TRANSFERs are frequently also stuck at 0/0 because the
    # simulator does not model the immediate downstream cash-out. This flag
    # captures "destination balance uninformative" without asserting fraud.
    df["dest_balance_artifact_flag"] = (
        (df["oldbalanceDest"] == 0) & (df["newbalanceDest"] == 0) & (df["amount"] > 0)
    ).astype("int8")
    return df


# ---------------------------------------------------------------------------
# 5. Time features (simulated relative time, NOT calendar time)
# ---------------------------------------------------------------------------
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["hour_of_day"] = (df["step"] % 24).astype("int8")
    df["day_index"] = (df["step"] // 24).astype("int16")
    angle = 2 * np.pi * df["hour_of_day"].to_numpy() / 24.0
    df["hour_sin"] = np.sin(angle).astype("float32")
    df["hour_cos"] = np.cos(angle).astype("float32")
    return df


# ---------------------------------------------------------------------------
# 7. Transaction-type features
# ---------------------------------------------------------------------------
def add_type_features(df: pd.DataFrame) -> pd.DataFrame:
    df["is_transfer"] = (df["type"] == "TRANSFER").astype("int8")
    df["is_cash_out"] = (df["type"] == "CASH_OUT").astype("int8")
    df["eligible_for_fraud_model"] = df["is_transfer"] | df["is_cash_out"]
    return df


# ---------------------------------------------------------------------------
# 6. Destination behavioral features - LEAKAGE-SAFE (strictly prior steps)
# ---------------------------------------------------------------------------
def compute_destination_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    For a transaction at step T to destination D, returns historical
    behavioral stats using ONLY transactions to D at step < T:
      hist_dest_txn_count, hist_dest_unique_origin_count,
      hist_dest_total_amount, hist_dest_avg_amount, hist_dest_max_amount

    Method (vectorized, no per-row Python loop):
      1. Sort by (nameDest, step) so each destination's rows are in
         chronological step order.
      2. Flag the first time each (nameDest, nameOrig) pair is ever seen for
         that destination (chronologically) -> "new unique origin" flag.
      3. Aggregate to one row per (nameDest, step): txn_count, total_amount,
         max_amount, new_unique_origin_count for that step.
      4. Within each destination, take a cumulative sum/max over that
         per-step table and subtract/shift out the current step's own
         contribution, leaving "everything strictly before this step".
      5. Merge these per-(nameDest, step) historical values back onto every
         transaction row sharing that (nameDest, step).

    This guarantees no transaction ever sees another transaction from its
    own step or later, for the same destination.
    """
    log("Computing leakage-safe destination history features...")
    key_cols = ["nameDest", "step", "nameOrig", "amount"]
    sorted_df = df[key_cols].sort_values(["nameDest", "step"], kind="mergesort")

    new_origin_flag = (~sorted_df.duplicated(subset=["nameDest", "nameOrig"], keep="first")).astype("int32")
    sorted_df = sorted_df.assign(new_origin_flag=new_origin_flag)

    per_step = sorted_df.groupby(["nameDest", "step"], sort=False, observed=True).agg(
        txn_count=("amount", "size"),
        total_amount=("amount", "sum"),
        max_amount=("amount", "max"),
        new_unique_origin_count=("new_origin_flag", "sum"),
    ).reset_index()

    # per_step is already grouped by nameDest with step ascending within each
    # group because sorted_df was sorted that way and groupby preserves
    # first-seen order when sort=False.
    g = per_step.groupby("nameDest", sort=False)

    cum_txn = g["txn_count"].cumsum()
    per_step["hist_dest_txn_count"] = (cum_txn - per_step["txn_count"]).astype("int32")

    cum_amt = g["total_amount"].cumsum()
    per_step["hist_dest_total_amount"] = (cum_amt - per_step["total_amount"]).astype("float64")

    cum_uniq = g["new_unique_origin_count"].cumsum()
    per_step["hist_dest_unique_origin_count"] = (cum_uniq - per_step["new_unique_origin_count"]).astype("int32")

    max_shifted = g["max_amount"].shift(1)
    per_step["hist_dest_max_amount"] = (
        max_shifted.groupby(per_step["nameDest"]).cummax().fillna(0.0)
    ).astype("float64")

    per_step["hist_dest_avg_amount"] = np.where(
        per_step["hist_dest_txn_count"].to_numpy() > 0,
        per_step["hist_dest_total_amount"].to_numpy() / np.maximum(per_step["hist_dest_txn_count"].to_numpy(), 1),
        0.0,
    )

    hist_cols = [
        "nameDest", "step", "hist_dest_txn_count", "hist_dest_unique_origin_count",
        "hist_dest_total_amount", "hist_dest_avg_amount", "hist_dest_max_amount",
    ]
    out = df.merge(per_step[hist_cols], on=["nameDest", "step"], how="left")

    out["destination_is_new"] = (out["hist_dest_txn_count"] == 0).astype("int8")

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(
            out["hist_dest_avg_amount"].to_numpy() > 0,
            out["amount"].to_numpy() / out["hist_dest_avg_amount"].to_numpy(),
            NEUTRAL_RATIO_NO_HISTORY,
        )
    out["amount_to_dest_avg_ratio"] = np.minimum(ratio, RATIO_SENTINEL_ZERO_DENOM).astype("float32")

    for c in ["hist_dest_txn_count", "hist_dest_unique_origin_count"]:
        out[c] = out[c].astype("int32")
    for c in ["hist_dest_total_amount", "hist_dest_avg_amount", "hist_dest_max_amount"]:
        out[c] = out[c].astype("float32")

    log("Destination history features computed.")
    return out


# ---------------------------------------------------------------------------
# 8. Amount-behavior features - LEAKAGE-SAFE (expanding, prior steps only)
# ---------------------------------------------------------------------------
def compute_type_historical_avg_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    amount_to_type_avg_ratio: current amount vs. the EXPANDING historical
    MEAN amount for the same `type`, using only steps < current step.

    Note on mean vs. median: an exact expanding *median* at row-level over
    6.36M rows requires an order-statistic structure (balanced two-heap or
    Fenwick tree) to stay efficient -- a naive re-sort per step is
    O(steps * n log n) and does not finish in reasonable time at this scale.
    We use the expanding MEAN instead, computed exactly via cumulative
    sum/count on a tiny (type, step) aggregate table (<= 5 * 743 rows) and
    merged back. It answers the same question ("is this amount unusually
    large for this transaction type, historically?"), is exactly leakage-safe
    (strictly prior steps only, no fraud label involved), and is documented
    here as a deliberate substitution for the median named in the original
    spec.
    """
    log("Computing expanding per-type historical average amount...")
    per_type_step = df.groupby(["type", "step"], observed=True).agg(
        txn_count=("amount", "size"), total_amount=("amount", "sum")
    ).reset_index().sort_values(["type", "step"])

    g = per_type_step.groupby("type", sort=False, observed=True)
    cum_count = g["txn_count"].cumsum()
    cum_amt = g["total_amount"].cumsum()
    per_type_step["hist_type_txn_count"] = (cum_count - per_type_step["txn_count"]).astype("int64")
    per_type_step["hist_type_total_amount"] = (cum_amt - per_type_step["total_amount"]).astype("float64")
    per_type_step["hist_type_avg_amount"] = np.where(
        per_type_step["hist_type_txn_count"].to_numpy() > 0,
        per_type_step["hist_type_total_amount"].to_numpy()
        / np.maximum(per_type_step["hist_type_txn_count"].to_numpy(), 1),
        np.nan,
    )

    merge_cols = ["type", "step", "hist_type_avg_amount"]
    out = df.merge(per_type_step[merge_cols], on=["type", "step"], how="left")

    hist_avg = out["hist_type_avg_amount"].to_numpy()
    valid = ~np.isnan(hist_avg) & (hist_avg > 0)
    ratio = np.where(valid, out["amount"].to_numpy() / np.where(valid, hist_avg, 1), NEUTRAL_RATIO_NO_HISTORY)
    out["amount_to_type_avg_ratio"] = np.minimum(ratio, RATIO_SENTINEL_ZERO_DENOM).astype("float32")
    out = out.drop(columns=["hist_type_avg_amount"])
    log("Type-level historical average ratio computed.")
    return out


def compute_high_amount_indicator(df: pd.DataFrame) -> pd.DataFrame:
    """
    high_amount_indicator: 1 if amount exceeds the 95th percentile of amount
    for that `type`, where the 95th percentile is a FROZEN statistic computed
    only from the training-period slice (first TRAIN_FRACTION_FOR_FROZEN_STATS
    of steps). This mirrors how a deployed model would use a threshold fit
    once at training time and applied unchanged going forward -- it never
    looks at amounts from later steps, and never uses isFraud.
    """
    log("Computing frozen training-period high-amount thresholds...")
    max_step = int(df["step"].max())
    cutoff_step = int(max_step * TRAIN_FRACTION_FOR_FROZEN_STATS)

    train_slice = df.loc[df["step"] <= cutoff_step]
    thresholds = train_slice.groupby("type", observed=True)["amount"].quantile(0.95)
    log(f"Frozen thresholds (fit on steps <= {cutoff_step}, {TRAIN_FRACTION_FOR_FROZEN_STATS:.0%} of range): "
        f"{thresholds.to_dict()}")

    threshold_map = df["type"].map(thresholds).astype("float64")
    df["high_amount_indicator"] = (df["amount"] > threshold_map).astype("int8")
    return df, cutoff_step, thresholds


# ---------------------------------------------------------------------------
# Orchestration - transaction-level features
# ---------------------------------------------------------------------------
def build_transaction_features(df: pd.DataFrame):
    df = add_basic_features(df)
    df = add_origin_balance_features(df)
    df = add_destination_balance_features(df)
    df = add_time_features(df)
    df = add_type_features(df)
    df = compute_destination_history(df)
    df = compute_type_historical_avg_ratio(df)
    df, cutoff_step, thresholds = compute_high_amount_indicator(df)
    return df, cutoff_step, thresholds


# ---------------------------------------------------------------------------
# 12-13. Time-window (spike-detection) features
# ---------------------------------------------------------------------------
def build_time_window_features(df: pd.DataFrame) -> pd.DataFrame:
    log("Building per-step time-window features for spike detection...")
    agg = df.groupby("step").agg(
        transaction_count=("amount", "size"),
        transfer_count=("is_transfer", "sum"),
        cash_out_count=("is_cash_out", "sum"),
        total_amount=("amount", "sum"),
        average_amount=("amount", "mean"),
        median_amount=("amount", "median"),
        high_amount_count=("high_amount_indicator", "sum"),
        gt_fraud_count=("isFraud", "sum"),
    ).reset_index()

    agg["gt_fraud_rate"] = agg["gt_fraud_count"] / agg["transaction_count"]

    agg["hour_of_day"] = (agg["step"] % 24).astype("int8")
    agg["day_index"] = (agg["step"] // 24).astype("int16")
    agg = agg.sort_values("step").reset_index(drop=True)

    # --- leakage-safe historical baseline by hour-of-day (predictive) ---
    g = agg.groupby("hour_of_day")
    shifted_txn = g["transaction_count"].shift(1)
    agg["_shifted_txn"] = shifted_txn
    agg["hist_avg_txn_count_same_hour"] = (
        agg.groupby("hour_of_day")["_shifted_txn"].expanding().mean().reset_index(level=0, drop=True)
    )

    shifted_amt = g["average_amount"].shift(1)
    agg["_shifted_amt"] = shifted_amt
    agg["hist_avg_amount_same_hour"] = (
        agg.groupby("hour_of_day")["_shifted_amt"].expanding().mean().reset_index(level=0, drop=True)
    )

    # global expanding fallback (all hours) for the very first cycle where a
    # given hour_of_day has no prior occurrence yet
    agg["_global_expanding_txn"] = agg["transaction_count"].shift(1).expanding().mean()
    agg["_global_expanding_amt"] = agg["average_amount"].shift(1).expanding().mean()
    agg["hist_avg_txn_count_same_hour"] = agg["hist_avg_txn_count_same_hour"].fillna(agg["_global_expanding_txn"])
    agg["hist_avg_amount_same_hour"] = agg["hist_avg_amount_same_hour"].fillna(agg["_global_expanding_amt"])
    # first-ever step: no history at all -> neutral fallback equal to its own value (ratio = 1)
    agg["hist_avg_txn_count_same_hour"] = agg["hist_avg_txn_count_same_hour"].fillna(agg["transaction_count"])
    agg["hist_avg_amount_same_hour"] = agg["hist_avg_amount_same_hour"].fillna(agg["average_amount"])

    agg["txn_count_vs_hour_baseline_ratio"] = (
        agg["transaction_count"] / agg["hist_avg_txn_count_same_hour"].replace(0, np.nan)
    ).fillna(NEUTRAL_RATIO_NO_HISTORY).astype("float32")
    agg["amount_vs_hour_baseline_ratio"] = (
        agg["average_amount"] / agg["hist_avg_amount_same_hour"].replace(0, np.nan)
    ).fillna(NEUTRAL_RATIO_NO_HISTORY).astype("float32")

    # --- ground-truth-only historical fraud-rate baseline (eval only) ---
    shifted_fraud_rate = g["gt_fraud_rate"].shift(1)
    agg["_shifted_fraud_rate"] = shifted_fraud_rate
    agg["gt_hist_avg_fraud_rate_same_hour"] = (
        agg.groupby("hour_of_day")["_shifted_fraud_rate"].expanding().mean().reset_index(level=0, drop=True)
    )

    agg = agg.drop(columns=[c for c in agg.columns if c.startswith("_")])

    dtype_map = {
        "transaction_count": "int32", "transfer_count": "int32", "cash_out_count": "int32",
        "total_amount": "float64", "average_amount": "float64", "median_amount": "float64",
        "high_amount_count": "int32", "gt_fraud_count": "int32", "gt_fraud_rate": "float32",
        "hist_avg_txn_count_same_hour": "float32", "hist_avg_amount_same_hour": "float32",
        "gt_hist_avg_fraud_rate_same_hour": "float32",
    }
    for c, t in dtype_map.items():
        agg[c] = agg[c].astype(t)

    log(f"Time-window feature table built: {len(agg)} rows (one per step).")
    return agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_pipeline():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    t0 = time.time()

    df = load_raw()
    n_raw_rows = len(df)

    df, cutoff_step, thresholds = build_transaction_features(df)

    log("Writing engineered_transactions.parquet ...")
    df.to_parquet(TXN_PARQUET, engine="pyarrow", index=False)
    log(f"Wrote {TXN_PARQUET} ({os.path.getsize(TXN_PARQUET)/1024**2:.1f} MB)")

    time_window_df = build_time_window_features(df)
    time_window_df.to_parquet(TIME_PARQUET, engine="pyarrow", index=False)
    log(f"Wrote {TIME_PARQUET} ({os.path.getsize(TIME_PARQUET)/1024**2:.1f} MB)")

    log(f"Pipeline complete in {time.time()-t0:.1f}s. Rows in: {n_raw_rows:,}, rows out: {len(df):,}")
    return df, time_window_df, cutoff_step, thresholds


if __name__ == "__main__":
    run_pipeline()
