"""
Phase 4 - Build frozen feature-reference statistics for single-transaction inference.
AI Fraud-Spike & Risk Detection System (Razorpay Buildathon, Track 02)

The Phase 2 feature pipeline (src/features/feature_engineering.py) computes
several "frozen at training time" statistics as part of building the full
6.36M-row batch dataset:

  - `high_amount_indicator` per-`type` 95th-percentile amount threshold,
    frozen from the first 80% of steps (steps <= 594) -- see
    `compute_high_amount_indicator()`.
  - the expanding per-`type` historical average amount used by
    `amount_to_type_avg_ratio` (computed row-by-row over strictly-prior
    steps in the batch pipeline).

Neither of these was ever persisted to its own file -- they only existed as
in-memory values while `feature_engineering.py` ran. The Phase 4 risk engine
(`src/risk_engine/`) needs to score ONE transaction at a time, outside that
batch pipeline, so it needs these same frozen statistics available as a
small, loadable artifact instead of recomputing them from the 6.36M-row
dataset on every prediction.

This script recomputes them from the existing
`data/processed/engineered_transactions.parquet` (already produced by the
Phase 2 pipeline -- this script does NOT re-read the raw CSV or redo feature
engineering) and writes `models/feature_reference_stats.json`.

Two frozen artifacts are produced, using two different (both real, both
already-established) cutoffs:

  1. `high_amount_thresholds_by_type` -- per-`type` 95th percentile of
     `amount`, computed on steps <= 594 (80% of the 743-step range). This is
     an EXACT reproduction of the cutoff and method already used by
     `compute_high_amount_indicator()` in the Phase 2 pipeline, so a
     single-transaction prediction's `high_amount_indicator` feature is
     computed identically to how it was computed for every training row.

  2. `type_avg_amount_reference` -- per-`type` mean `amount`, computed on
     steps <= 520 (the END of the Phase 3 classifier's TRAIN split -- see
     `TRAIN_STEPS` in `src/models/train_transaction_models.py`). This is a
     NEW derived reference (the batch pipeline computed an expanding,
     row-by-row version of this instead of a single frozen snapshot). It is
     used by the risk engine as a practical stand-in for "the historical
     average amount for this type, as of the point the model stopped
     learning" -- consistent with the model's own train/val/test boundary,
     computed from real data, not fabricated. This is a documented
     simplification for standalone single-transaction inference: the batch
     pipeline's per-row expanding average is not reproducible outside the
     full dataset.

Run:
    python scripts/build_feature_reference_stats.py
"""
import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TXN_PARQUET = os.path.join(ROOT, "data", "processed", "engineered_transactions.parquet")
MODELS_DIR = os.path.join(ROOT, "models")

HIGH_AMOUNT_CUTOFF_STEP = 594  # matches feature_engineering.py TRAIN_FRACTION_FOR_FROZEN_STATS=0.8 * 743
TYPE_AVG_CUTOFF_STEP = 520     # matches train_transaction_models.py TRAIN_STEPS[1]


def main():
    if not os.path.exists(TXN_PARQUET):
        raise FileNotFoundError(
            f"{TXN_PARQUET} not found. Run the Phase 2 feature pipeline first: "
            "python src/features/feature_engineering.py"
        )

    df = pd.read_parquet(TXN_PARQUET, columns=["type", "amount", "step"])

    high_amount_slice = df.loc[df["step"] <= HIGH_AMOUNT_CUTOFF_STEP]
    high_amount_thresholds = high_amount_slice.groupby("type", observed=True)["amount"].quantile(0.95)

    type_avg_slice = df.loc[df["step"] <= TYPE_AVG_CUTOFF_STEP]
    type_avg_amount = type_avg_slice.groupby("type", observed=True)["amount"].mean()

    stats = {
        "source": "computed by scripts/build_feature_reference_stats.py from data/processed/engineered_transactions.parquet",
        "high_amount_thresholds_by_type": {
            "cutoff_step": HIGH_AMOUNT_CUTOFF_STEP,
            "method": "95th percentile of amount per type, steps <= cutoff_step (matches feature_engineering.py compute_high_amount_indicator)",
            "values": {str(k): float(v) for k, v in high_amount_thresholds.items()},
        },
        "type_avg_amount_reference": {
            "cutoff_step": TYPE_AVG_CUTOFF_STEP,
            "method": "mean amount per type, steps <= cutoff_step (matches classifier TRAIN_STEPS end; a frozen "
                      "snapshot standing in for the batch pipeline's per-row expanding average, for standalone "
                      "single-transaction inference only)",
            "values": {str(k): float(v) for k, v in type_avg_amount.items()},
        },
    }

    out_path = os.path.join(MODELS_DIR, "feature_reference_stats.json")
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Wrote {out_path}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
