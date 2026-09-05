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
            f"{TXN_PARQUET} not found. The Phase 2 feature pipeline must run first."
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
