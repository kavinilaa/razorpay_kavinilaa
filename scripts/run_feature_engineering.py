"""
Reproducibility wrapper - Phase 2 feature engineering.
AI Fraud-Spike & Risk Detection System (Razorpay Buildathon, Track 02)

Runs src/features/feature_engineering.py end to end. This reads the full
6.36M-row raw PaySim CSV and takes on the order of minutes - it is guarded
behind --confirm so it is never triggered by accident (e.g. by a test
runner or CI job that imports this file).

Usage:
    python scripts/run_feature_engineering.py --confirm

Produces:
    data/processed/engineered_transactions.parquet
    data/processed/time_window_features.parquet
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm", action="store_true",
                         help="Actually run the expensive pipeline (reads the full raw CSV).")
    args = parser.parse_args()

    raw_csv = os.path.join(ROOT, "data", "raw", "PS_20174392719_1491204439457_log.csv")
    if not os.path.exists(raw_csv):
        print(f"ERROR: raw dataset not found at {raw_csv}. Feature engineering cannot run.")
        sys.exit(1)

    if not args.confirm:
        print("This reprocesses the full 6.36M-row raw CSV (several minutes). "
              "Re-run with --confirm to proceed.")
        print(f"Raw CSV found at: {raw_csv}")
        sys.exit(0)

    from features.feature_engineering import run_pipeline
    run_pipeline()


if __name__ == "__main__":
    main()
