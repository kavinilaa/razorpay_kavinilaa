"""
Reproducibility wrapper - Phase 3, Part A transaction-model training.
AI Fraud-Spike & Risk Detection System (Razorpay Buildathon, Track 02)

Runs src/models/train_transaction_models.py end to end (trains Logistic
Regression, LightGBM, and XGBoost on the engineered dataset). Requires
data/processed/engineered_transactions.parquet to already exist (see
scripts/run_feature_engineering.py). Guarded behind --confirm since this
overwrites the models/ artifacts already committed in this repo and takes
noticeable time.

Usage:
    python scripts/run_transaction_model_training.py --confirm

Produces:
    models/logistic_regression.joblib, models/lightgbm.joblib, models/xgboost.joblib
    models/model_metadata.json, models/predictions_cache.npz
    reports/phase3_model_stats.json
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm", action="store_true",
                         help="Actually retrain and overwrite models/*.joblib.")
    args = parser.parse_args()

    txn_parquet = os.path.join(ROOT, "data", "processed", "engineered_transactions.parquet")
    if not os.path.exists(txn_parquet):
        print(f"ERROR: {txn_parquet} not found. Run scripts/run_feature_engineering.py --confirm first.")
        sys.exit(1)

    if not args.confirm:
        print("This retrains all three models and OVERWRITES the committed models/*.joblib files.")
        print("If you just want to inspect existing results, use scripts/evaluate_transaction_model.py instead "
              "(reads the already-saved metrics/predictions - no retraining).")
        print("Re-run with --confirm to proceed.")
        sys.exit(0)

    from models.train_transaction_models import main as train_main
    train_main()


if __name__ == "__main__":
    main()
