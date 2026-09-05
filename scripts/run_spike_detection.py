import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm", action="store_true",
                         help="Actually re-run spike detection and overwrite its artifacts.")
    args = parser.parse_args()

    required = [
        os.path.join(ROOT, "data", "processed", "time_window_features.parquet"),
        os.path.join(ROOT, "models", "model_metadata.json"),
    ]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        print("ERROR: missing prerequisite artifact(s):")
        for p in missing:
            print(f"  - {p}")
        print("Run scripts/run_feature_engineering.py and scripts/run_transaction_model_training.py first.")
        sys.exit(1)

    if not args.confirm:
        print("This re-scores every eligible transaction with the trained model, refits Isolation Forest, "
              "and OVERWRITES models/isolation_forest.joblib and reports/phase3_spike_stats.json.")
        print("Re-run with --confirm to proceed.")
        sys.exit(0)

    from models.spike_detection import main as spike_main
    spike_main()


if __name__ == "__main__":
    main()
