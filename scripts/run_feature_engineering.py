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
