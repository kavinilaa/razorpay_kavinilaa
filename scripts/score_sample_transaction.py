"""
Reproducibility wrapper - load the trained model and score one sample
transaction through the full Phase 4 risk engine (model loading + feature
generation + explanation + spike context). Cheap, read-only, no --confirm
needed - this is the "does the whole pipeline actually work end to end"
smoke check.

Usage:
    python scripts/score_sample_transaction.py
    python scripts/score_sample_transaction.py --mode HIGH_PRECISION
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

SAMPLE_TRANSACTION = {
    "step": 5,
    "type": "TRANSFER",
    "amount": 1_500_000.0,
    "oldbalanceOrg": 1_500_000.0,
    "newbalanceOrig": 0.0,
    "oldbalanceDest": 0.0,
    "newbalanceDest": 0.0,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="BALANCED", choices=["HIGH_RECALL", "BALANCED", "HIGH_PRECISION"])
    parser.add_argument("--transaction-json", default=None,
                         help="Path to a JSON file with a transaction dict to score instead of the built-in sample.")
    args = parser.parse_args()

    from risk_engine import predict_transaction_risk

    txn = SAMPLE_TRANSACTION
    if args.transaction_json:
        with open(args.transaction_json) as f:
            txn = json.load(f)

    result = predict_transaction_risk(txn, mode=args.mode)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
