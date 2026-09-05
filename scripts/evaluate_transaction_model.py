import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATS_PATH = os.path.join(ROOT, "reports", "phase3_model_stats.json")
METADATA_PATH = os.path.join(ROOT, "models", "model_metadata.json")


def main():
    if not os.path.exists(STATS_PATH) or not os.path.exists(METADATA_PATH):
        print(f"ERROR: {STATS_PATH} or {METADATA_PATH} not found. "
              "Run scripts/run_transaction_model_training.py --confirm first.")
        sys.exit(1)

    with open(STATS_PATH) as f:
        stats = json.load(f)
    with open(METADATA_PATH) as f:
        meta = json.load(f)

    selected = meta["selected_model"]
    print(f"Selected model: {selected} (criterion: {meta['selection_criterion']})")
    print(f"Train/Val/Test steps: {meta['train_steps']} / {meta['val_steps']} / {meta['test_steps']}")
    print()
    print("Validation vs. Test metrics:")
    for split in ["val_metrics", "test_metrics"]:
        m = stats["model_results"][selected][split]
        print(f"  {split:12s}: PR-AUC={m['pr_auc']:.4f}  ROC-AUC={m['roc_auc']:.4f}  "
              f"Precision={m['precision']:.4f}  Recall={m['recall']:.4f}  "
              f"TP={m['confusion_matrix']['tp']} FP={m['confusion_matrix']['fp']} FN={m['confusion_matrix']['fn']}")
    print()
    print("Operating modes (validation-derived thresholds, confirmed once on test):")
    for mode, point in meta["operating_modes_validation_thresholds"].items():
        test_eval = stats["final_test_evaluation_by_mode"][mode]
        print(f"  {mode:15s}: threshold={point}  test P={test_eval['precision']:.4f}  "
              f"test R={test_eval['recall']:.4f}  test FP={test_eval['confusion_matrix']['fp']}  "
              f"test FN={test_eval['confusion_matrix']['fn']}")

    print()
    print("See reports/phase3_model_comparison.md and reports/model_card.md for full context, "
          "including the synthetic-data caveat on these near-perfect numbers.")


if __name__ == "__main__":
    main()
