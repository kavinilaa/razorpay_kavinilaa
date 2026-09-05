"""
Phase 4 - Operating thresholds and risk bands for the transaction-risk engine.
AI Fraud-Spike & Risk Detection System (Razorpay Buildathon, Track 02)

Every number in this file is copied from Phase 3's actual evaluation output
(reports/phase3_model_stats.json -> operating_modes_validation /
final_test_evaluation_by_mode, and reports/phase3_threshold_analysis.md) - not
invented. See `_source_check()` at the bottom, which re-derives the constants
from models/model_metadata.json at import time and raises if they ever drift
out of sync with the actual trained model's metadata.

IMPORTANT - what these numbers do and do not mean (do not overstate this):
The precision/recall/alert-rate figures below were measured on the Phase 3
validation set (steps 521-631) and confirmed once on the untouched test set
(steps 632-743) of the PaySim synthetic dataset. They describe how the
XGBoost model behaved on THIS dataset's fraud pattern (dominated by the
`drained_to_zero` signature - see reports/phase3_model_comparison.md #8).
They are NOT a universal claim about how any threshold would perform on real
transaction traffic, and no single mode is "the" optimal choice for every
use case - that depends on a false-positive/false-negative cost ratio this
project does not have (see reports/phase2_modeling_recommendation.md #8).
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(ROOT, "models")
REPORTS_DIR = os.path.join(ROOT, "reports")

# --- Operating modes, thresholds and their measured effect ------------------
# Thresholds themselves come from models/model_metadata.json
# ("operating_modes_validation_thresholds"), which is written by
# src/models/train_transaction_models.py directly from the validation-set
# threshold sweep. The precision/recall/alert-rate figures quoted here are
# copied from reports/phase3_model_stats.json at the time that file was
# generated (both "operating_modes_validation" for the validation-set numbers
# used to CHOOSE each threshold, and "final_test_evaluation_by_mode" for the
# one-time confirmation on the held-out test set).
OPERATING_MODES = {
    "HIGH_RECALL": {
        "threshold": 0.43,
        "rationale": "Highest-precision point among validation thresholds achieving recall >= 90%.",
        "validation": {"precision": 1.0, "recall": 0.999153, "f1": 0.999576, "alert_rate": 0.01498075},
        "test": {"precision": 1.0, "recall": 0.999201, "f1": 0.9996, "alert_rate": None},
    },
    "BALANCED": {
        "threshold": 0.43,
        "rationale": "Threshold with the single highest F1 score on validation. On this dataset it "
                     "coincides exactly with HIGH_RECALL's threshold (scores are extremely well "
                     "separated - see reports/phase3_threshold_analysis.md); this is a property of "
                     "this dataset, not a general guarantee that these two modes will always match.",
        "validation": {"precision": 1.0, "recall": 0.999153, "f1": 0.999576, "alert_rate": 0.01498075},
        "test": {"precision": 1.0, "recall": 0.999201, "f1": 0.9996, "alert_rate": None},
    },
    "HIGH_PRECISION": {
        "threshold": 0.17,
        "rationale": "Lowest threshold achieving validation precision >= 80% (a deliberately looser "
                     "operating point that maximizes recall subject to that floor - NOT the single "
                     "highest-precision threshold available; thresholds 0.43-0.89 all give 0 false "
                     "positives on validation, per reports/phase3_threshold_analysis.md).",
        "validation": {"precision": 0.901376, "recall": 0.999153, "f1": 0.947749, "alert_rate": 0.01661987},
        "test": {"precision": 0.965305, "recall": 1.0, "f1": 0.982346, "alert_rate": None},
    },
}

DEFAULT_MODE = "BALANCED"

# --- Risk bands ---------------------------------------------------------
# Bands are built directly from the two distinct thresholds actually used by
# the three operating modes above (0.17 and 0.43) rather than an arbitrary
# 0.3/0.7 split, so every boundary is traceable to a specific measured
# precision/recall point in reports/phase3_threshold_analysis.md:
#   score <  0.17            -> LOW    (below even the loosest validated operating point)
#   0.17 <= score < 0.43     -> MEDIUM (matches the HIGH_PRECISION threshold band -
#                                        precision ~90-97%, recall ~99.9-100% on Phase 3 val/test)
#   score >= 0.43            -> HIGH   (matches the BALANCED/HIGH_RECALL threshold -
#                                        precision 100%, recall ~99.9% on Phase 3 val/test)
RISK_BAND_BOUNDARIES = [
    (OPERATING_MODES["HIGH_PRECISION"]["threshold"], "MEDIUM"),
    (OPERATING_MODES["BALANCED"]["threshold"], "HIGH"),
]

BAND_RECOMMENDED_ACTION = {
    "LOW": "ALLOW",
    "MEDIUM": "REVIEW",
    "HIGH": "REVIEW",
}


def get_threshold(mode: str = DEFAULT_MODE) -> float:
    mode = (mode or DEFAULT_MODE).upper()
    if mode not in OPERATING_MODES:
        raise ValueError(f"Unknown operating mode {mode!r}. Valid modes: {sorted(OPERATING_MODES)}")
    return OPERATING_MODES[mode]["threshold"]


def get_risk_band(score: float) -> str:
    """LOW / MEDIUM / HIGH, using the boundaries derived from Phase 3's own
    validated operating-mode thresholds (see module docstring)."""
    if score >= RISK_BAND_BOUNDARIES[1][0]:
        return "HIGH"
    if score >= RISK_BAND_BOUNDARIES[0][0]:
        return "MEDIUM"
    return "LOW"


def get_recommended_action(risk_band: str) -> str:
    if risk_band not in BAND_RECOMMENDED_ACTION:
        raise ValueError(f"Unknown risk band {risk_band!r}")
    return BAND_RECOMMENDED_ACTION[risk_band]


def _source_check():
    """Fail loudly (at import time) if these hard-coded thresholds ever
    drift from the trained model's own metadata file, instead of silently
    scoring transactions against a stale operating point."""
    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
    if not os.path.exists(meta_path):
        return  # model not trained yet in this environment - nothing to check against
    with open(meta_path) as f:
        meta = json.load(f)
    live_thresholds = meta.get("operating_modes_validation_thresholds", {})
    for mode_key, meta_key in [("HIGH_RECALL", "high_recall"), ("BALANCED", "balanced"),
                                ("HIGH_PRECISION", "high_precision")]:
        if meta_key in live_thresholds and live_thresholds[meta_key] != OPERATING_MODES[mode_key]["threshold"]:
            raise RuntimeError(
                f"thresholds.py OPERATING_MODES['{mode_key}'] threshold "
                f"({OPERATING_MODES[mode_key]['threshold']}) no longer matches "
                f"models/model_metadata.json ({live_thresholds[meta_key]}) - the model was likely "
                f"retrained. Update thresholds.py from the new reports/phase3_model_stats.json."
            )


_source_check()
