"""
Phase 4 - Production-style transaction risk engine.
AI Fraud-Spike & Risk Detection System (Razorpay Buildathon, Track 02)

Combines:
  A. the Phase 3 transaction-level XGBoost risk model
  B. the Phase 3 spike-detection layer (read-only lookup of a step's spike
     status, produced offline by src/models/spike_detection.py - this module
     does not retrain or re-run spike detection)
  C. explainability (src/risk_engine/explanations.py, real SHAP values)
  D. the Phase 3 operating thresholds (src/risk_engine/thresholds.py)

Public entry point: `predict_transaction_risk(transaction, mode="BALANCED")`.

Failure handling: every external dependency (model file, feature-reference
artifact, spike-detection artifact) is loaded defensively. No failure mode
described in reports/phase4_failure_recovery.md is allowed to silently
produce a fabricated score - see `_degraded_response()`.
"""
import json
import logging
import os

import joblib
import numpy as np
import pandas as pd

from . import thresholds
from .explanations import explain_prediction
from .schemas import (
    FeatureGenerationError,
    InvalidTransactionError,
    ModelUnavailableError,
    SpikeDetectorUnavailableError,
    validate_transaction,
)

logger = logging.getLogger("risk_engine")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(ROOT, "models")
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")

MODEL_PATH = os.path.join(MODELS_DIR, "xgboost.joblib")
MODEL_METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")
FEATURE_REFERENCE_STATS_PATH = os.path.join(MODELS_DIR, "feature_reference_stats.json")
EXTENDED_TIME_WINDOW_PATH = os.path.join(PROCESSED_DIR, "time_window_features_extended.parquet")

RATIO_SENTINEL_ZERO_DENOM = 1000.0   # matches src/features/feature_engineering.py
NEUTRAL_RATIO_NO_HISTORY = 1.0       # matches src/features/feature_engineering.py

# ---------------------------------------------------------------------------
# Cached artifact loaders. Caching is per explicit path so tests can point at
# fixture files without disturbing the real, default-path cache.
# ---------------------------------------------------------------------------
_MODEL_CACHE = {}
_METADATA_CACHE = {}
_REFERENCE_STATS_CACHE = {}
_SPIKE_TABLE_CACHE = {}


def clear_caches():
    """Test helper: drop all cached artifacts so the next call re-loads from disk."""
    _MODEL_CACHE.clear()
    _METADATA_CACHE.clear()
    _REFERENCE_STATS_CACHE.clear()
    _SPIKE_TABLE_CACHE.clear()


def load_model(path: str = None):
    # NOTE: `path` intentionally resolves module-level MODEL_PATH at CALL time
    # (not as a bound default at def-time) so tests can monkeypatch
    # risk_engine.MODEL_PATH and have every caller pick up the new value.
    if path is None:
        path = MODEL_PATH
    if path in _MODEL_CACHE:
        return _MODEL_CACHE[path]
    if not os.path.exists(path):
        raise ModelUnavailableError(f"model file not found at {path}")
    try:
        model = joblib.load(path)
    except Exception as e:  # corrupt / unreadable joblib file
        raise ModelUnavailableError(f"model file at {path} could not be loaded ({type(e).__name__}: {e})") from e
    _MODEL_CACHE[path] = model
    return model


def load_model_metadata(path: str = None) -> dict:
    if path is None:
        path = MODEL_METADATA_PATH
    if path in _METADATA_CACHE:
        return _METADATA_CACHE[path]
    if not os.path.exists(path):
        raise ModelUnavailableError(f"model metadata not found at {path}")
    try:
        with open(path) as f:
            meta = json.load(f)
    except Exception as e:
        raise ModelUnavailableError(f"model metadata at {path} could not be parsed ({type(e).__name__}: {e})") from e
    _METADATA_CACHE[path] = meta
    return meta


def load_feature_reference_stats(path: str = None) -> dict:
    if path is None:
        path = FEATURE_REFERENCE_STATS_PATH
    if path in _REFERENCE_STATS_CACHE:
        return _REFERENCE_STATS_CACHE[path]
    if not os.path.exists(path):
        raise FeatureGenerationError(
            f"feature reference stats artifact not found at {path} - run "
            "scripts/build_feature_reference_stats.py first"
        )
    try:
        with open(path) as f:
            stats = json.load(f)
    except Exception as e:
        raise FeatureGenerationError(f"feature reference stats at {path} could not be parsed ({type(e).__name__}: {e})") from e
    _REFERENCE_STATS_CACHE[path] = stats
    return stats


def load_spike_table(path: str = None) -> pd.DataFrame:
    if path is None:
        path = EXTENDED_TIME_WINDOW_PATH
    if path in _SPIKE_TABLE_CACHE:
        return _SPIKE_TABLE_CACHE[path]
    if not os.path.exists(path):
        raise SpikeDetectorUnavailableError(
            f"spike-detection table not found at {path} - run src/models/spike_detection.py first"
        )
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        raise SpikeDetectorUnavailableError(f"spike-detection table at {path} could not be read ({type(e).__name__}: {e})") from e
    _SPIKE_TABLE_CACHE[path] = df
    return df


# ---------------------------------------------------------------------------
# Feature generation for a SINGLE transaction.
#
# This deliberately mirrors src/features/feature_engineering.py's per-row
# formulas exactly (see that file for the batch/leakage-safe version). Two
# documented simplifications are necessary because this function scores one
# transaction with no access to the full 6.36M-row dataset:
#
#   1. Destination history (hist_dest_*) - the batch pipeline computes this
#      from a strict "all transactions to this destination at steps < T"
#      window. A standalone scoring function has no such window unless the
#      caller supplies it. If the caller does not pass hist_dest_* fields,
#      this function treats the destination as brand-new (all zeros,
#      destination_is_new=1) - a conservative, explicitly documented default,
#      NOT a fabricated lookup.
#   2. `amount_to_type_avg_ratio` and `high_amount_indicator` use the frozen
#      reference statistics in models/feature_reference_stats.json (built by
#      scripts/build_feature_reference_stats.py from the real training data)
#      instead of the batch pipeline's per-row expanding average - see that
#      script's docstring for the exact cutoffs used.
# ---------------------------------------------------------------------------
def build_features(cleaned: dict, feature_reference_stats: dict = None) -> pd.DataFrame:
    if feature_reference_stats is None:
        feature_reference_stats = load_feature_reference_stats()

    amount = cleaned["amount"]
    old_orig = cleaned["oldbalanceOrg"]
    new_orig = cleaned["newbalanceOrig"]
    old_dest = cleaned["oldbalanceDest"]
    new_dest = cleaned["newbalanceDest"]
    step = cleaned["step"]
    txn_type = cleaned["type"]

    hist_dest_txn_count = cleaned["hist_dest_txn_count"]
    hist_dest_unique_origin_count = cleaned["hist_dest_unique_origin_count"]
    hist_dest_total_amount = cleaned["hist_dest_total_amount"]
    hist_dest_avg_amount = cleaned["hist_dest_avg_amount"]
    hist_dest_max_amount = cleaned["hist_dest_max_amount"]

    log_amount = float(np.log1p(amount))
    balance_change_orig = old_orig - new_orig
    balance_error_orig = old_orig - amount - new_orig
    drained_to_zero = int(new_orig == 0)
    origin_balance_zero_before = int(old_orig == 0)
    amount_to_origin_balance_ratio = min(
        (amount / old_orig) if old_orig > 0 else RATIO_SENTINEL_ZERO_DENOM,
        RATIO_SENTINEL_ZERO_DENOM,
    )

    balance_change_dest = new_dest - old_dest
    balance_error_dest = old_dest + amount - new_dest
    dest_balance_artifact_flag = int(old_dest == 0 and new_dest == 0 and amount > 0)

    hour_of_day = step % 24
    angle = 2 * np.pi * hour_of_day / 24.0
    hour_sin = float(np.sin(angle))
    hour_cos = float(np.cos(angle))

    is_transfer = int(txn_type == "TRANSFER")

    destination_is_new = int(hist_dest_txn_count == 0)
    amount_to_dest_avg_ratio = min(
        (amount / hist_dest_avg_amount) if hist_dest_avg_amount > 0 else NEUTRAL_RATIO_NO_HISTORY,
        RATIO_SENTINEL_ZERO_DENOM,
    )

    type_avg_lookup = feature_reference_stats.get("type_avg_amount_reference", {}).get("values", {})
    type_avg = type_avg_lookup.get(txn_type)
    amount_to_type_avg_ratio = min(
        (amount / type_avg) if type_avg else NEUTRAL_RATIO_NO_HISTORY,
        RATIO_SENTINEL_ZERO_DENOM,
    )

    high_amount_thresholds = feature_reference_stats.get("high_amount_thresholds_by_type", {}).get("values", {})
    high_amount_threshold = high_amount_thresholds.get(txn_type)
    high_amount_indicator = int(high_amount_threshold is not None and amount > high_amount_threshold)

    row = {
        "amount": amount,
        "log_amount": log_amount,
        "balance_change_orig": balance_change_orig,
        "balance_error_orig": balance_error_orig,
        "drained_to_zero": drained_to_zero,
        "origin_balance_zero_before": origin_balance_zero_before,
        "amount_to_origin_balance_ratio": amount_to_origin_balance_ratio,
        "balance_change_dest": balance_change_dest,
        "balance_error_dest": balance_error_dest,
        "dest_balance_artifact_flag": dest_balance_artifact_flag,
        "hour_of_day": hour_of_day,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "is_transfer": is_transfer,
        "hist_dest_txn_count": hist_dest_txn_count,
        "hist_dest_unique_origin_count": hist_dest_unique_origin_count,
        "hist_dest_total_amount": hist_dest_total_amount,
        "hist_dest_avg_amount": hist_dest_avg_amount,
        "hist_dest_max_amount": hist_dest_max_amount,
        "destination_is_new": destination_is_new,
        "amount_to_dest_avg_ratio": amount_to_dest_avg_ratio,
        "amount_to_type_avg_ratio": amount_to_type_avg_ratio,
        "high_amount_indicator": high_amount_indicator,
    }

    try:
        metadata = load_model_metadata()
        feature_list = metadata["feature_list"]
    except ModelUnavailableError:
        feature_list = list(row.keys())  # fall back to natural order; validated below anyway

    missing = [f for f in feature_list if f not in row]
    if missing:
        raise FeatureGenerationError(f"required feature(s) could not be generated: {missing}")

    X_row = pd.DataFrame([{f: row[f] for f in feature_list}])

    non_finite = [c for c in X_row.columns if not np.isfinite(X_row[c].astype(float)).all()]
    if non_finite:
        raise FeatureGenerationError(f"feature(s) contained NaN/Infinity after generation: {non_finite}")

    return X_row


# ---------------------------------------------------------------------------
# Spike context lookup (read-only; does not run Isolation Forest live)
# ---------------------------------------------------------------------------
def get_step_spike_context(step: int) -> dict:
    df = load_spike_table()  # raises SpikeDetectorUnavailableError if missing/corrupt
    match = df.loc[df["step"] == step]
    if match.empty:
        return {
            "available": True,
            "is_spike": False,
            "note": (
                f"step {step} is outside the historical spike-detection analysis window "
                f"(steps {int(df['step'].min())}-{int(df['step'].max())}); no spike context available for it."
            ),
        }
    row = match.iloc[0]
    stat_flag = bool(row.get("stat_method_pred", 0))
    iso_flag = bool(row.get("iso_method_pred", 0))
    return {
        "available": True,
        "is_spike": stat_flag or iso_flag,  # hybrid (OR) decision - see reports/phase3_spike_detection.md decision record
        "statistical_method_flag": stat_flag,
        "isolation_forest_flag": iso_flag,
        "txn_count_vs_hour_baseline_ratio": float(row.get("txn_count_vs_hour_baseline_ratio", np.nan)),
        "predicted_high_risk_count_vs_hour_baseline_ratio": float(
            row.get("predicted_high_risk_count_vs_hour_baseline_ratio", np.nan)
        ),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def _degraded_response(mode, message) -> dict:
    logger.warning("risk_engine degraded: %s", message)
    return {
        "status": "degraded",
        "risk_score": None,
        "risk_band": None,
        "recommended_action": "MANUAL_REVIEW",
        "mode": mode,
        "threshold_used": None,
        "top_reasons": [],
        "explanation": None,
        "spike_context": None,
        "message": message,
        "fallback": "MANUAL_REVIEW",
    }


def predict_transaction_risk(transaction: dict, mode: str = thresholds.DEFAULT_MODE) -> dict:
    """API-ready single-transaction risk scoring function.

    1. Validates input.
    2. Generates the same 23 features used during Phase 3 training.
    3. Loads the saved XGBoost model.
    4. Generates a risk score.
    5. Assigns a risk band + recommended action from the Phase 3 threshold analysis.
    6. Returns a SHAP-based explanation and (best-effort) spike context.

    Never raises for expected failure modes - see reports/phase4_failure_recovery.md.
    Returns a dict; `status` is one of "ok" / "invalid_input" / "degraded".
    """
    mode = (mode or thresholds.DEFAULT_MODE).upper()

    try:
        cleaned = validate_transaction(transaction)
    except InvalidTransactionError as e:
        return {
            "status": "invalid_input",
            "risk_score": None,
            "risk_band": None,
            "recommended_action": None,
            "mode": mode,
            "threshold_used": None,
            "top_reasons": [],
            "explanation": None,
            "spike_context": None,
            "message": "Transaction failed validation: " + "; ".join(e.errors),
            "fallback": "MANUAL_REVIEW",
        }

    try:
        model = load_model()
    except ModelUnavailableError as e:
        return _degraded_response(mode, f"Transaction risk model unavailable: {e}")

    try:
        X_row = build_features(cleaned)
    except FeatureGenerationError as e:
        return _degraded_response(mode, f"Feature generation failed: {e}")

    try:
        threshold = thresholds.get_threshold(mode)
    except ValueError as e:
        return _degraded_response(mode, str(e))

    try:
        risk_score = float(model.predict_proba(X_row)[:, 1][0])
    except Exception as e:
        return _degraded_response(mode, f"Model inference failed ({type(e).__name__}): {e}")

    if not np.isfinite(risk_score):
        return _degraded_response(mode, "Model produced a non-finite risk score (NaN/Infinity); refusing to report it.")

    risk_band = thresholds.get_risk_band(risk_score)
    recommended_action = thresholds.get_recommended_action(risk_band)

    explanation = None
    top_reasons = []
    try:
        explanation = explain_prediction(model, X_row, risk_score, risk_band)
        top_reasons = [c["phrase"] for c in explanation["top_positive_contributors"][:3]]
        if not top_reasons:
            top_reasons = [c["phrase"] for c in explanation["top_negative_contributors"][:3]]
    except Exception as e:
        logger.warning("SHAP explanation failed (%s: %s); returning score without explanation.", type(e).__name__, e)

    try:
        spike_context = get_step_spike_context(cleaned["step"])
    except SpikeDetectorUnavailableError as e:
        spike_context = {"available": False, "message": str(e)}

    if spike_context.get("available") and spike_context.get("is_spike") and risk_band == "LOW":
        recommended_action = "REVIEW"
        top_reasons = top_reasons + [
            "transaction occurred during a step flagged by the spike-detection layer as an elevated-risk time window"
        ]

    return {
        "status": "ok",
        "risk_score": round(risk_score, 6),
        "risk_band": risk_band,
        "recommended_action": recommended_action,
        "mode": mode,
        "threshold_used": threshold,
        "top_reasons": top_reasons,
        "explanation": explanation,
        "spike_context": spike_context,
        "message": None,
        "fallback": None,
    }
