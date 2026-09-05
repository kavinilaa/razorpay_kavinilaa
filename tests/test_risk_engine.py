"""
Unit tests for src/risk_engine/risk_engine.py.

These use the REAL, already-trained artifacts committed in models/ (small
files: xgboost.joblib is ~147KB) rather than the 6.36M-row dataset - no
retraining, no raw CSV needed.
"""
import json
import os

import numpy as np
import pytest

import risk_engine.risk_engine as re_mod
from risk_engine import build_features, predict_transaction_risk
from risk_engine.schemas import FeatureGenerationError


@pytest.fixture(autouse=True)
def _reset_caches_around_each_test():
    re_mod.clear_caches()
    yield
    re_mod.clear_caches()


# ---------------------------------------------------------------------------
# Feature generation
# ---------------------------------------------------------------------------
def test_build_features_returns_expected_columns(valid_transaction):
    from risk_engine.schemas import validate_transaction
    cleaned = validate_transaction(valid_transaction)
    X = build_features(cleaned)
    metadata = re_mod.load_model_metadata()
    assert list(X.columns) == metadata["feature_list"]
    assert len(X) == 1
    assert np.isfinite(X.to_numpy().astype(float)).all()


def test_build_features_drained_to_zero_flag(valid_transaction):
    from risk_engine.schemas import validate_transaction
    cleaned = validate_transaction(valid_transaction)  # newbalanceOrig == 0
    X = build_features(cleaned)
    assert X.iloc[0]["drained_to_zero"] == 1


def test_build_features_new_destination_defaults(legit_transaction):
    from risk_engine.schemas import validate_transaction
    txn = dict(legit_transaction)
    # remove the supplied history so defaults kick in
    for k in ["hist_dest_txn_count", "hist_dest_unique_origin_count", "hist_dest_total_amount",
              "hist_dest_avg_amount", "hist_dest_max_amount"]:
        txn.pop(k, None)
    cleaned = validate_transaction(txn)
    X = build_features(cleaned)
    assert X.iloc[0]["destination_is_new"] == 1
    assert X.iloc[0]["hist_dest_txn_count"] == 0


def test_build_features_raises_on_nan_history_override(valid_transaction):
    from risk_engine.schemas import validate_transaction
    txn = dict(valid_transaction)
    txn["hist_dest_avg_amount"] = float("nan")
    cleaned = validate_transaction(txn)
    with pytest.raises(FeatureGenerationError):
        build_features(cleaned)


# ---------------------------------------------------------------------------
# End-to-end risk prediction
# ---------------------------------------------------------------------------
def test_predict_transaction_risk_ok_for_fraud_like(valid_transaction):
    result = predict_transaction_risk(valid_transaction)
    assert result["status"] == "ok"
    assert 0.0 <= result["risk_score"] <= 1.0
    assert result["risk_band"] in {"LOW", "MEDIUM", "HIGH"}
    assert result["recommended_action"] in {"ALLOW", "REVIEW"}
    assert result["explanation"] is not None
    assert isinstance(result["top_reasons"], list) and len(result["top_reasons"]) > 0


def test_predict_transaction_risk_ok_for_legit_like(legit_transaction):
    result = predict_transaction_risk(legit_transaction)
    assert result["status"] == "ok"
    assert 0.0 <= result["risk_score"] <= 1.0


def test_two_different_transactions_get_different_explanations(valid_transaction, legit_transaction):
    r1 = predict_transaction_risk(valid_transaction)
    r2 = predict_transaction_risk(legit_transaction)
    assert r1["explanation"]["narrative"] != r2["explanation"]["narrative"]


def test_predict_transaction_risk_invalid_input():
    result = predict_transaction_risk({"step": 1, "type": "NOT_A_TYPE", "amount": -5})
    assert result["status"] == "invalid_input"
    assert result["risk_score"] is None
    assert result["fallback"] == "MANUAL_REVIEW"


def test_mode_changes_threshold_used(valid_transaction):
    r_balanced = predict_transaction_risk(valid_transaction, mode="BALANCED")
    r_precision = predict_transaction_risk(valid_transaction, mode="HIGH_PRECISION")
    assert r_balanced["threshold_used"] == 0.43
    assert r_precision["threshold_used"] == 0.17


def test_unknown_mode_degrades_gracefully(valid_transaction):
    result = predict_transaction_risk(valid_transaction, mode="ULTRA_MODE")
    assert result["status"] == "degraded"
    assert result["fallback"] == "MANUAL_REVIEW"


# ---------------------------------------------------------------------------
# Failure handling: missing / corrupt model
# ---------------------------------------------------------------------------
def test_missing_model_file(valid_transaction, monkeypatch):
    monkeypatch.setattr(re_mod, "MODEL_PATH", os.path.join("nonexistent_dir", "xgboost.joblib"))
    result = predict_transaction_risk(valid_transaction)
    assert result["status"] == "degraded"
    assert result["risk_score"] is None
    assert "model unavailable" in result["message"].lower()
    assert result["fallback"] == "MANUAL_REVIEW"


def test_corrupt_model_file(valid_transaction, monkeypatch, tmp_path):
    bad_model_path = tmp_path / "corrupt_xgboost.joblib"
    bad_model_path.write_bytes(b"this is not a valid joblib pickle")
    monkeypatch.setattr(re_mod, "MODEL_PATH", str(bad_model_path))
    result = predict_transaction_risk(valid_transaction)
    assert result["status"] == "degraded"
    assert result["risk_score"] is None
    assert "could not be loaded" in result["message"]


def test_missing_feature_reference_stats(valid_transaction, monkeypatch):
    monkeypatch.setattr(re_mod, "FEATURE_REFERENCE_STATS_PATH", os.path.join("nonexistent_dir", "stats.json"))
    result = predict_transaction_risk(valid_transaction)
    assert result["status"] == "degraded"
    assert "feature reference stats" in result["message"].lower()


# ---------------------------------------------------------------------------
# Failure handling: spike detector unavailable (must NOT block scoring)
# ---------------------------------------------------------------------------
def test_spike_detector_unavailable_does_not_block_scoring(valid_transaction, monkeypatch):
    monkeypatch.setattr(re_mod, "EXTENDED_TIME_WINDOW_PATH", os.path.join("nonexistent_dir", "twf.parquet"))
    result = predict_transaction_risk(valid_transaction)
    assert result["status"] == "ok"
    assert result["risk_score"] is not None
    assert result["spike_context"]["available"] is False
