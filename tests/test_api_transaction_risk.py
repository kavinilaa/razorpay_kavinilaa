"""Unit tests for POST /api/v1/transactions/score."""
import os

import pytest

import risk_engine.risk_engine as re_mod


@pytest.fixture(autouse=True)
def _reset_caches_around_each_test():
    re_mod.clear_caches()
    yield
    re_mod.clear_caches()


def test_score_happy_path(api_client, valid_transaction, auth_headers):
    r = api_client.post("/api/v1/transactions/score", json=valid_transaction, headers=auth_headers)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert 0.0 <= body["risk_score"] <= 1.0
    assert body["risk_band"] in {"LOW", "MEDIUM", "HIGH"}
    assert body["mode"] == "BALANCED"
    assert body["threshold_used"] == 0.43
    assert len(body["top_reasons"]) > 0
    assert body["explanation"] is not None


def test_score_respects_operating_mode_query_param(api_client, valid_transaction, auth_headers):
    r = api_client.post(
        "/api/v1/transactions/score?operating_mode=HIGH_PRECISION", json=valid_transaction, headers=auth_headers
    )
    assert r.status_code == 200
    assert r.json()["threshold_used"] == 0.17
    assert r.json()["mode"] == "HIGH_PRECISION"


def test_score_invalid_operating_mode_rejected_by_pydantic(api_client, valid_transaction, auth_headers):
    r = api_client.post(
        "/api/v1/transactions/score?operating_mode=SUPER_AGGRESSIVE", json=valid_transaction, headers=auth_headers
    )
    assert r.status_code == 422
    assert "detail" in r.json()  # FastAPI's own enum-validation error shape


def test_score_shape_invalid_negative_amount_returns_pydantic_422(api_client, auth_headers):
    bad = {
        "step": 5, "type": "TRANSFER", "amount": -100.0,
        "oldbalanceOrg": 100.0, "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
    }
    r = api_client.post("/api/v1/transactions/score", json=bad, headers=auth_headers)
    assert r.status_code == 422
    body = r.json()
    assert "detail" in body
    assert any("amount" in str(err.get("loc")) for err in body["detail"])
    # this is FastAPI/Pydantic's OWN error shape - never reached predict_transaction_risk()
    assert "status" not in body


def test_score_shape_invalid_missing_field_returns_pydantic_422(api_client, auth_headers):
    r = api_client.post("/api/v1/transactions/score", json={"step": 5, "type": "TRANSFER"}, headers=auth_headers)
    assert r.status_code == 422
    assert "detail" in r.json()


def test_score_domain_invalid_unknown_type_returns_engine_422(api_client, auth_headers):
    """type is shape-valid (non-empty string) but not a real PaySim type -
    this must reach predict_transaction_risk() and come back as its own
    invalid_input envelope, distinct from the Pydantic error shape above."""
    payload = {
        "step": 5, "type": "NOT_A_REAL_TYPE", "amount": 100.0,
        "oldbalanceOrg": 100.0, "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
    }
    r = api_client.post("/api/v1/transactions/score", json=payload, headers=auth_headers)
    assert r.status_code == 422
    body = r.json()
    assert body["status"] == "invalid_input"
    assert "unknown transaction type" in body["message"].lower()
    assert body["risk_score"] is None
    assert body["fallback"] == "MANUAL_REVIEW"


def test_score_degraded_missing_model_returns_200_not_500(api_client, valid_transaction, auth_headers, monkeypatch):
    monkeypatch.setattr(re_mod, "MODEL_PATH", os.path.join("nonexistent_dir", "xgboost.joblib"))
    r = api_client.post("/api/v1/transactions/score", json=valid_transaction, headers=auth_headers)
    assert r.status_code == 200  # a handled degradation - NOT a 500
    body = r.json()
    assert body["status"] == "degraded"
    assert body["risk_score"] is None
    assert body["fallback"] == "MANUAL_REVIEW"


def test_score_spike_context_included_when_available(api_client, valid_transaction, auth_headers):
    r = api_client.post("/api/v1/transactions/score", json=valid_transaction, headers=auth_headers)
    body = r.json()
    assert body["spike_context"] is not None
    assert "is_spike" in body["spike_context"]


def test_score_spike_unavailable_does_not_block_scoring(api_client, valid_transaction, auth_headers, monkeypatch):
    monkeypatch.setattr(re_mod, "EXTENDED_TIME_WINDOW_PATH", os.path.join("nonexistent_dir", "twf.parquet"))
    r = api_client.post("/api/v1/transactions/score", json=valid_transaction, headers=auth_headers)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["spike_context"]["available"] is False
