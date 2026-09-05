import os

import pytest

import risk_engine.risk_engine as re_mod


@pytest.fixture(autouse=True)
def _reset_caches_around_each_test():
    re_mod.clear_caches()
    yield
    re_mod.clear_caches()


def test_health_happy_path_reports_healthy(api_client):
    # Deliberately NO auth_headers here (and nowhere else in this file) - Phase 7 exempts
    # /api/v1/health from require_api_key precisely so this call can succeed with no key at all.
    # See tests/test_api_auth.py::test_health_reachable_without_any_key for the dedicated check.
    r = api_client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "healthy"
    for component in ["transaction_model", "spike_detector", "feature_reference_stats"]:
        assert body["components"][component]["loaded"] is True
        assert body["components"][component]["error"] is None


def test_health_wrong_method_rejected(api_client):
    r = api_client.post("/api/v1/health")
    assert r.status_code == 405  # not a "handled degradation" - a routing-level rejection


def test_health_reports_degraded_when_model_missing(api_client, monkeypatch):
    monkeypatch.setattr(re_mod, "MODEL_PATH", os.path.join("nonexistent_dir", "xgboost.joblib"))
    r = api_client.get("/api/v1/health")
    assert r.status_code == 200  # health endpoint itself always responds; the CONTENT signals degradation
    body = r.json()
    assert body["status"] == "degraded"
    assert body["components"]["transaction_model"]["loaded"] is False
    assert "not found" in body["components"]["transaction_model"]["error"]
    # other components must be unaffected
    assert body["components"]["feature_reference_stats"]["loaded"] is True
