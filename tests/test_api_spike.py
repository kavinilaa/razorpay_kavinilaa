"""Unit tests for POST /api/v1/spikes/analyze."""
import os

import pytest

import risk_engine.risk_engine as re_mod


@pytest.fixture(autouse=True)
def _reset_caches_around_each_test():
    re_mod.clear_caches()
    yield
    re_mod.clear_caches()


def test_spike_lookup_happy_path(api_client, auth_headers):
    r = api_client.post("/api/v1/spikes/analyze", json={"step": 5}, headers=auth_headers)
    assert r.status_code == 200
    body = r.json()
    assert body["step"] == 5
    ctx = body["spike_context"]
    assert ctx["available"] is True
    # the two signals must remain SEPARATE fields, never collapsed
    assert "statistical_method_flag" in ctx
    assert "isolation_forest_flag" in ctx
    assert "is_spike" in ctx


def test_spike_root_cause_happy_path(api_client, auth_headers):
    step_aggregate = {
        "step": 100, "hour_of_day": 3, "transaction_count": 20,
        "hist_avg_txn_count_same_hour": 18,
        "transfer_count": 40, "hist_avg_transfer_count_same_hour": 10,
        "cash_out_count": 10, "hist_avg_cash_out_count_same_hour": 9,
        "high_amount_count": 2, "hist_avg_high_amount_count_same_hour": 2,
        "predicted_high_risk_count": 3, "hist_avg_predicted_high_risk_count_same_hour": 3,
    }
    r = api_client.post("/api/v1/spikes/analyze", json={"step_aggregate": step_aggregate}, headers=auth_headers)
    assert r.status_code == 200
    root_cause = r.json()["root_cause"]
    assert root_cause["primary_driver"] == "TRANSFER volume"
    assert len(root_cause["evidence"]) > 0


def test_spike_root_cause_with_destination_concentration(api_client, auth_headers):
    step_aggregate = {"step": 50, "hour_of_day": 2}
    transactions = [{"nameDest": "C_MULE", "amount": 1000}] * 8 + [
        {"nameDest": f"C_other_{i}", "amount": 1000} for i in range(2)
    ]
    r = api_client.post(
        "/api/v1/spikes/analyze",
        json={"step_aggregate": step_aggregate, "transactions_in_step": transactions},
        headers=auth_headers,
    )
    assert r.status_code == 200
    assert r.json()["root_cause"]["primary_driver"] == "destination concentration"


def test_spike_invalid_neither_field_supplied(api_client, auth_headers):
    r = api_client.post("/api/v1/spikes/analyze", json={}, headers=auth_headers)
    assert r.status_code == 422


def test_spike_invalid_negative_step(api_client, auth_headers):
    r = api_client.post("/api/v1/spikes/analyze", json={"step": -1}, headers=auth_headers)
    assert r.status_code == 422


def test_spike_detector_unavailable_returns_200_with_flag(api_client, auth_headers, monkeypatch):
    monkeypatch.setattr(re_mod, "EXTENDED_TIME_WINDOW_PATH", os.path.join("nonexistent_dir", "twf.parquet"))
    r = api_client.post("/api/v1/spikes/analyze", json={"step": 5}, headers=auth_headers)
    assert r.status_code == 200
    assert r.json()["spike_context"]["available"] is False
