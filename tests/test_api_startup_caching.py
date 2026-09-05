"""
Verifies the cross-cutting requirement: the XGBoost model (and the other
risk-engine artifacts) are loaded ONCE, not per request. This directly
exercises the API (not just risk_engine in isolation) to prove the FastAPI
wiring doesn't accidentally defeat risk_engine's Phase 4 caching - e.g. by
constructing a fresh model instance per call somewhere in the request path.
"""
from unittest.mock import patch

import risk_engine.risk_engine as re_mod


def _reset():
    re_mod.clear_caches()


def test_model_object_identity_stable_across_two_requests(api_client, valid_transaction, auth_headers):
    _reset()
    model_before_any_request = re_mod.load_model()

    r1 = api_client.post("/api/v1/transactions/score", json=valid_transaction, headers=auth_headers)
    assert r1.status_code == 200
    model_after_first_request = re_mod.load_model()

    r2 = api_client.post("/api/v1/transactions/score", json=valid_transaction, headers=auth_headers)
    assert r2.status_code == 200
    model_after_second_request = re_mod.load_model()

    assert model_before_any_request is model_after_first_request is model_after_second_request
    _reset()


def test_joblib_load_called_at_most_once_across_two_requests(api_client, valid_transaction, auth_headers):
    _reset()
    with patch("risk_engine.risk_engine.joblib.load", wraps=re_mod.joblib.load) as spy_load:
        api_client.post("/api/v1/transactions/score", json=valid_transaction, headers=auth_headers)
        api_client.post("/api/v1/transactions/score", json=valid_transaction, headers=auth_headers)
        api_client.post("/api/v1/transactions/score", json=valid_transaction, headers=auth_headers)
        assert spy_load.call_count <= 1, (
            f"expected the model file to be read from disk at most once across 3 requests, "
            f"got {spy_load.call_count} calls - caching is not working as intended"
        )
    _reset()


def test_spike_table_also_cached_across_requests():
    _reset()
    t1 = re_mod.load_spike_table()
    t2 = re_mod.load_spike_table()
    assert t1 is t2
    _reset()
