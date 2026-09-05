"""
Phase 7 - dedicated tests for the single shared-secret API key auth
mechanism (backend/core/auth.py). Covers all three PROTECTED endpoints
(transactions/score, spikes/analyze, model-info) plus the one EXEMPT
endpoint (health).
"""
import pytest

import risk_engine.risk_engine as re_mod


@pytest.fixture(autouse=True)
def _reset_caches_around_each_test():
    re_mod.clear_caches()
    yield
    re_mod.clear_caches()


def _protected_calls(client, valid_transaction, headers):
    """One representative call per protected endpoint, using the given headers."""
    return {
        "transactions/score": client.post("/api/v1/transactions/score", json=valid_transaction, headers=headers),
        "spikes/analyze": client.post("/api/v1/spikes/analyze", json={"step": 5}, headers=headers),
        "model-info": client.get("/api/v1/model-info", headers=headers),
    }


def test_missing_key_returns_401_on_every_protected_endpoint(api_client, valid_transaction):
    responses = _protected_calls(api_client, valid_transaction, headers=None)
    for name, r in responses.items():
        assert r.status_code == 401, f"{name} should 401 with no API key at all"
        assert r.json() == {"detail": "Unauthorized"}


def test_wrong_key_returns_401_with_identical_body_to_missing_key(api_client, valid_transaction, auth_headers):
    header_name = next(iter(auth_headers))
    wrong_headers = {header_name: "definitely-the-wrong-key"}

    responses = _protected_calls(api_client, valid_transaction, headers=wrong_headers)
    for name, r in responses.items():
        assert r.status_code == 401, f"{name} should 401 with a wrong API key"
        # SAME body as the missing-key case - a caller cannot tell "wrong" from "missing"
        assert r.json() == {"detail": "Unauthorized"}


def test_correct_key_allows_normal_behavior_on_every_protected_endpoint(api_client, valid_transaction, auth_headers):
    responses = _protected_calls(api_client, valid_transaction, headers=auth_headers)
    assert responses["transactions/score"].status_code == 200
    assert responses["transactions/score"].json()["status"] == "ok"

    assert responses["spikes/analyze"].status_code == 200
    assert responses["spikes/analyze"].json()["step"] == 5

    assert responses["model-info"].status_code == 200
    assert responses["model-info"].json()["model_name"] == "xgboost"


def test_empty_string_key_is_treated_as_wrong_not_bypassed(api_client, valid_transaction):
    r = api_client.post(
        "/api/v1/transactions/score", json=valid_transaction, headers={"X-API-Key": ""}
    )
    assert r.status_code == 401


def test_health_reachable_without_any_key(api_client):
    r = api_client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json()["status"] in {"healthy", "degraded"}


def test_health_reachable_even_with_a_wrong_key_present(api_client):
    """A caller sending a garbage key to /health must not be punished for it -
    /health ignores auth entirely, per design."""
    r = api_client.get("/api/v1/health", headers={"X-API-Key": "garbage"})
    assert r.status_code == 200


def test_configured_header_name_matches_settings(api_client, valid_transaction):
    """The header name itself is configurable (settings.api_key_header_name) -
    this test pins the DEFAULT name so a rename is a deliberate, visible change."""
    from backend.core.config import settings

    assert settings.api_key_header_name == "X-API-Key"
    r = api_client.post("/api/v1/transactions/score", json=valid_transaction, headers={"X-Wrong-Header": settings.api_key})
    assert r.status_code == 401  # the key was sent under the wrong header name entirely
