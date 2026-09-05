import os

import pytest

import risk_engine.risk_engine as re_mod


@pytest.fixture(autouse=True)
def _reset_caches_around_each_test():
    re_mod.clear_caches()
    yield
    re_mod.clear_caches()


def test_model_info_happy_path_contains_required_facts(api_client, auth_headers):
    r = api_client.get("/api/v1/model-info", headers=auth_headers)
    assert r.status_code == 200
    body = r.json()

    assert body["model_name"] == "xgboost"
    assert body["model_file_sha256"] is not None and len(body["model_file_sha256"]) == 64
    assert body["train_steps"] == [1, 520]
    assert body["val_steps"] == [521, 631]
    assert body["test_steps"] == [632, 743]
    assert body["feature_count"] == 23
    assert set(body["operating_modes"]) == {"HIGH_RECALL", "BALANCED", "HIGH_PRECISION"}
    assert body["operating_modes"]["BALANCED"]["threshold"] == 0.43

    # the literal, machine-checkable data-provenance field required by the Phase 5 brief
    assert "synthetic" in body["data_provenance"].lower()
    assert "paysim" in body["data_provenance"].lower()

    # sourced from reports/model_card.md #10, not hand-retyped - must mention the real caveat content
    assert "drained" in body["synthetic_data_caveat"].lower() or "balance" in body["synthetic_data_caveat"].lower()


def test_model_info_wrong_method_rejected(api_client):
    # 405 is resolved at the routing layer (no POST route exists for /model-info) before any
    # dependency - including auth - ever runs, so no auth header is needed for this test.
    r = api_client.post("/api/v1/model-info")
    assert r.status_code == 405


def test_model_info_degrades_to_503_when_metadata_missing(api_client, auth_headers, monkeypatch):
    monkeypatch.setattr(re_mod, "MODEL_METADATA_PATH", os.path.join("nonexistent_dir", "model_metadata.json"))
    r = api_client.get("/api/v1/model-info", headers=auth_headers)
    assert r.status_code == 503
    assert "unavailable" in r.json()["detail"].lower()
