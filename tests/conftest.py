"""
Phase 4 - pytest configuration shared by all tests.
AI Fraud-Spike & Risk Detection System (Razorpay Buildathon, Track 02)

Adds `src/` to sys.path so tests can `import risk_engine` and `from models import
spike_detection` the same way the rest of the codebase does, without installing
the project as a package.

None of these tests load the full 6.36M-row PaySim dataset. Tests that need
"real" artifacts use the small trained model files already checked into
`models/` (a few KB to ~1.5MB each); everything else uses small synthetic
fixtures built in-line.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
if ROOT not in sys.path:
    # so `import backend` (backend/ lives at the repo root, Phase 5) resolves
    # even when pytest is invoked without `python -m` (which would otherwise
    # be the thing putting ROOT on sys.path for us).
    sys.path.insert(0, ROOT)

import pytest


@pytest.fixture
def valid_transaction():
    """A synthetic, clearly fraud-pattern-like transaction (drains origin to
    zero, brand-new destination) - not real data."""
    return {
        "step": 5,
        "type": "TRANSFER",
        "amount": 1_500_000.0,
        "oldbalanceOrg": 1_500_000.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
    }


@pytest.fixture(scope="session")
def api_client():
    """A single FastAPI TestClient for the whole test session (Phase 5).
    The app's startup/shutdown lifespan runs once; individual tests that need
    a degraded artifact monkeypatch risk_engine.risk_engine's module-level
    path constants and call risk_engine.risk_engine.clear_caches() - every
    route calls the risk engine live per-request, so this reflects
    immediately without needing a fresh app/client per test.

    Carries NO default headers (in particular, no API key - see Phase 7's
    `auth_headers` fixture below) so a test that forgets to pass
    `auth_headers` fails loudly with a 401 rather than silently passing
    because a client-wide default happened to supply a valid key."""
    from fastapi.testclient import TestClient

    from backend.main import app

    with TestClient(app) as client:
        yield client


@pytest.fixture
def auth_headers():
    """Phase 7 - the header a caller must send to pass `require_api_key`.
    Reads the REAL configured key from backend.core.config.settings (the
    same object backend/core/auth.py checks against) rather than a
    hand-typed duplicate string, so this fixture can never silently drift
    from what the app actually requires."""
    from backend.core.config import settings

    return {settings.api_key_header_name: settings.api_key}


@pytest.fixture
def legit_transaction():
    """A synthetic, clearly ordinary-looking transaction."""
    return {
        "step": 200,
        "type": "CASH_OUT",
        "amount": 5_000.0,
        "oldbalanceOrg": 20_000.0,
        "newbalanceOrig": 15_000.0,
        "oldbalanceDest": 3_000.0,
        "newbalanceDest": 8_000.0,
        "hist_dest_txn_count": 12,
        "hist_dest_unique_origin_count": 9,
        "hist_dest_total_amount": 60_000.0,
        "hist_dest_avg_amount": 5_000.0,
        "hist_dest_max_amount": 9_000.0,
    }
