"""
Phase 5 - GET /api/v1/health

Reports whether the three artifacts predict_transaction_risk()/analyze_spike()
actually depend on are loaded successfully: the XGBoost model, the spike
detection context table, and the feature-reference-stats artifact. Each is
checked by calling risk_engine's OWN cached loader functions (load_model(),
load_spike_table(), load_feature_reference_stats()) - not by re-implementing
a file-exists check - so this reports the SAME load outcome the scoring
endpoints would actually get, not a weaker proxy for it.

Because risk_engine's loaders cache by path for the life of the process
(Phase 4), calling them here on every /health request is cheap after the
first successful call (a dict lookup, not a disk read) - so this can be
polled freely without violating "load once at startup, not per request".
The `backend.main` startup event calls `run_health_check()` once so the
first real user request doesn't pay the cold-load cost and so a broken
deployment is visible immediately, not only on the first scoring request.
"""
from datetime import datetime, timezone

from fastapi import APIRouter

import risk_engine.risk_engine as re_mod
from risk_engine.schemas import FeatureGenerationError, ModelUnavailableError, SpikeDetectorUnavailableError

router = APIRouter()

_KNOWN_ERRORS = (ModelUnavailableError, SpikeDetectorUnavailableError, FeatureGenerationError)


def _check_component(loader, path: str) -> dict:
    try:
        loader()
        return {"loaded": True, "path": path, "error": None}
    except _KNOWN_ERRORS as e:
        return {"loaded": False, "path": path, "error": str(e)}
    except Exception as e:  # genuinely unexpected - still reported, never hidden
        return {"loaded": False, "path": path, "error": f"unexpected error ({type(e).__name__}): {e}"}


def run_health_check() -> dict:
    """Re-runs (cache-hit-cheap, see module docstring) all three artifact
    loads used by the risk engine and reports their live status."""
    components = {
        "transaction_model": _check_component(re_mod.load_model, re_mod.MODEL_PATH),
        "spike_detector": _check_component(re_mod.load_spike_table, re_mod.EXTENDED_TIME_WINDOW_PATH),
        "feature_reference_stats": _check_component(re_mod.load_feature_reference_stats, re_mod.FEATURE_REFERENCE_STATS_PATH),
    }
    overall = "healthy" if all(c["loaded"] for c in components.values()) else "degraded"
    return {
        "status": overall,
        "components": components,
        "notes": {
            "spike_detector": (
                "Reflects data/processed/time_window_features_extended.parquet - the precomputed "
                "statistical + Isolation Forest spike context this API reads. models/isolation_forest.joblib "
                "itself is not loaded directly by this service (it was used offline to produce that table)."
            ),
        },
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/health")
async def health():
    return run_health_check()
