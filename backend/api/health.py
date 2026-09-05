
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
