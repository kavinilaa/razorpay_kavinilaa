
import pandas as pd
from fastapi import APIRouter

from risk_engine import analyze_spike, get_step_spike_context
from risk_engine.schemas import SpikeDetectorUnavailableError

from backend.schemas.spike import SpikeAnalysisRequest

router = APIRouter()


@router.post(
    "/analyze",
    summary="Look up a step's spike status and/or run root-cause analysis on a step aggregate",
    responses={
        200: {"description": "Always 200 - spike_context.available=False communicates an unavailable "
                              "spike detector without failing the request (Phase 4 policy: the spike "
                              "layer is a secondary signal, never a hard dependency)."},
        422: {"description": "Neither 'step' nor 'step_aggregate' was supplied, or the payload shape is invalid."},
    },
)
async def analyze_spike_endpoint(payload: SpikeAnalysisRequest):
    result: dict = {}

    if payload.step is not None:
        result["step"] = payload.step
        try:
            result["spike_context"] = get_step_spike_context(payload.step)
        except SpikeDetectorUnavailableError as e:
            result["spike_context"] = {"available": False, "message": str(e)}

    if payload.step_aggregate is not None:
        transactions_df = (
            pd.DataFrame(payload.transactions_in_step) if payload.transactions_in_step else None
        )
        result["root_cause"] = analyze_spike(payload.step_aggregate, transactions_in_step=transactions_df)

    return result
