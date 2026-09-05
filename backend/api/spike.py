"""
Phase 5 - POST /api/v1/spikes/analyze

Thin wrapper over two Phase 3/4 read paths, kept separate in the response
exactly as they are separate in risk_engine (never merged into one score):

  - `step`           -> risk_engine.get_step_spike_context(step): a read-only
                         lookup of the precomputed hybrid (statistical OR
                         Isolation Forest) spike flag for that step, with both
                         signals reported individually.
  - `step_aggregate` -> risk_engine.analyze_spike(step_row, transactions_in_step):
                         a live, evidence-backed root-cause breakdown (Phase 4).

Either or both may be supplied in one request. No new spike-detection logic
is implemented here.
"""
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
