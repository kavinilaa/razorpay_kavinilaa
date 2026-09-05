from fastapi import APIRouter, Query, Request, Response

from risk_engine import predict_transaction_risk

from backend.schemas.transaction import OperatingMode, TransactionRequest

router = APIRouter()


@router.post(
    "/score",
    summary="Score a single transaction for fraud risk",
    responses={
        200: {"description": "Scored successfully (status='ok') or a handled degradation (status='degraded')."},
        422: {"description": "Request-shape validation failure (Pydantic) OR a domain validation failure "
                              "(status='invalid_input' from risk_engine) - see reports/phase5_api_summary.md."},
    },
)
async def score_transaction(
    payload: TransactionRequest,
    request: Request,
    response: Response,
    operating_mode: OperatingMode = Query(
        OperatingMode.BALANCED,
        description="HIGH_RECALL / BALANCED / HIGH_PRECISION - see GET /api/v1/model-info for each "
                     "mode's measured precision/recall/alert-rate. Never silently defaulted to a "
                     "single 'best' mode - this remains the caller's choice per the Phase 4 decision "
                     "that no cost-optimal threshold exists yet.",
    ),
):
    # payload has already passed Pydantic SHAPE validation (types, required-ness,
    # numeric ranges) by the time this line runs. exclude_none drops unset optional
    # fields so predict_transaction_risk()'s own defaulting logic applies to them.
    transaction_dict = payload.model_dump(exclude_none=True)
    result = predict_transaction_risk(transaction_dict, mode=operating_mode.value)

    response.status_code = 422 if result["status"] == "invalid_input" else 200

    request.state.log_extra = {
        "operating_mode": result.get("mode"),
        "risk_band": result.get("risk_band"),
    }
    return result
