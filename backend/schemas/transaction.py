"""
Phase 5 - Pydantic request/response models for POST /api/v1/transactions/score.

Reuses src/risk_engine/schemas.py as the single source of truth for field
NAMES rather than retyping them:
  - required field names come from `REQUIRED_FIELDS`
  - optional field names AND their types come from `OPTIONAL_FIELDS_DEFAULTS`
    (the type is inferred from each default value, so adding a new optional
    field there does not require touching this file)
  - the response model's field set comes directly from the `RiskAssessment`
    dataclass via `dataclasses.fields()` - not hand-copied.

What this file intentionally does NOT reuse: numeric range constraints
(`step >= 1`, `amount >= 0`, ...) - those don't exist as machine-readable
metadata in schemas.py (REQUIRED_FIELDS is just a list of names), so they are
declared here once, explicitly, as the Pydantic-layer's SHAPE validation.
`test_transaction_schema.py` asserts this file's required/optional field sets
stay in sync with schemas.py, so drift is caught immediately rather than
silently.

Design decision (see reports/phase5_api_summary.md): `type` is validated here
only as "non-empty string" (a shape concern). Whether it's one of the five
valid PaySim types is left to risk_engine.schemas.validate_transaction()
(a domain concern) - this is the split the Phase 5 brief explicitly asked
for: "unknown transaction type" is domain validation, not shape validation.
"""
import dataclasses
from enum import Enum
from typing import Optional

from pydantic import Field, create_model

from risk_engine.schemas import OPTIONAL_FIELDS_DEFAULTS, REQUIRED_FIELDS, RiskAssessment
from risk_engine.thresholds import OPERATING_MODES

# ---------------------------------------------------------------------------
# operating_mode - sourced from thresholds.OPERATING_MODES's own keys, so a
# new/renamed mode there is automatically reflected here.
# ---------------------------------------------------------------------------
OperatingMode = Enum("OperatingMode", {mode: mode for mode in OPERATING_MODES})


# ---------------------------------------------------------------------------
# Request model - field NAMES sourced from schemas.py, constraints declared
# here (shape validation only).
# ---------------------------------------------------------------------------
_REQUIRED_FIELD_SPECS = {
    "step": (int, Field(..., ge=1, description="PaySim simulated hour (1-based).")),
    "type": (str, Field(..., min_length=1, description=(
        "Transaction type string. Shape-validated here as non-empty only; "
        "membership in the valid PaySim type set is a DOMAIN rule enforced "
        "downstream by risk_engine.schemas.validate_transaction()."
    ))),
    "amount": (float, Field(..., ge=0)),
    "oldbalanceOrg": (float, Field(..., ge=0)),
    "newbalanceOrig": (float, Field(..., ge=0)),
    "oldbalanceDest": (float, Field(..., ge=0)),
    "newbalanceDest": (float, Field(..., ge=0)),
}

# Sanity check at import time: every name risk_engine.schemas actually
# requires must have a shape spec here, and vice versa.
assert set(_REQUIRED_FIELD_SPECS) == set(REQUIRED_FIELDS), (
    f"backend/schemas/transaction.py is out of sync with risk_engine.schemas.REQUIRED_FIELDS: "
    f"{set(_REQUIRED_FIELD_SPECS) ^ set(REQUIRED_FIELDS)}"
)


def _infer_optional_type(default):
    """Infer a Pydantic-friendly type purely from an OPTIONAL_FIELDS_DEFAULTS
    default value, so this file never hand-lists optional field types."""
    if isinstance(default, bool):
        return Optional[bool]
    if isinstance(default, int):
        return Optional[int]
    if isinstance(default, float):
        return Optional[float]
    return Optional[str]


_optional_field_specs = {
    name: (_infer_optional_type(default), Field(default=None))
    for name, default in OPTIONAL_FIELDS_DEFAULTS.items()
}

TransactionRequest = create_model(
    "TransactionRequest",
    **_REQUIRED_FIELD_SPECS,
    **_optional_field_specs,
)
TransactionRequest.__doc__ = (
    "Single-transaction scoring request. Required fields match "
    "risk_engine.schemas.REQUIRED_FIELDS; optional fields (destination-history "
    "overrides, reference-only nameOrig/nameDest) match OPTIONAL_FIELDS_DEFAULTS "
    "and default to 'treat as a brand-new destination' when omitted - see "
    "src/risk_engine/risk_engine.py::build_features."
)


# ---------------------------------------------------------------------------
# Response model - field set sourced directly from the RiskAssessment
# dataclass in risk_engine.schemas, so it cannot silently drift from what
# predict_transaction_risk() actually returns.
# ---------------------------------------------------------------------------
def _default_for(f: dataclasses.Field):
    if f.default is not dataclasses.MISSING:
        return f.default
    if f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
        return f.default_factory()  # type: ignore[misc]
    return ...  # required, no default


_response_field_specs = {
    f.name: (f.type, _default_for(f)) for f in dataclasses.fields(RiskAssessment)
}

TransactionRiskResponse = create_model(
    "TransactionRiskResponse",
    **_response_field_specs,
)
TransactionRiskResponse.__doc__ = (
    "Response shape for POST /transactions/score, generated directly from "
    "risk_engine.schemas.RiskAssessment's fields (dataclasses.fields()) - "
    "documented here for OpenAPI only; the endpoint returns the risk engine's "
    "dict as-is and does not re-validate/strip it through this model."
)
