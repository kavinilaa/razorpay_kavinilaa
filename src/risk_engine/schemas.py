from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Errors - one per failure mode documented in reports/phase4_failure_recovery.md
# ---------------------------------------------------------------------------
class RiskEngineError(Exception):
    """Base class for all risk-engine failures. Never caught silently -
    always converted to an explicit, typed response by risk_engine.py."""


class InvalidTransactionError(RiskEngineError):
    """Raised when the input transaction fails validation (missing fields,
    invalid amount, unknown transaction type, non-finite values)."""

    def __init__(self, errors: list):
        self.errors = errors
        super().__init__("; ".join(errors))


class ModelUnavailableError(RiskEngineError):
    """Raised when the transaction-risk model file is missing, unreadable, or
    fails to load (corrupt file)."""


class FeatureGenerationError(RiskEngineError):
    """Raised when the engineered feature vector cannot be safely built - e.g.
    a required feature ends up missing, NaN, or infinite after generation."""


class SpikeDetectorUnavailableError(RiskEngineError):
    """Raised when the spike-detection layer (Isolation Forest model or the
    extended time-window feature table) cannot be loaded or evaluated."""


# ---------------------------------------------------------------------------
# Input schema
# ---------------------------------------------------------------------------
# Required raw PaySim-style fields. These are the same raw columns
# src/features/feature_engineering.py reads from the source CSV (Phase 2).
REQUIRED_FIELDS = [
    "step", "type", "amount",
    "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
]

# Optional fields. nameOrig/nameDest are reference-only (never used as raw
# model features - Phase 2 leakage audit). The hist_dest_* fields let a
# caller supply real destination-history statistics if it has them (e.g. from
# a live feature store); when omitted, the engine treats the destination as
# brand-new, which is a documented, conservative default - see
# risk_engine.py::build_features and reports/phase4_failure_recovery.md.
OPTIONAL_FIELDS_DEFAULTS = {
    "nameOrig": None,
    "nameDest": None,
    "hist_dest_txn_count": 0,
    "hist_dest_unique_origin_count": 0,
    "hist_dest_total_amount": 0.0,
    "hist_dest_avg_amount": 0.0,
    "hist_dest_max_amount": 0.0,
}

VALID_TYPES = {"PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"}


def _is_finite_number(x: Any) -> bool:
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return False
    return xf == xf and xf not in (float("inf"), float("-inf"))  # NaN check + inf check


def validate_transaction(raw: dict) -> dict:
    """Validate a raw transaction dict. Returns a cleaned dict (required
    fields normalized to the right type, optional fields defaulted) on
    success. Raises InvalidTransactionError with a list of ALL problems found
    (not just the first) on failure, so a caller/UI can show a complete error
    list in one round trip.
    """
    if not isinstance(raw, dict):
        raise InvalidTransactionError([f"transaction must be a dict/object, got {type(raw).__name__}"])

    errors = []
    cleaned = {}

    missing = [f for f in REQUIRED_FIELDS if f not in raw or raw[f] is None]
    if missing:
        errors.append(f"missing required field(s): {', '.join(missing)}")

    # step
    if "step" in raw and raw["step"] is not None:
        try:
            step = int(raw["step"])
            if step < 1:
                errors.append(f"invalid step: {raw['step']} (must be >= 1)")
            else:
                cleaned["step"] = step
        except (TypeError, ValueError):
            errors.append(f"invalid step: {raw['step']!r} (must be an integer)")

    # type
    if "type" in raw and raw["type"] is not None:
        txn_type = str(raw["type"]).strip().upper()
        if txn_type not in VALID_TYPES:
            errors.append(f"unknown transaction type: {raw['type']!r} (must be one of {sorted(VALID_TYPES)})")
        else:
            cleaned["type"] = txn_type

    # amount
    if "amount" in raw and raw["amount"] is not None:
        if not _is_finite_number(raw["amount"]):
            errors.append(f"invalid amount: {raw['amount']!r} (must be a finite number)")
        elif float(raw["amount"]) < 0:
            errors.append(f"invalid amount: {raw['amount']!r} (must be >= 0)")
        else:
            cleaned["amount"] = float(raw["amount"])

    # balances
    for bal_field in ["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]:
        if bal_field in raw and raw[bal_field] is not None:
            if not _is_finite_number(raw[bal_field]):
                errors.append(f"invalid {bal_field}: {raw[bal_field]!r} (must be a finite number)")
            elif float(raw[bal_field]) < 0:
                errors.append(f"invalid {bal_field}: {raw[bal_field]!r} (must be >= 0)")
            else:
                cleaned[bal_field] = float(raw[bal_field])

    if errors:
        raise InvalidTransactionError(errors)

    # optional fields
    for k, default in OPTIONAL_FIELDS_DEFAULTS.items():
        val = raw.get(k, default)
        if val is None:
            val = default
        cleaned[k] = val

    return cleaned


# ---------------------------------------------------------------------------
# Output schemas (documentation dataclasses - risk_engine.py builds plain
# dicts matching these shapes so the eventual API layer has a stable,
# type-checkable contract to adopt without redesigning the payload).
# ---------------------------------------------------------------------------
@dataclass
class Contributor:
    feature: str
    value: float
    shap_value: float
    phrase: str


@dataclass
class Explanation:
    top_positive_contributors: list
    top_negative_contributors: list
    narrative: str


@dataclass
class RiskAssessment:
    status: str                       # "ok" | "invalid_input" | "degraded"
    risk_score: Optional[float]
    risk_band: Optional[str]          # "LOW" | "MEDIUM" | "HIGH"
    recommended_action: Optional[str]  # "ALLOW" | "REVIEW" | "MANUAL_REVIEW"
    mode: Optional[str]                # "HIGH_RECALL" | "BALANCED" | "HIGH_PRECISION"
    threshold_used: Optional[float]
    top_reasons: list = field(default_factory=list)
    explanation: Optional[dict] = None
    spike_context: Optional[dict] = None
    message: Optional[str] = None
    fallback: Optional[str] = None
