"""
Phase 4 - Production-style risk engine.
AI Fraud-Spike & Risk Detection System (Razorpay Buildathon, Track 02)

Public API:
    predict_transaction_risk(transaction: dict, mode: str = "BALANCED") -> dict
        See risk_engine.py for the full contract and failure-mode behavior.

    analyze_spike(step_row: dict, transactions_in_step=None) -> dict
        See root_cause.py.

This package does not build or expose a web/API layer (FastAPI, frontend) -
that is explicitly out of scope for Phase 4 per the project brief. It is a
plain, importable Python module intended to be wrapped by an API layer later.
"""
from .risk_engine import build_features, get_step_spike_context, predict_transaction_risk
from .root_cause import analyze_spike
from .schemas import (
    FeatureGenerationError,
    InvalidTransactionError,
    ModelUnavailableError,
    RiskEngineError,
    SpikeDetectorUnavailableError,
    validate_transaction,
)
from . import thresholds

__all__ = [
    "predict_transaction_risk",
    "build_features",
    "get_step_spike_context",
    "analyze_spike",
    "validate_transaction",
    "thresholds",
    "RiskEngineError",
    "InvalidTransactionError",
    "ModelUnavailableError",
    "FeatureGenerationError",
    "SpikeDetectorUnavailableError",
]
