"""
Phase 5 - FastAPI service layer.
AI Fraud-Spike & Risk Detection System (Razorpay Buildathon, Track 02)

This package is a THIN wrapper around src/risk_engine/ (Phase 4). It must not
reimplement, retrain, or modify any Phase 1-4 logic - every endpoint calls
straight into risk_engine.predict_transaction_risk(), risk_engine.analyze_spike(),
or risk_engine's cached artifact loaders, and returns their output essentially
unmodified in shape.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    # Every backend.* submodule needs `import risk_engine` to resolve to
    # src/risk_engine without installing this project as a package. Doing it
    # once here (the package __init__) guarantees it runs before any
    # submodule's own imports, since Python always imports a parent package
    # first.
    sys.path.insert(0, _SRC)
