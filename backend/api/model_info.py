"""
Phase 5 - GET /api/v1/model-info

Every value returned here is sourced live from the real artifacts, never
re-typed by hand:
    - models/model_metadata.json          -> selected model, split boundaries, feature list
    - models/phase3_model_stats.json      -> validation/test PR-AUC, ROC-AUC, precision, recall
    - risk_engine.thresholds.OPERATING_MODES -> the same object used by the scoring endpoint
    - models/xgboost.joblib               -> a live sha256 hash (a real content fingerprint,
                                                                                        not a hand-assigned version string)
    - models/model_card.md                -> the synthetic-data caveat and prohibited-use text,
                                                                                        extracted verbatim from their named sections

If model_card.md is restructured and a named section can no longer be found,
this endpoint says so explicitly in the field instead of silently returning
stale hand-typed text.
"""
import hashlib
import json
import os

from fastapi import APIRouter, HTTPException

import risk_engine.risk_engine as re_mod
from risk_engine import thresholds
from risk_engine.schemas import ModelUnavailableError

router = APIRouter()

MODELS_DIR = os.path.join(re_mod.ROOT, "models")
MODEL_CARD_PATH = os.path.join(MODELS_DIR, "model_card.md")
MODEL_STATS_PATH = os.path.join(MODELS_DIR, "phase3_model_stats.json")


def _extract_markdown_section(md_text: str, heading_prefix: str) -> str:
    """Return the body of the first markdown section whose heading starts
    with `heading_prefix` (e.g. "## 10. Synthetic-data limitation"), up to
    (not including) the next "## " heading. Empty string if not found."""
    lines = md_text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith(heading_prefix):
            start = i + 1
            break
    if start is None:
        return ""
    end = len(lines)
    for j in range(start, len(lines)):
        if lines[j].strip().startswith("## "):
            end = j
            break
    return "\n".join(lines[start:end]).strip()


def _sha256_of(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def build_model_info() -> dict:
    metadata = re_mod.load_model_metadata()
    selected = metadata["selected_model"]

    val_metrics = test_metrics = None
    if os.path.exists(MODEL_STATS_PATH):
        with open(MODEL_STATS_PATH) as f:
            stats = json.load(f)
        val_metrics = stats["model_results"][selected]["val_metrics"]
        test_metrics = stats["model_results"][selected]["test_metrics"]

    model_hash = _sha256_of(re_mod.MODEL_PATH)

    caveat, prohibited = "", ""
    if os.path.exists(MODEL_CARD_PATH):
        with open(MODEL_CARD_PATH, encoding="utf-8") as f:
            card_text = f.read()
        caveat = _extract_markdown_section(card_text, "## 10. Synthetic-data limitation")
        prohibited = _extract_markdown_section(card_text, "## 13. Prohibited use")

    _not_found_msg = "section not found in models/model_card.md (it may have been restructured)."

    return {
        "model_name": selected,
        "model_version": f"{selected}-seed{metadata['random_seed']}" + (f"-{model_hash[:12]}" if model_hash else ""),
        "model_file_sha256": model_hash,
        "random_seed": metadata["random_seed"],
        "train_steps": metadata["train_steps"],
        "val_steps": metadata["val_steps"],
        "test_steps": metadata["test_steps"],
        "modeling_universe": metadata["modeling_universe"],
        "target": metadata["target"],
        "feature_count": len(metadata["feature_list"]),
        "features": metadata["feature_list"],
        "validation_metrics": (
            {k: val_metrics[k] for k in ("pr_auc", "roc_auc", "precision", "recall")} if val_metrics else None
        ),
        "test_metrics": (
            {k: test_metrics[k] for k in ("pr_auc", "roc_auc", "precision", "recall")} if test_metrics else None
        ),
        "operating_modes": {
            mode: {"threshold": cfg["threshold"], "validation": cfg["validation"], "test": cfg["test"]}
            for mode, cfg in thresholds.OPERATING_MODES.items()
        },
        "data_provenance": "synthetic PaySim dataset; not validated on real transaction data",
        "synthetic_data_caveat": caveat or _not_found_msg,
        "prohibited_use": prohibited or _not_found_msg,
        "source_artifacts": {
            "model_metadata": "models/model_metadata.json",
            "model_stats": "models/phase3_model_stats.json",
            "model_card": "models/model_card.md",
            "thresholds_module": "src/risk_engine/thresholds.py",
        },
    }


@router.get(
    "/model-info",
    responses={503: {"description": "models/model_metadata.json is missing/corrupt - the same "
                                     "condition GET /api/v1/health reports under 'transaction_model'."}},
)
async def model_info():
    # Not one of the 8 documented predict_transaction_risk() failure modes (this endpoint never
    # calls predict_transaction_risk()), but the same underlying artifact can fail the same way -
    # reported honestly as 503 rather than an unhandled 500.
    try:
        return build_model_info()
    except ModelUnavailableError as e:
        raise HTTPException(status_code=503, detail=f"Model metadata unavailable: {e}")
