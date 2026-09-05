"""
Phase 4 - Per-transaction SHAP explainability.
AI Fraud-Spike & Risk Detection System (Razorpay Buildathon, Track 02)

Generates the "top_reasons" / natural-language explanation for a single
transaction's risk score using REAL SHAP values from the trained XGBoost
model (models/xgboost.joblib) - never a fixed, hard-coded sentence per
transaction. Each feature has ONE factual description template, filled in
with THIS transaction's own value (e.g. "the origin balance exactly matches
the expected debit formula" is only said when balance_error_orig is actually
~0 for this row) - the model's SHAP sign then determines whether that fact is
reported as raising or lowering the score, rather than the phrase itself
guessing a direction. This avoids the bug class of a template that assumes
"large deviation = risky" when, for this dataset, the opposite can be true
(fraud in PaySim almost always satisfies the debit formula exactly - see
reports/phase3_model_comparison.md #8).
"""
import numpy as np
import pandas as pd
import shap


def _money(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    return f"{'-' if v < 0 else ''}Rs.{abs(v):,.0f}"


# ---------------------------------------------------------------------------
# One factual description per feature in the Phase 3 FEATURE_LIST
# (models/model_metadata.json). Purely descriptive of the actual value -
# no assumption baked in about whether that value is "risky".
# ---------------------------------------------------------------------------
FEATURE_DESCRIBE = {
    "amount": lambda v: f"the transaction amount is {_money(v)}",
    "log_amount": lambda v: f"the log-scaled transaction amount is {float(v):.2f}",
    "balance_change_orig": lambda v: (
        f"the origin account balance decreased by {_money(v)}" if v >= 0
        else f"the origin account balance increased by {_money(-v)}"
    ),
    "balance_error_orig": lambda v: (
        "the origin balance exactly matches the expected debit formula (oldbalance - amount = newbalance)"
        if abs(v) < 1e-6 else f"the origin balance deviates from the expected debit formula by {_money(v)}"
    ),
    "drained_to_zero": lambda v: (
        "the transaction drains the origin account to exactly zero" if v == 1
        else "the origin account is not fully drained by this transaction"
    ),
    "origin_balance_zero_before": lambda v: (
        "the origin account already had a zero balance before this transaction" if v == 1
        else "the origin account had a non-zero balance before this transaction"
    ),
    "amount_to_origin_balance_ratio": lambda v: (
        f"the transaction amount is {float(v):.1f}x the origin account's pre-transaction balance"
    ),
    "balance_change_dest": lambda v: (
        f"the destination account balance increased by {_money(v)}" if v >= 0
        else f"the destination account balance decreased by {_money(-v)}"
    ),
    "balance_error_dest": lambda v: (
        "the destination balance exactly matches the expected credit formula (oldbalance + amount = newbalance)"
        if abs(v) < 1e-6 else f"the destination balance deviates from the expected credit formula by {_money(v)}"
    ),
    "dest_balance_artifact_flag": lambda v: (
        "the destination's balance fields are both zero (an untracked/mule-like destination pattern)" if v == 1
        else "the destination's balance fields look normally tracked"
    ),
    "hour_of_day": lambda v: f"it occurred at hour {int(v)} of the simulated 24-hour cycle",
    "hour_sin": lambda v: "part of this transaction's cyclical time-of-day encoding",
    "hour_cos": lambda v: "part of this transaction's cyclical time-of-day encoding",
    "is_transfer": lambda v: "it is a TRANSFER transaction" if v == 1 else "it is a CASH_OUT transaction",
    "hist_dest_txn_count": lambda v: f"this destination has {int(v)} prior recorded transactions",
    "hist_dest_unique_origin_count": lambda v: f"this destination has received from {int(v)} distinct senders historically",
    "hist_dest_total_amount": lambda v: f"this destination has received {_money(v)} historically",
    "hist_dest_avg_amount": lambda v: f"this destination typically receives around {_money(v)}",
    "hist_dest_max_amount": lambda v: f"the largest amount this destination has ever received is {_money(v)}",
    "destination_is_new": lambda v: (
        "the destination account has no prior transaction history (brand new)" if v == 1
        else "the destination account has prior transaction history"
    ),
    "amount_to_dest_avg_ratio": lambda v: f"the amount is {float(v):.1f}x this destination's historical average received amount",
    "amount_to_type_avg_ratio": lambda v: f"the amount is {float(v):.1f}x the historical average amount for this transaction type",
    "high_amount_indicator": lambda v: (
        "the amount exceeds the historical 95th-percentile threshold for this transaction type" if v == 1
        else "the amount is within the normal range for this transaction type"
    ),
}


def _describe(feature: str, value) -> str:
    fn = FEATURE_DESCRIBE.get(feature)
    if fn is None:
        return f"{feature} = {value!r}"
    try:
        return fn(value)
    except Exception:
        return f"{feature} = {value!r}"


# ---------------------------------------------------------------------------
# SHAP explainer, cached per model object (by id) so it is built once, not
# once per prediction.
# ---------------------------------------------------------------------------
_EXPLAINER_CACHE = {}


def _get_explainer(model):
    key = id(model)
    if key not in _EXPLAINER_CACHE:
        _EXPLAINER_CACHE[key] = shap.TreeExplainer(model)
    return _EXPLAINER_CACHE[key]


def explain_prediction(model, X_row: pd.DataFrame, risk_score: float, risk_band: str, top_n: int = 3) -> dict:
    """Compute real SHAP contributions for one transaction's feature row and
    turn them into a structured explanation + natural-language narrative.

    Returns a dict with: top_positive_contributors, top_negative_contributors,
    feature_values, narrative.
    """
    if len(X_row) != 1:
        raise ValueError("explain_prediction expects a single-row DataFrame")

    explainer = _get_explainer(model)
    raw = explainer(X_row)
    values = np.asarray(raw.values)
    if values.ndim == 3:
        # (n_samples, n_features, n_outputs) - take the "fraud" (positive) class output
        values = values[..., -1]
    values = values[0]  # the one row we passed in

    feature_names = list(X_row.columns)
    feature_values = X_row.iloc[0].to_dict()

    contributions = list(zip(feature_names, values.tolist()))
    contributions.sort(key=lambda t: t[1], reverse=True)

    positive = [(f, v) for f, v in contributions if v > 0][:top_n]
    negative = [(f, v) for f, v in contributions if v < 0][:top_n]
    negative.sort(key=lambda t: t[1])  # most negative (most score-lowering) first

    def build_contributor(feature, shap_value):
        value = feature_values[feature]
        return {
            "feature": feature,
            "value": value,
            "shap_value": round(float(shap_value), 6),
            "phrase": _describe(feature, value),
        }

    top_positive = [build_contributor(f, v) for f, v in positive]
    top_negative = [build_contributor(f, v) for f, v in negative]

    narrative = _build_narrative(risk_score, risk_band, top_positive, top_negative)

    return {
        "top_positive_contributors": top_positive,
        "top_negative_contributors": top_negative,
        "feature_values": feature_values,
        "narrative": narrative,
    }


def _build_narrative(risk_score: float, risk_band: str, top_positive: list, top_negative: list) -> str:
    if risk_band in ("HIGH", "MEDIUM") and top_positive:
        clauses = [c["phrase"] for c in top_positive[:2]]
        return f"{risk_band.title()} risk (score {risk_score:.3f}) because " + " and ".join(clauses) + "."
    if risk_band == "LOW":
        if top_positive:
            raising = top_positive[0]["phrase"]
            if top_negative:
                offsetting = "; ".join(c["phrase"] for c in top_negative[:2])
                return (
                    f"Low risk overall (score {risk_score:.3f}). The model's strongest score-raising factor was that "
                    f"{raising}, but this was outweighed by: {offsetting}."
                )
            return f"Low risk overall (score {risk_score:.3f}), despite {raising}."
        return f"Low risk (score {risk_score:.3f}): no score-raising factors found among the model's top contributing features."
    return f"Risk score {risk_score:.3f} ({risk_band})."
