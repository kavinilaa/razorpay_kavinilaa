from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class SpikeAnalysisRequest(BaseModel):
    """Either `step` (a precomputed spike-context lookup via
    risk_engine.get_step_spike_context) or `step_aggregate` (a live
    root-cause breakdown via risk_engine.analyze_spike) must be supplied;
    both may be supplied together. `transactions_in_step` is an optional
    list of {"nameDest": ..., "amount": ...} rows used only for the
    destination-concentration check inside analyze_spike().
    """
    step: Optional[int] = None
    step_aggregate: Optional[dict[str, Any]] = None
    transactions_in_step: Optional[list[dict[str, Any]]] = None

    @field_validator("step")
    @classmethod
    def step_must_be_positive(cls, v):
        if v is not None and v < 1:
            raise ValueError("step must be >= 1")
        return v

    @model_validator(mode="after")
    def require_step_or_aggregate(self):
        if self.step is None and self.step_aggregate is None:
            raise ValueError("either 'step' or 'step_aggregate' must be provided")
        return self


class SpikeContext(BaseModel):
    """Mirrors risk_engine.get_step_spike_context()'s return shape. Keeps the
    statistical and Isolation Forest signals as SEPARATE fields (Phase 3
    decision: never collapse into one unexplained 'spike score')."""
    model_config = ConfigDict(extra="allow")

    available: bool
    is_spike: Optional[bool] = None
    statistical_method_flag: Optional[bool] = None
    isolation_forest_flag: Optional[bool] = None
    txn_count_vs_hour_baseline_ratio: Optional[float] = None
    predicted_high_risk_count_vs_hour_baseline_ratio: Optional[float] = None
    note: Optional[str] = None
    message: Optional[str] = None


class RootCauseResult(BaseModel):
    """Mirrors risk_engine.root_cause.analyze_spike()'s return shape."""
    model_config = ConfigDict(extra="allow")

    step: Optional[int] = None
    hour_of_day: Optional[int] = None
    primary_driver: str
    contributors: list = []
    baseline_comparison: dict = {}
    evidence: list = []


class SpikeAnalysisResponse(BaseModel):
    """Documented for OpenAPI only - see TransactionRiskResponse's docstring
    for why the endpoint itself returns risk_engine's dict as-is."""
    model_config = ConfigDict(extra="allow")

    step: Optional[int] = None
    spike_context: Optional[SpikeContext] = None
    root_cause: Optional[RootCauseResult] = None
