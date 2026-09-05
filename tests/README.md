# Tests

Unit tests for `src/risk_engine/` and the pure functions in
`src/models/spike_detection.py`. None of these tests require the 6.36M-row
raw PaySim CSV or the full processed parquet files — they use small
synthetic fixtures (`conftest.py`, in-line DataFrames) and the small,
already-trained model artifacts already committed in `models/`
(`xgboost.joblib` is ~147KB).

## Run

```
python -m pytest tests/ -v
```

## Coverage

| File | Covers |
|---|---|
| `test_schemas.py` | Input validation: missing fields, invalid amount, unknown type, NaN, defaults |
| `test_thresholds.py` | Operating modes, risk-band boundaries, recommended-action mapping |
| `test_risk_engine.py` | Feature generation, end-to-end risk prediction, failure handling (missing/corrupt model, missing reference stats, NaN features, spike detector unavailable) |
| `test_explanations.py` | Real SHAP-based explanation generation, narrative uniqueness across transactions |
| `test_root_cause.py` | Spike root-cause driver identification, destination concentration, "no driver found" case |
| `test_spike_detection.py` | The statistical z-score method, Isolation Forest method, evaluation scoring, and root-cause analysis in `src/models/spike_detection.py`, exercised on small synthetic per-step tables |
