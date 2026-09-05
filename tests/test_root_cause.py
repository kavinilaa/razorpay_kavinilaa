"""Unit tests for src/risk_engine/root_cause.py - spike root-cause analysis."""
import pandas as pd

from risk_engine.root_cause import analyze_spike


def _base_step_row(**overrides):
    row = {
        "step": 100,
        "hour_of_day": 3,
        "transaction_count": 20,
        "hist_avg_txn_count_same_hour": 18,
        "average_amount": 50000.0,
        "hist_avg_amount_same_hour": 48000.0,
        "transfer_count": 10,
        "hist_avg_transfer_count_same_hour": 9,
        "cash_out_count": 10,
        "hist_avg_cash_out_count_same_hour": 9,
        "high_amount_count": 2,
        "hist_avg_high_amount_count_same_hour": 2,
        "predicted_high_risk_count": 3,
        "hist_avg_predicted_high_risk_count_same_hour": 3,
    }
    row.update(overrides)
    return row


def test_identifies_transfer_volume_as_primary_driver():
    row = _base_step_row(transfer_count=40, hist_avg_transfer_count_same_hour=10)  # 4x baseline
    result = analyze_spike(row)
    assert result["primary_driver"] == "TRANSFER volume"
    assert any("TRANSFER volume" in c["factor"] for c in result["contributors"])
    assert any("4.0x" in e for e in result["evidence"])


def test_no_driver_when_nothing_exceeds_threshold():
    row = _base_step_row()  # all ratios close to 1.0
    result = analyze_spike(row)
    assert result["primary_driver"] == "no single driver identified"
    assert any("No individual factor exceeded" in e for e in result["evidence"])


def test_destination_concentration_detected_when_data_supplied():
    row = _base_step_row()  # no type-count driver
    transactions = pd.DataFrame({
        "nameDest": ["C_MULE_1"] * 8 + ["C_other_%d" % i for i in range(2)],
        "amount": [10000] * 10,
    })
    result = analyze_spike(row, transactions_in_step=transactions)
    assert result["primary_driver"] == "destination concentration"
    assert result["baseline_comparison"]["destination_concentration"]["txn_count_share"] == 0.8


def test_destination_concentration_not_evaluated_without_data():
    row = _base_step_row()
    result = analyze_spike(row, transactions_in_step=None)
    assert "destination_concentration" not in result["baseline_comparison"]
    assert any("not evaluated" in e for e in result["evidence"])


def test_missing_keys_are_skipped_not_fabricated():
    minimal_row = {"step": 1, "hour_of_day": 0}
    result = analyze_spike(minimal_row)
    assert result["contributors"] == []
    assert result["primary_driver"] == "no single driver identified"
