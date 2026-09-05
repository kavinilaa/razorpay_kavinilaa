"""Unit tests for src/risk_engine/schemas.py - input validation."""
import math

import pytest

from risk_engine.schemas import InvalidTransactionError, validate_transaction


def test_valid_transaction_passes(valid_transaction):
    cleaned = validate_transaction(valid_transaction)
    assert cleaned["step"] == 5
    assert cleaned["type"] == "TRANSFER"
    assert cleaned["amount"] == 1_500_000.0
    # optional fields defaulted
    assert cleaned["hist_dest_txn_count"] == 0
    assert "destination_is_new" not in cleaned  # derived later in build_features, not by schema validation


def test_missing_required_fields_lists_all_of_them():
    with pytest.raises(InvalidTransactionError) as exc_info:
        validate_transaction({"step": 5, "type": "TRANSFER"})
    msg = str(exc_info.value)
    for f in ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]:
        assert f in msg


def test_invalid_negative_amount():
    txn = {
        "step": 1, "type": "TRANSFER", "amount": -100.0,
        "oldbalanceOrg": 100.0, "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
    }
    with pytest.raises(InvalidTransactionError) as exc_info:
        validate_transaction(txn)
    assert "invalid amount" in str(exc_info.value)


def test_invalid_non_numeric_amount():
    txn = {
        "step": 1, "type": "TRANSFER", "amount": "not-a-number",
        "oldbalanceOrg": 100.0, "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
    }
    with pytest.raises(InvalidTransactionError):
        validate_transaction(txn)


def test_nan_amount_rejected():
    txn = {
        "step": 1, "type": "TRANSFER", "amount": float("nan"),
        "oldbalanceOrg": 100.0, "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
    }
    with pytest.raises(InvalidTransactionError):
        validate_transaction(txn)


def test_unknown_transaction_type():
    txn = {
        "step": 1, "type": "WIRE_XYZ", "amount": 100.0,
        "oldbalanceOrg": 100.0, "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
    }
    with pytest.raises(InvalidTransactionError) as exc_info:
        validate_transaction(txn)
    assert "unknown transaction type" in str(exc_info.value)


def test_multiple_errors_reported_together():
    txn = {"step": 1, "type": "BAD_TYPE", "amount": -5}
    with pytest.raises(InvalidTransactionError) as exc_info:
        validate_transaction(txn)
    errors = exc_info.value.errors
    assert len(errors) >= 3  # missing balances + bad type + bad amount


def test_optional_history_fields_default_to_new_destination():
    txn = {
        "step": 1, "type": "CASH_OUT", "amount": 100.0,
        "oldbalanceOrg": 100.0, "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0, "newbalanceDest": 100.0,
    }
    cleaned = validate_transaction(txn)
    assert cleaned["hist_dest_txn_count"] == 0
    assert cleaned["hist_dest_avg_amount"] == 0.0


def test_not_a_dict_raises():
    with pytest.raises(InvalidTransactionError):
        validate_transaction(["not", "a", "dict"])
