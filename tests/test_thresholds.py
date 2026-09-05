"""Unit tests for src/risk_engine/thresholds.py - operating modes and risk bands."""
import pytest

from risk_engine import thresholds


def test_all_three_modes_defined_with_required_fields():
    for mode in ["HIGH_RECALL", "BALANCED", "HIGH_PRECISION"]:
        cfg = thresholds.OPERATING_MODES[mode]
        assert "threshold" in cfg
        assert "validation" in cfg and "precision" in cfg["validation"] and "recall" in cfg["validation"]
        assert "alert_rate" in cfg["validation"]


def test_get_threshold_known_modes():
    assert thresholds.get_threshold("BALANCED") == 0.43
    assert thresholds.get_threshold("high_precision") == 0.17  # case-insensitive
    assert thresholds.get_threshold("HIGH_RECALL") == 0.43


def test_get_threshold_unknown_mode_raises():
    with pytest.raises(ValueError):
        thresholds.get_threshold("SUPER_AGGRESSIVE")


@pytest.mark.parametrize("score,expected_band", [
    (0.0, "LOW"),
    (0.05, "LOW"),
    (0.169, "LOW"),
    (0.17, "MEDIUM"),
    (0.30, "MEDIUM"),
    (0.429, "MEDIUM"),
    (0.43, "HIGH"),
    (0.99, "HIGH"),
    (1.0, "HIGH"),
])
def test_risk_band_boundaries_match_phase3_thresholds(score, expected_band):
    assert thresholds.get_risk_band(score) == expected_band


def test_recommended_action_mapping():
    assert thresholds.get_recommended_action("LOW") == "ALLOW"
    assert thresholds.get_recommended_action("MEDIUM") == "REVIEW"
    assert thresholds.get_recommended_action("HIGH") == "REVIEW"
    with pytest.raises(ValueError):
        thresholds.get_recommended_action("CRITICAL")
