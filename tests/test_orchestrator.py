"""
Tests for orchestrator signal calculations.
Threshold logic:
  - Injury risk HIGH: 5+ games missed in a single season, OR 9+ combined, OR soft tissue
  - Bust signal: ECR overvalued (>3), OR 9+ missed in one season, OR 8+ combined,
                 OR declining trend AND age >= 29
"""
import pytest
from src.ingestion.orchestrator import (
    _calculate_injury_risk,
    _calculate_bust_signal,
    _calculate_trend,
    _calculate_sleeper_signal,
)


# --- injury risk ---

def test_injury_risk_low():
    assert _calculate_injury_risk(0, 1, None) == "low"

def test_injury_risk_medium():
    assert _calculate_injury_risk(2, 2, None) == "medium"

def test_injury_risk_high_combined():
    assert _calculate_injury_risk(4, 6, None) == "high"

def test_injury_risk_high_single_season_5_games():
    # Missing 5 games in 2025 alone → high (e.g. Bijan early exit)
    assert _calculate_injury_risk(0, 5, None) == "high"

def test_injury_risk_high_10_missed_single_season():
    # 10 games missed in one season → high risk
    assert _calculate_injury_risk(0, 10, None) == "high"

def test_injury_risk_high_15_missed_single_season():
    # 15 games missed in one season → high risk
    assert _calculate_injury_risk(0, 15, None) == "high"

def test_injury_risk_high_prior_season_injury():
    # 10 missed in 2024 (e.g. PCL), 0 in 2025 → still high via combined (10 > 9)
    assert _calculate_injury_risk(10, 0, None) == "high"

def test_injury_risk_high_soft_tissue():
    assert _calculate_injury_risk(0, 0, "hamstring") == "high"

def test_injury_risk_high_acl():
    assert _calculate_injury_risk(0, 2, "ACL") == "high"

def test_injury_risk_none_body_part_not_high():
    # No injury body part, 3 missed total → medium (not high)
    assert _calculate_injury_risk(2, 1, None) == "medium"


# --- bust signal ---

def test_bust_signal_ecr_overvalued():
    assert _calculate_bust_signal(4.0, 0, 0, "stable", 25) is True

def test_bust_signal_combined_missed_over_8():
    assert _calculate_bust_signal(None, 4, 6, "stable", 25) is True

def test_bust_signal_single_season_9_missed():
    # 15 missed in one season alone → bust
    assert _calculate_bust_signal(None, 0, 15, "stable", 26) is True

def test_bust_signal_single_season_9_missed_alt():
    # 13 missed in one season alone → bust
    assert _calculate_bust_signal(None, 0, 13, "stable", 24) is True

def test_bust_signal_prior_season_injury_age_29():
    # 10 missed in 2024 (e.g. PCL), healthy in 2025, age 29 → bust via combined + age
    assert _calculate_bust_signal(None, 10, 0, "declining", 29) is True

def test_bust_signal_declining_age_29_even_without_missed():
    # Age 29+ declining is a bust signal on its own
    assert _calculate_bust_signal(None, 0, 0, "declining", 29) is True

def test_bust_signal_declining_age_28_no_other_flags():
    # Age 28 declining without missed games or overvalued ECR → not a bust
    assert _calculate_bust_signal(None, 0, 0, "declining", 28) is False

def test_bust_signal_clean_young_player():
    assert _calculate_bust_signal(None, 0, 0, "stable", 24) is False

def test_bust_signal_threshold_exactly_8_combined_not_bust():
    # Exactly 8 combined is NOT > 8
    assert _calculate_bust_signal(None, 4, 4, "stable", 25) is False

def test_bust_signal_threshold_9_combined_is_bust():
    assert _calculate_bust_signal(None, 4, 5, "stable", 25) is True


# --- trend ---

def test_trend_improving():
    # Went from RB20 to RB5 — improved 15 spots
    assert _calculate_trend(20, 5) == "improving"

def test_trend_declining():
    # Went from RB3 to RB18 — declined 15 spots
    assert _calculate_trend(3, 18) == "declining"

def test_trend_stable():
    assert _calculate_trend(5, 8) == "stable"

def test_trend_unknown_missing_data():
    assert _calculate_trend(None, 5) == "unknown"
    assert _calculate_trend(5, None) == "unknown"


# --- sleeper signal ---

def test_sleeper_signal_true():
    # ECR much higher than ADP, improving, healthy season
    assert _calculate_sleeper_signal(-5.0, 20, 8, 16) is True

def test_sleeper_signal_false_not_enough_games():
    assert _calculate_sleeper_signal(-5.0, 20, 8, 10) is False

def test_sleeper_signal_false_not_improving():
    assert _calculate_sleeper_signal(-5.0, 8, 20, 16) is False

def test_sleeper_signal_false_ecr_not_undervalued():
    assert _calculate_sleeper_signal(-1.0, 20, 8, 16) is False

def test_sleeper_signal_false_none_ecr():
    assert _calculate_sleeper_signal(None, 20, 8, 16) is False
