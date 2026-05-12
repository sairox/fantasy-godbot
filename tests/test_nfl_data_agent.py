"""
Tests for nfl_data_agent helper functions.
External network calls (nfl.import_ngs_data, nfl.import_injuries) are mocked.
"""
import json
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd

from src.ingestion.nfl_data_agent import (
    _safe_val,
    _row_to_dict,
    hash_nfl_player,
    find_changed_nfl_players,
    build_nfl_data,
    save_nfl_data,
    load_nfl_data,
)


# ── _safe_val ────────────────────────────────────────────────────────────────

def test_safe_val_none():
    assert _safe_val(None) is None

def test_safe_val_nan():
    import math
    assert _safe_val(float("nan")) is None

def test_safe_val_int():
    assert _safe_val(42) == 42

def test_safe_val_float():
    assert _safe_val(3.14) == 3.14

def test_safe_val_string():
    assert _safe_val("hello") == "hello"

def test_safe_val_numpy_like():
    class FakeNumpy:
        def item(self):
            return 7
    assert _safe_val(FakeNumpy()) == 7


# ── _row_to_dict ─────────────────────────────────────────────────────────────

def test_row_to_dict_filters_cols():
    row = pd.Series({"a": 1, "b": 2, "c": 3})
    result = _row_to_dict(row, ["a", "c"])
    assert result == {"a": 1, "c": 3}
    assert "b" not in result

def test_row_to_dict_missing_col_skipped():
    row = pd.Series({"a": 1})
    result = _row_to_dict(row, ["a", "z"])
    assert "z" not in result

def test_row_to_dict_nan_becomes_none():
    row = pd.Series({"a": float("nan")})
    result = _row_to_dict(row, ["a"])
    assert result["a"] is None


# ── hash_nfl_player ───────────────────────────────────────────────────────────

def test_hash_nfl_player_deterministic():
    player = {"ypc_2025": 4.5, "ryoe_2025": 10.0, "rec_yards_2025": 300,
              "games_missed_2025": 2, "pass_yards_2025": None}
    assert hash_nfl_player(player) == hash_nfl_player(player)

def test_hash_nfl_player_changes_on_diff():
    a = {"ypc_2025": 4.5, "ryoe_2025": 10.0, "rec_yards_2025": None,
         "games_missed_2025": 0, "pass_yards_2025": None}
    b = {**a, "games_missed_2025": 3}
    assert hash_nfl_player(a) != hash_nfl_player(b)

def test_hash_nfl_player_ignores_irrelevant_fields():
    a = {"ypc_2025": 4.0, "ryoe_2025": None, "rec_yards_2025": None,
         "games_missed_2025": 0, "pass_yards_2025": None, "team": "KC"}
    b = {**a, "team": "SF"}
    # team is not in the watchlist, so hash must be equal
    assert hash_nfl_player(a) == hash_nfl_player(b)

def test_hash_nfl_player_returns_32_char_md5():
    result = hash_nfl_player({})
    assert len(result) == 32


# ── find_changed_nfl_players ──────────────────────────────────────────────────

def test_find_changed_new_player():
    old = []
    new = {"gsis-1": {"gsis_id": "gsis-1", "games_missed_2025": 0,
                       "ypc_2025": 4.0, "ryoe_2025": None,
                       "rec_yards_2025": None, "pass_yards_2025": None}}
    changed = find_changed_nfl_players(old, new)
    assert "gsis-1" in changed

def test_find_changed_unchanged_player():
    player = {"gsis_id": "g1", "games_missed_2025": 0, "ypc_2025": 4.0,
              "ryoe_2025": None, "rec_yards_2025": None, "pass_yards_2025": None}
    old = [player]
    new = {"g1": player}
    changed = find_changed_nfl_players(old, new)
    assert "g1" not in changed

def test_find_changed_modified_player():
    old_player = {"gsis_id": "g2", "games_missed_2025": 0, "ypc_2025": 4.0,
                  "ryoe_2025": None, "rec_yards_2025": None, "pass_yards_2025": None}
    new_player = {**old_player, "games_missed_2025": 5}
    old = [old_player]
    new = {"g2": new_player}
    changed = find_changed_nfl_players(old, new)
    assert "g2" in changed

def test_find_changed_skips_players_without_gsis():
    old = [{"no_gsis": True}]  # missing gsis_id
    new = {"g3": {"gsis_id": "g3", "games_missed_2025": 1, "ypc_2025": None,
                  "ryoe_2025": None, "rec_yards_2025": None, "pass_yards_2025": None}}
    # should not crash, g3 treated as new
    changed = find_changed_nfl_players(old, new)
    assert "g3" in changed


# ── save_nfl_data / load_nfl_data ─────────────────────────────────────────────

def test_save_and_load_roundtrip(tmp_path):
    data = {
        "g1": {"gsis_id": "g1", "ypc_2025": 4.5, "games_missed_2025": 2},
        "g2": {"gsis_id": "g2", "targets_2025": 120, "games_missed_2025": 0},
    }
    with patch("src.ingestion.nfl_data_agent.RAW_DATA_PATH", tmp_path):
        save_nfl_data(data)
        loaded = load_nfl_data()

    assert set(loaded.keys()) == {"g1", "g2"}
    assert loaded["g1"]["ypc_2025"] == 4.5
    assert loaded["g2"]["targets_2025"] == 120

def test_load_nfl_data_missing_file(tmp_path):
    with patch("src.ingestion.nfl_data_agent.RAW_DATA_PATH", tmp_path):
        result = load_nfl_data()
    assert result == {}


# ── build_nfl_data (integration-style with mocked nfl library) ───────────────

def _make_rush_df(season: int) -> pd.DataFrame:
    return pd.DataFrame([{
        "player_gsis_id": "g1",
        "player_display_name": "Test Back",
        "player_position": "RB",
        "team_abbr": "KC",
        "season": season,
        "week": 0,
        "rush_attempts": 250,
        "rush_yards": 1100,
        "avg_rush_yards": 4.4,
        "rush_touchdowns": 8,
        "rush_yards_over_expected": 150.0,
        "rush_yards_over_expected_per_att": 0.6,
        "efficiency": 0.82,
        "percent_attempts_gte_eight_defenders": 22.5,
        "expected_rush_yards": 950.0,
    }])


def _make_recv_df(season: int) -> pd.DataFrame:
    return pd.DataFrame([{
        "player_gsis_id": "g2",
        "player_display_name": "Test Receiver",
        "player_position": "WR",
        "team_abbr": "SF",
        "season": season,
        "week": 0,
        "targets": 140,
        "receptions": 100,
        "catch_percentage": 71.4,
        "yards": 1300,
        "rec_touchdowns": 9,
        "avg_separation": 2.8,
        "avg_cushion": 5.1,
        "avg_intended_air_yards": 10.2,
        "percent_share_of_intended_air_yards": 30.5,
        "avg_yac": 4.5,
        "avg_yac_above_expectation": 1.2,
    }])


def _make_pass_df(season: int) -> pd.DataFrame:
    return pd.DataFrame([{
        "player_gsis_id": "g3",
        "player_display_name": "Test QB",
        "player_position": "QB",
        "team_abbr": "BUF",
        "season": season,
        "week": 0,
        "attempts": 560,
        "completions": 380,
        "completion_percentage": 67.9,
        "pass_yards": 4500,
        "pass_touchdowns": 35,
        "interceptions": 10,
        "passer_rating": 108.0,
        "expected_completion_percentage": 65.0,
        "completion_percentage_above_expectation": 2.9,
        "aggressiveness": 14.5,
        "avg_time_to_throw": 2.55,
        "avg_intended_air_yards": 9.1,
    }])


def _make_injury_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"season": 2025, "week": 1, "gsis_id": "g1",
         "report_status": "Out", "report_primary_injury": "Hamstring"},
        {"season": 2025, "week": 2, "gsis_id": "g1",
         "report_status": "Out", "report_primary_injury": "Hamstring"},
        {"season": 2024, "week": 5, "gsis_id": "g2",
         "report_status": "Out", "report_primary_injury": "Knee"},
    ])


@pytest.fixture
def mock_nfl_lib():
    with patch("src.ingestion.nfl_data_agent.nfl") as mock_nfl:
        def import_ngs_data(stat_type, years):
            dfs = []
            for yr in years:
                if stat_type == "rushing":
                    dfs.append(_make_rush_df(yr))
                elif stat_type == "receiving":
                    dfs.append(_make_recv_df(yr))
                elif stat_type == "passing":
                    dfs.append(_make_pass_df(yr))
            return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

        mock_nfl.import_ngs_data.side_effect = import_ngs_data
        mock_nfl.import_injuries.return_value = _make_injury_df()
        yield mock_nfl


def test_build_nfl_data_has_rushing(mock_nfl_lib):
    result = build_nfl_data([2024, 2025])
    assert "g1" in result
    assert result["g1"]["rush_attempts_2025"] == 250
    assert result["g1"]["ypc_2025"] == 4.4
    assert result["g1"]["ryoe_2025"] == 150.0

def test_build_nfl_data_has_receiving(mock_nfl_lib):
    result = build_nfl_data([2024, 2025])
    assert "g2" in result
    assert result["g2"]["targets_2025"] == 140
    assert result["g2"]["catch_pct_2025"] == 71.4
    assert result["g2"]["air_yards_share_2025"] == 30.5
    assert result["g2"]["yac_above_expected_2025"] == 1.2

def test_build_nfl_data_has_passing(mock_nfl_lib):
    result = build_nfl_data([2024, 2025])
    assert "g3" in result
    assert result["g3"]["pass_yards_2025"] == 4500
    assert result["g3"]["cpoe_2025"] == 2.9
    assert result["g3"]["aggressiveness_2025"] == 14.5
    assert result["g3"]["time_to_throw_2025"] == 2.55

def test_build_nfl_data_games_missed(mock_nfl_lib):
    result = build_nfl_data([2024, 2025])
    # g1 was Out weeks 1 and 2 in 2025 → 2 games missed
    assert result["g1"]["games_missed_2025"] == 2
    # g1 has injury type from most recent Out week
    assert result["g1"]["injury_type_2025"] == "Hamstring"

def test_build_nfl_data_games_missed_2024(mock_nfl_lib):
    result = build_nfl_data([2024, 2025])
    # g2 was Out week 5 in 2024 → 1 game missed
    assert result["g2"]["games_missed_2024"] == 1
    assert result["g2"]["injury_type_2024"] == "Knee"

def test_build_nfl_data_merges_sources(mock_nfl_lib):
    # g1 appears in rushing; g2 in receiving; g3 in passing
    result = build_nfl_data([2024, 2025])
    assert "g1" in result
    assert "g2" in result
    assert "g3" in result

def test_build_nfl_data_empty_on_ngs_failure():
    with patch("src.ingestion.nfl_data_agent.nfl") as mock_nfl:
        mock_nfl.import_ngs_data.side_effect = Exception("network error")
        mock_nfl.import_injuries.return_value = _make_injury_df()
        result = build_nfl_data([2025])
    # Should not raise; rushing/receiving/passing all return {}
    assert isinstance(result, dict)
