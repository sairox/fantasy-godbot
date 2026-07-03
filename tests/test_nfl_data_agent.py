"""
Tests for the nflreadpy-based NFL data agent.
External calls (nflreadpy loaders, nfl_data_py injuries) are mocked.
"""
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.ingestion.nfl_data_agent import (
    _safe_val,
    _norm_name,
    _compute_age,
    hash_nfl_player,
    find_changed_nfl_players,
    build_nfl_data,
    build_name_gsis_lookup,
    save_nfl_data,
    load_nfl_data,
)


# ── _safe_val ────────────────────────────────────────────────────────────────

def test_safe_val_none():
    assert _safe_val(None) is None

def test_safe_val_nan():
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


# ── _norm_name / _compute_age ────────────────────────────────────────────────

def test_norm_name_strips_suffix():
    assert _norm_name("Kenneth Walker III") == "kenneth walker"

def test_norm_name_lowercases():
    assert _norm_name("Bijan Robinson") == "bijan robinson"

def test_compute_age_none():
    assert _compute_age(None) is None

def test_compute_age_invalid():
    assert _compute_age("not-a-date") is None

def test_compute_age_valid():
    age = _compute_age("2000-01-15")
    assert isinstance(age, int) and age >= 26  # sanity for any run date >= 2026


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
    assert hash_nfl_player(a) == hash_nfl_player(b)

def test_hash_nfl_player_returns_32_char_md5():
    assert len(hash_nfl_player({})) == 32


# ── find_changed_nfl_players ──────────────────────────────────────────────────

def test_find_changed_new_player():
    new = {"gsis-1": {"gsis_id": "gsis-1", "games_missed_2025": 0,
                      "ypc_2025": 4.0, "ryoe_2025": None,
                      "rec_yards_2025": None, "pass_yards_2025": None}}
    assert "gsis-1" in find_changed_nfl_players([], new)

def test_find_changed_unchanged_player():
    player = {"gsis_id": "g1", "games_missed_2025": 0, "ypc_2025": 4.0,
              "ryoe_2025": None, "rec_yards_2025": None, "pass_yards_2025": None}
    assert "g1" not in find_changed_nfl_players([player], {"g1": player})

def test_find_changed_modified_player():
    old_player = {"gsis_id": "g2", "games_missed_2025": 0, "ypc_2025": 4.0,
                  "ryoe_2025": None, "rec_yards_2025": None, "pass_yards_2025": None}
    new_player = {**old_player, "games_missed_2025": 5}
    assert "g2" in find_changed_nfl_players([old_player], {"g2": new_player})

def test_find_changed_skips_players_without_gsis():
    old = [{"no_gsis": True}]
    new = {"g3": {"gsis_id": "g3", "games_missed_2025": 1, "ypc_2025": None,
                  "ryoe_2025": None, "rec_yards_2025": None, "pass_yards_2025": None}}
    assert "g3" in find_changed_nfl_players(old, new)


# ── save / load roundtrip ─────────────────────────────────────────────────────

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

def test_load_nfl_data_missing_file(tmp_path):
    with patch("src.ingestion.nfl_data_agent.RAW_DATA_PATH", tmp_path):
        assert load_nfl_data() == {}


# ── build_name_gsis_lookup ────────────────────────────────────────────────────

def test_build_name_gsis_lookup():
    nfl_data = {
        "g1": {"gsis_id": "g1", "player_display_name": "Kenneth Walker III"},
        "g2": {"gsis_id": "g2", "player_display_name": "Bijan Robinson"},
        "g3": {"gsis_id": "g3", "player_display_name": ""},  # skipped
    }
    lookup = build_name_gsis_lookup(nfl_data)
    assert lookup["kenneth walker"] == "g1"
    assert lookup["bijan robinson"] == "g2"
    assert len(lookup) == 2


# ── build_nfl_data (integration-style with mocked loaders) ───────────────────

def _pl(df: pd.DataFrame) -> MagicMock:
    """Wraps a pandas DataFrame to mimic a polars frame with .to_pandas()."""
    m = MagicMock()
    m.to_pandas.return_value = df
    return m


def _make_roster_df(season: int) -> pd.DataFrame:
    return pd.DataFrame([{
        "gsis_id": "g1", "full_name": "Test Back", "position": "RB",
        "team": "KC", "birth_date": "2000-01-15", "years_exp": 4,
        "depth_chart_position": 1, "status": "ACT",
        "sleeper_id": "s1", "espn_id": "e1", "yahoo_id": "y1",
        "season": season, "week": 1,
    }])


def _make_seasonal_df(seasons: list[int]) -> pd.DataFrame:
    rows = []
    for season in seasons:
        rows.append({
            "player_id": "g1", "player_display_name": "Test Back",
            "season": season, "carries": 250, "rushing_yards": 1100,
            "rushing_tds": 8, "targets": 40, "receptions": 30,
            "receiving_yards": 250, "receiving_tds": 1,
            "fantasy_points": 200.0, "fantasy_points_ppr": 230.0,
            "games": 16, "target_share": 0.12, "wopr": 0.35,
        })
    return pd.DataFrame(rows)


def _make_rush_df(seasons: list[int]) -> pd.DataFrame:
    rows = []
    for season in seasons:
        rows.append({
            "player_gsis_id": "g1", "player_display_name": "Test Back",
            "season": season, "week": 0, "rush_attempts": 250,
            "rush_yards": 1100, "avg_rush_yards": 4.4, "rush_touchdowns": 8,
            "rush_yards_over_expected": 150.0,
            "rush_yards_over_expected_per_att": 0.6, "efficiency": 0.82,
            "percent_attempts_gte_eight_defenders": 22.5,
        })
    return pd.DataFrame(rows)


def _make_recv_df(seasons: list[int]) -> pd.DataFrame:
    rows = []
    for season in seasons:
        rows.append({
            "player_gsis_id": "g2", "player_display_name": "Test Receiver",
            "season": season, "week": 0, "targets": 140, "receptions": 100,
            "yards": 1300, "rec_touchdowns": 9, "catch_percentage": 71.4,
            "avg_separation": 2.8, "percent_share_of_intended_air_yards": 30.5,
            "avg_yac_above_expectation": 1.2, "avg_yac": 4.5,
        })
    return pd.DataFrame(rows)


def _make_pass_df(seasons: list[int]) -> pd.DataFrame:
    rows = []
    for season in seasons:
        rows.append({
            "player_gsis_id": "g3", "player_display_name": "Test QB",
            "season": season, "week": 0, "attempts": 560, "pass_yards": 4500,
            "pass_touchdowns": 35, "interceptions": 10,
            "completion_percentage": 67.9,
            "completion_percentage_above_expectation": 2.9,
            "aggressiveness": 14.5, "avg_time_to_throw": 2.55,
            "passer_rating": 108.0,
        })
    return pd.DataFrame(rows)


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
def mock_data_sources():
    with patch("src.ingestion.nfl_data_agent.nflr") as mock_nflr, \
         patch("src.ingestion.nfl_data_agent.nfl") as mock_nfl:

        mock_nflr.load_rosters_weekly.side_effect = \
            lambda years: _pl(_make_roster_df(years[0]))
        mock_nflr.load_player_stats.side_effect = \
            lambda years, summary_level: _pl(_make_seasonal_df(years))

        def load_ngs(years, stat_type):
            makers = {"rushing": _make_rush_df, "receiving": _make_recv_df,
                      "passing": _make_pass_df}
            return _pl(makers[stat_type](years))

        mock_nflr.load_nextgen_stats.side_effect = load_ngs
        mock_nfl.import_injuries.return_value = _make_injury_df()
        yield


def test_build_nfl_data_has_rushing(mock_data_sources):
    result = build_nfl_data([2024, 2025])
    assert result["g1"]["rush_attempts_2025"] == 250
    assert result["g1"]["ypc_2025"] == 4.4
    assert result["g1"]["ryoe_2025"] == 150.0

def test_build_nfl_data_has_receiving(mock_data_sources):
    result = build_nfl_data([2024, 2025])
    assert result["g2"]["targets_2025"] == 140
    assert result["g2"]["catch_pct_2025"] == 71.4
    assert result["g2"]["air_yards_share_2025"] == 30.5

def test_build_nfl_data_has_passing(mock_data_sources):
    result = build_nfl_data([2024, 2025])
    assert result["g3"]["pass_yards_2025"] == 4500
    assert result["g3"]["cpoe_2025"] == 2.9
    assert result["g3"]["time_to_throw_2025"] == 2.55

def test_build_nfl_data_seasonal_fills_gaps_only(mock_data_sources):
    result = build_nfl_data([2024, 2025])
    # NGS rushing (250 att) and seasonal (250 carries) agree here, but the
    # merged fantasy points can only come from seasonal
    assert result["g1"]["fantasy_points_half_ppr_2025"] == 215.0  # (200+230)/2
    assert result["g1"]["target_share_2025"] == 12.0  # 0.12 → 12.0%

def test_build_nfl_data_games_missed(mock_data_sources):
    result = build_nfl_data([2024, 2025])
    assert result["g1"]["games_missed_2025"] == 2
    assert result["g1"]["injury_type_2025"] == "Hamstring"

def test_build_nfl_data_games_missed_2024(mock_data_sources):
    result = build_nfl_data([2024, 2025])
    assert result["g2"]["games_missed_2024"] == 1
    assert result["g2"]["injury_type_2024"] == "Knee"

def test_build_nfl_data_roster_metadata(mock_data_sources):
    result = build_nfl_data([2024, 2025])
    assert result["g1"]["position"] == "RB"
    assert result["g1"]["team"] == "KC"
    assert isinstance(result["g1"]["age"], int)

def test_build_nfl_data_survives_source_failure():
    with patch("src.ingestion.nfl_data_agent.nflr") as mock_nflr, \
         patch("src.ingestion.nfl_data_agent.nfl") as mock_nfl:
        mock_nflr.load_rosters_weekly.side_effect = Exception("network error")
        mock_nflr.load_player_stats.side_effect = Exception("network error")
        mock_nflr.load_nextgen_stats.side_effect = Exception("network error")
        mock_nfl.import_injuries.return_value = _make_injury_df()
        result = build_nfl_data([2025])
    # Injuries still merge; nothing raises
    assert isinstance(result, dict)
    assert result["g1"]["games_missed_2025"] == 2
