"""
Tests for the nflreadpy-based FantasyPros agent (rankings via load_ff_rankings).
Network calls are mocked — no live nflverse fetches.
"""
from unittest.mock import MagicMock, patch

import pandas as pd

from src.ingestion.fantasypros_agent import (
    _normalize_name,
    _safe_int,
    _safe_float,
    fetch_rankings,
    fetch_stats,
    fetch_adp,
    hash_fp_player,
    find_changed_fp_players,
    save_rankings,
    load_existing_rankings,
)


# --- normalize_name tests ---

def test_normalize_name_lowercase():
    assert _normalize_name("Christian McCaffrey") == "christian mccaffrey"


def test_normalize_name_strips_jr():
    assert _normalize_name("Odell Beckham Jr.") == "odell beckham"


def test_normalize_name_strips_iii():
    assert _normalize_name("John Smith III") == "john smith"


def test_normalize_name_strips_sr():
    assert _normalize_name("Calvin Johnson Sr") == "calvin johnson"


# --- safe conversion tests ---

def test_safe_int_valid():
    assert _safe_int("42") == 42


def test_safe_int_rounds_float():
    assert _safe_int(3.77) == 4


def test_safe_int_invalid():
    assert _safe_int("N/A") is None


def test_safe_int_none():
    assert _safe_int(None) is None


def test_safe_float_valid():
    assert _safe_float("19.5") == 19.5


def test_safe_float_invalid():
    assert _safe_float("-") is None


# --- fetch_rankings with mocked load_ff_rankings ---

def _rankings_fixture_df() -> pd.DataFrame:
    rows = [
        # redraft overall
        {"page_type": "redraft-overall", "player": "Ja'Marr Chase", "pos": "WR",
         "team": "CIN", "ecr": 1.2, "best": 1, "worst": 3, "sd": 0.5,
         "scrape_date": "2026-06-01", "player_filename": "jamarr-chase", "id": 1},
        {"page_type": "redraft-overall", "player": "Bijan Robinson", "pos": "RB",
         "team": "ATL", "ecr": 2.1, "best": 1, "worst": 4, "sd": 0.7,
         "scrape_date": "2026-06-01", "player_filename": "bijan-robinson", "id": 2},
        # dynasty overall
        {"page_type": "dynasty-overall", "player": "Bijan Robinson", "pos": "RB",
         "team": "ATL", "ecr": 1.0, "best": 1, "worst": 2, "sd": 0.3,
         "scrape_date": "2026-06-01", "player_filename": "bijan-robinson", "id": 2},
        # position pages
        {"page_type": "redraft-rb", "player": "Bijan Robinson", "pos": "RB",
         "team": "ATL", "ecr": 1.0, "best": 1, "worst": 1, "sd": 0.1,
         "scrape_date": "2026-06-01", "player_filename": "bijan-robinson", "id": 2},
        {"page_type": "redraft-wr", "player": "Ja'Marr Chase", "pos": "WR",
         "team": "CIN", "ecr": 1.0, "best": 1, "worst": 1, "sd": 0.1,
         "scrape_date": "2026-06-01", "player_filename": "jamarr-chase", "id": 1},
    ]
    return pd.DataFrame(rows)


def _mock_nflr_rankings():
    loaded = MagicMock()
    loaded.to_pandas.return_value = _rankings_fixture_df()
    mock_nflr = MagicMock()
    mock_nflr.load_ff_rankings.return_value = loaded
    return mock_nflr


def test_fetch_rankings_merges_pages():
    with patch("src.ingestion.fantasypros_agent.nflr", _mock_nflr_rankings()):
        result = fetch_rankings()
    assert set(result.keys()) == {"ja'marr chase", "bijan robinson"}


def test_fetch_rankings_overall_rank_from_ecr():
    with patch("src.ingestion.fantasypros_agent.nflr", _mock_nflr_rankings()):
        result = fetch_rankings()
    chase = result["ja'marr chase"]
    assert chase["rank_half_ppr_2026"] == 1
    assert chase["adp_2025"] == 1.2  # ECR used as ADP proxy


def test_fetch_rankings_dynasty_rank():
    with patch("src.ingestion.fantasypros_agent.nflr", _mock_nflr_rankings()):
        result = fetch_rankings()
    assert result["bijan robinson"]["rank_dynasty_2026"] == 1


def test_fetch_rankings_position_ranks():
    with patch("src.ingestion.fantasypros_agent.nflr", _mock_nflr_rankings()):
        result = fetch_rankings()
    assert result["bijan robinson"]["pos_rank_half_ppr_2026"] == 1
    assert result["ja'marr chase"]["pos_rank_half_ppr_2026"] == 1


def test_fetch_rankings_empty_on_failure():
    mock_nflr = MagicMock()
    mock_nflr.load_ff_rankings.side_effect = Exception("network error")
    with patch("src.ingestion.fantasypros_agent.nflr", mock_nflr):
        result = fetch_rankings()
    assert result == {}


# --- intentional no-op stubs ---

def test_fetch_stats_is_noop():
    assert fetch_stats() == {}


def test_fetch_adp_is_noop():
    assert fetch_adp() == {}


# --- change detection ---

def test_hash_fp_player_deterministic():
    player = {"rank_half_ppr_2026": 5, "adp_2025": 6.2}
    assert hash_fp_player(player) == hash_fp_player(player)


def test_hash_fp_player_changes_on_rank_change():
    a = {"rank_half_ppr_2026": 5, "adp_2025": 6.2}
    b = {**a, "rank_half_ppr_2026": 8}
    assert hash_fp_player(a) != hash_fp_player(b)


def test_find_changed_fp_players_detects_new_and_modified():
    old = [{"name": "Bijan Robinson", "rank_half_ppr_2026": 2, "adp_2025": 2.1}]
    new = [
        {"name": "Bijan Robinson", "rank_half_ppr_2026": 1, "adp_2025": 2.1},  # modified
        {"name": "Ja'Marr Chase", "rank_half_ppr_2026": 2, "adp_2025": 1.2},   # new
    ]
    changed = find_changed_fp_players(old, new)
    changed_names = {p["name"] for p in changed}
    assert changed_names == {"Bijan Robinson", "Ja'Marr Chase"}


def test_find_changed_fp_players_skips_unchanged():
    player = {"name": "Bijan Robinson", "rank_half_ppr_2026": 2, "adp_2025": 2.1}
    assert find_changed_fp_players([player], [player]) == []


# --- save / load roundtrip ---

def test_save_and_load_rankings_roundtrip(tmp_path):
    rankings = {
        "bijan robinson": {"name": "Bijan Robinson", "rank_half_ppr_2026": 1},
    }
    with patch("src.ingestion.fantasypros_agent.RAW_DATA_PATH", tmp_path):
        save_rankings(rankings)
        loaded = load_existing_rankings()
    assert len(loaded) == 1
    assert loaded[0]["name"] == "Bijan Robinson"


def test_load_existing_rankings_missing_file(tmp_path):
    with patch("src.ingestion.fantasypros_agent.RAW_DATA_PATH", tmp_path):
        assert load_existing_rankings() == []
