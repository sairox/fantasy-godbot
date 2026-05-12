import json
import pytest
from src.ingestion.sleeper_agent import filter_players, hash_player, find_changed_players


MOCK_PLAYERS = {
    "4046": {
        "player_id": "4046",
        "full_name": "Christian McCaffrey",
        "fantasy_positions": ["RB"],
        "position": "RB",
        "team": "SF",
        "status": "Active",
        "depth_chart_order": 1,
        "injury_status": None,
        "injury_body_part": None,
        "injury_start_date": None,
        "practice_participation": None,
        "search_rank": 1,
        "age": 28,
        "years_exp": 8,
    },
    "9999": {
        "player_id": "9999",
        "full_name": "Irrelevant Player",
        "fantasy_positions": ["RB"],
        "position": "RB",
        "team": "NE",
        "status": "Active",
        "depth_chart_order": 1,
        "injury_status": None,
        "injury_body_part": None,
        "injury_start_date": None,
        "practice_participation": None,
        "search_rank": 9999999,
        "age": 25,
        "years_exp": 2,
    },
    "1111": {
        "player_id": "1111",
        "full_name": "Deep Depth Player",
        "fantasy_positions": ["WR"],
        "position": "WR",
        "team": "DAL",
        "status": "Active",
        "depth_chart_order": 6,
        "injury_status": None,
        "injury_body_part": None,
        "injury_start_date": None,
        "practice_participation": None,
        "search_rank": 500,
        "age": 24,
        "years_exp": 1,
    },
    "2222": {
        "player_id": "2222",
        "full_name": "Practice Squad Guy",
        "fantasy_positions": ["WR"],
        "position": "WR",
        "team": "GB",
        "status": "Practice Squad",
        "depth_chart_order": 5,
        "injury_status": None,
        "injury_body_part": None,
        "injury_start_date": None,
        "practice_participation": None,
        "search_rank": 800,
        "age": 23,
        "years_exp": 1,
    },
}


def test_filter_players_basic():
    result = filter_players(MOCK_PLAYERS, "redraft")
    ids = [p["player_id"] for p in result]
    assert "4046" in ids   # CMC: valid redraft player


def test_filter_excludes_search_rank_9999999():
    result = filter_players(MOCK_PLAYERS, "redraft")
    ids = [p["player_id"] for p in result]
    assert "9999" not in ids


def test_filter_excludes_deep_depth():
    result = filter_players(MOCK_PLAYERS, "redraft")
    ids = [p["player_id"] for p in result]
    assert "1111" not in ids


def test_filter_excludes_practice_squad_in_redraft():
    result = filter_players(MOCK_PLAYERS, "redraft")
    ids = [p["player_id"] for p in result]
    assert "2222" not in ids


def test_filter_includes_practice_squad_in_dynasty():
    result = filter_players(MOCK_PLAYERS, "dynasty")
    ids = [p["player_id"] for p in result]
    assert "2222" in ids


def test_hash_player_deterministic():
    player = filter_players(MOCK_PLAYERS, "redraft")[0]
    assert hash_player(player) == hash_player(player)


def test_hash_player_changes_on_team_change():
    players = filter_players(MOCK_PLAYERS, "redraft")
    cmc = next(p for p in players if p["player_id"] == "4046")

    original_hash = hash_player(cmc)
    modified = {**cmc, "team": "NYG"}
    modified_hash = hash_player(modified)

    assert original_hash != modified_hash


def test_find_changed_players_detects_new():
    new_players = filter_players(MOCK_PLAYERS, "redraft")
    changed = find_changed_players(old_players=[], new_players=new_players)
    assert len(changed) == len(new_players)


def test_find_changed_players_no_change():
    players = filter_players(MOCK_PLAYERS, "redraft")
    changed = find_changed_players(old_players=players, new_players=players)
    assert len(changed) == 0


def test_find_changed_players_detects_status_change():
    players = filter_players(MOCK_PLAYERS, "redraft")
    cmc = next(p for p in players if p["player_id"] == "4046")

    modified_players = [
        {**p, "status": "Injured Reserve"} if p["player_id"] == "4046" else p
        for p in players
    ]
    changed = find_changed_players(old_players=players, new_players=modified_players)
    changed_ids = [p["player_id"] for p in changed]
    assert "4046" in changed_ids
