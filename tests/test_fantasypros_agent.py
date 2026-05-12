"""
Tests for FantasyPros agent HTML parsing using fixture HTML.
"""
import pytest
from src.ingestion.fantasypros_agent import (
    _normalize_name,
    _safe_int,
    _safe_float,
    _parse_stats_page,
    _parse_adp_page,
    compute_value_signals,
)

# --- Fixture HTML for stats page ---
FIXTURE_STATS_HTML = """
<html><body>
<table id="data">
  <thead><tr><th>Rank</th><th>Player</th><th>G</th><th>PPG</th><th>FPTS</th></tr></thead>
  <tbody>
    <tr data-fp-id="16800">
      <td>1</td>
      <td><a class="player-name" href="#">Christian McCaffrey</a></td>
      <td>16</td>
      <td>19.5</td>
      <td>312.4</td>
    </tr>
    <tr data-fp-id="16801">
      <td>2</td>
      <td><a class="player-name" href="#">CeeDee Lamb</a></td>
      <td>17</td>
      <td>22.1</td>
      <td>375.8</td>
    </tr>
  </tbody>
</table>
</body></html>
"""

# --- Fixture HTML for ADP page ---
FIXTURE_ADP_HTML = """
<html><body>
<table id="data">
  <thead><tr><th>Rank</th><th>Player</th><th>ADP</th><th>ECR</th><th>Best</th><th>Worst</th><th>StdDev</th></tr></thead>
  <tbody>
    <tr data-fp-id="16800">
      <td>1</td>
      <td><a href="#">Christian McCaffrey</a></td>
      <td>1.2</td>
      <td>1.0</td>
      <td>1</td>
      <td>3</td>
      <td>0.5</td>
    </tr>
  </tbody>
</table>
</body></html>
"""


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


def test_safe_int_with_comma():
    assert _safe_int("1,234") == 1234


def test_safe_int_invalid():
    assert _safe_int("N/A") is None


def test_safe_int_none():
    assert _safe_int(None) is None


def test_safe_float_valid():
    assert _safe_float("19.5") == 19.5


def test_safe_float_invalid():
    assert _safe_float("-") is None


# --- stats page parsing ---

def test_parse_stats_page_player_count():
    result = _parse_stats_page(FIXTURE_STATS_HTML, "rb", 2025, "half")
    assert len(result) == 2


def test_parse_stats_page_cmc_present():
    result = _parse_stats_page(FIXTURE_STATS_HTML, "rb", 2025, "half")
    assert "christian mccaffrey" in result


def test_parse_stats_page_fpts():
    result = _parse_stats_page(FIXTURE_STATS_HTML, "rb", 2025, "half")
    cmc = result["christian mccaffrey"]
    assert cmc.get("fantasy_points_half_2025") == 312.4


def test_parse_stats_page_finish_rank():
    result = _parse_stats_page(FIXTURE_STATS_HTML, "rb", 2025, "half")
    cmc = result["christian mccaffrey"]
    assert cmc.get("finish_rank_half_2025") == 1


def test_parse_stats_page_games():
    result = _parse_stats_page(FIXTURE_STATS_HTML, "rb", 2025, "half")
    cmc = result["christian mccaffrey"]
    assert cmc.get("games_played_half_2025") == 16


# --- ADP page parsing ---

def test_parse_adp_page_player_count():
    result = _parse_adp_page(FIXTURE_ADP_HTML)
    assert len(result) == 1


def test_parse_adp_page_adp_value():
    result = _parse_adp_page(FIXTURE_ADP_HTML)
    cmc = result.get("christian mccaffrey", {})
    assert cmc.get("adp_2025") == 1.2


def test_parse_adp_page_ecr_value():
    result = _parse_adp_page(FIXTURE_ADP_HTML)
    cmc = result.get("christian mccaffrey", {})
    assert cmc.get("ecr_2025") == 1.0


# --- compute_value_signals ---

def test_value_signals_undervalued():
    player = {"adp_2025": 10.0, "ecr_2025": 5.0, "finish_rank_half_2025": 3}
    result = compute_value_signals(player.copy())
    assert result["ecr_vs_adp_2025"] == -5.0
    assert result["value_vs_adp_2025"] == -7.0


def test_value_signals_overvalued():
    player = {"adp_2025": 5.0, "ecr_2025": 12.0, "finish_rank_half_2025": 20}
    result = compute_value_signals(player.copy())
    assert result["ecr_vs_adp_2025"] == 7.0


def test_value_signals_none_when_missing():
    player = {"adp_2025": None, "ecr_2025": None, "finish_rank_half_2025": None}
    result = compute_value_signals(player.copy())
    assert result["ecr_vs_adp_2025"] is None
    assert result["value_vs_adp_2025"] is None
