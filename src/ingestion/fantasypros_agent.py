import httpx
import json
import hashlib
import logging
import time
import re
from pathlib import Path
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

RAW_DATA_PATH = Path(__file__).parent.parent.parent / "data" / "raw"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

RANKING_URLS = {
    "standard_2026": "https://www.fantasypros.com/nfl/rankings/cheatsheets.php",
    "half_ppr_2026": "https://www.fantasypros.com/nfl/rankings/half-point-ppr-cheatsheets.php",
    "ppr_2026": "https://www.fantasypros.com/nfl/rankings/ppr-cheatsheets.php",
    "dynasty_2026": "https://www.fantasypros.com/nfl/rankings/dynasty-overall.php",
}

STATS_URLS_2025 = {
    "qb_std": "https://www.fantasypros.com/nfl/stats/qb.php?year=2025&scoring=STD",
    "qb_half": "https://www.fantasypros.com/nfl/stats/qb.php?year=2025&scoring=HALF",
    "qb_ppr": "https://www.fantasypros.com/nfl/stats/qb.php?year=2025&scoring=PPR",
    "rb_std": "https://www.fantasypros.com/nfl/stats/rb.php?year=2025&scoring=STD",
    "rb_half": "https://www.fantasypros.com/nfl/stats/rb.php?year=2025&scoring=HALF",
    "rb_ppr": "https://www.fantasypros.com/nfl/stats/rb.php?year=2025&scoring=PPR",
    "wr_std": "https://www.fantasypros.com/nfl/stats/wr.php?year=2025&scoring=STD",
    "wr_half": "https://www.fantasypros.com/nfl/stats/wr.php?year=2025&scoring=HALF",
    "wr_ppr": "https://www.fantasypros.com/nfl/stats/wr.php?year=2025&scoring=PPR",
    "te_std": "https://www.fantasypros.com/nfl/stats/te.php?year=2025&scoring=STD",
    "te_half": "https://www.fantasypros.com/nfl/stats/te.php?year=2025&scoring=HALF",
    "te_ppr": "https://www.fantasypros.com/nfl/stats/te.php?year=2025&scoring=PPR",
}

STATS_URLS_2024 = {
    "qb_half": "https://www.fantasypros.com/nfl/stats/qb.php?year=2024&scoring=HALF",
    "rb_half": "https://www.fantasypros.com/nfl/stats/rb.php?year=2024&scoring=HALF",
    "wr_half": "https://www.fantasypros.com/nfl/stats/wr.php?year=2024&scoring=HALF",
    "te_half": "https://www.fantasypros.com/nfl/stats/te.php?year=2024&scoring=HALF",
}

ADP_URL = "https://www.fantasypros.com/nfl/adp/overall.php"


def _fetch_with_retry(url: str, max_retries: int = 3) -> str:
    """Fetches URL content with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            with httpx.Client(headers=HEADERS, timeout=30.0, follow_redirects=True) as client:
                response = client.get(url)
                response.raise_for_status()
                return response.text
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                wait = 2 ** attempt * 5
                logger.warning(f"Rate limited on {url}, waiting {wait}s")
                time.sleep(wait)
            elif attempt == max_retries - 1:
                raise
            else:
                wait = 2 ** attempt
                logger.warning(f"HTTP error on {url} (attempt {attempt+1}): {e}, retrying in {wait}s")
                time.sleep(wait)
        except (httpx.RequestError, httpx.TimeoutException) as e:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
            logger.warning(f"Request error on {url} (attempt {attempt+1}): {e}, retrying in {wait}s")
            time.sleep(wait)
    return ""


def _extract_embedded_json(html: str, var_pattern: str) -> list:
    """Tries to extract JSON data embedded in a JS variable in the page."""
    pattern = rf'{var_pattern}\s*=\s*(\[.*?\]);'
    match = re.search(pattern, html, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    return []


def _normalize_name(name: str) -> str:
    """Lowercases and strips common suffixes for matching."""
    name = name.lower().strip()
    # longest suffixes first to avoid partial matches (e.g. " ii" inside " iii")
    for suffix in [" jr.", " sr.", " iii", " iv", " ii", " jr", " sr", " v"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name.strip()


def _safe_int(val) -> int | None:
    try:
        return int(str(val).replace(",", "").strip())
    except (ValueError, TypeError):
        return None


def _safe_float(val) -> float | None:
    try:
        return float(str(val).replace(",", "").strip())
    except (ValueError, TypeError):
        return None


def _parse_rankings_page(html: str, rank_key: str) -> dict:
    """
    Parses a FantasyPros cheatsheet/rankings page.
    Returns dict keyed by normalized player name with ranking data.
    """
    players = {}
    soup = BeautifulSoup(html, "html.parser")

    # Strategy 1: try embedded JSON (ecrData variable)
    json_data = _extract_embedded_json(html, r"var\s+ecrData")
    if not json_data:
        json_data = _extract_embedded_json(html, r"ecrData\s*=")

    if json_data:
        for item in json_data:
            if not isinstance(item, dict):
                continue
            name = item.get("player_name") or item.get("name", "")
            if not name:
                continue
            key = _normalize_name(name)
            players[key] = {
                "name": name,
                "fp_id": str(item.get("player_id", "")),
                "team": item.get("player_team_id", ""),
                "position": item.get("player_position_id", ""),
                rank_key: _safe_int(item.get("rank_ecr") or item.get("rank", 9999)),
            }
        if players:
            logger.info(f"Parsed {len(players)} players from embedded JSON for {rank_key}")
            return players

    # Strategy 2: parse HTML table rows with data-fp-id attribute
    rows = soup.select("table tbody tr[data-fp-id], table tbody tr[class*='mpb-available']")
    if not rows:
        rows = soup.select("table tbody tr")

    for rank_idx, row in enumerate(rows, start=1):
        fp_id = row.get("data-fp-id", "")
        cells = row.find_all("td")
        if len(cells) < 2:
            continue

        # name is typically in a link inside the row
        name_link = row.find("a", class_=re.compile(r"player-name|fp-player-name", re.I))
        if not name_link:
            name_link = row.find("a")
        name = name_link.get_text(strip=True) if name_link else cells[1].get_text(strip=True)

        # team/position from small tags or separate cells
        team_tag = row.find("small")
        team = team_tag.get_text(strip=True) if team_tag else ""

        pos_tag = row.find("span", class_=re.compile(r"position", re.I))
        position = pos_tag.get_text(strip=True) if pos_tag else ""

        rank_cell = cells[0].get_text(strip=True)
        rank = _safe_int(rank_cell) or rank_idx

        if not name:
            continue

        key = _normalize_name(name)
        players[key] = {
            "name": name,
            "fp_id": fp_id,
            "team": team,
            "position": position,
            rank_key: rank,
        }

    logger.info(f"Parsed {len(players)} players from HTML for {rank_key}")
    return players


def _parse_stats_page(html: str, position: str, year: int, scoring: str) -> dict:
    """
    Parses a FantasyPros stats page for a given position/year/scoring.
    Returns dict keyed by normalized name with stats.
    """
    players = {}
    soup = BeautifulSoup(html, "html.parser")

    table = soup.find("table", id="data") or soup.find("table", class_=re.compile(r"statistics|stats", re.I))
    if not table:
        table = soup.find("table")
    if not table:
        logger.warning(f"No stats table found for {position} {year} {scoring}")
        return players

    tbody = table.find("tbody")
    if not tbody:
        return players

    rows = tbody.find_all("tr")
    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 3:
            continue

        fp_id = row.get("data-fp-id", "")
        name_link = row.find("a", class_=re.compile(r"player-name|fp-player-name", re.I))
        if not name_link:
            name_link = cells[1].find("a") if len(cells) > 1 else None
        name = name_link.get_text(strip=True) if name_link else cells[1].get_text(strip=True)

        if not name:
            continue

        # rank is in first column
        rank_text = cells[0].get_text(strip=True)
        finish_rank = _safe_int(rank_text)

        # last column is typically total fantasy points
        # second-to-last is points per game, third-to-last is games
        # this varies by position; we grab the last few columns generically
        text_vals = [c.get_text(strip=True) for c in cells]

        # fantasy points usually last col, ppg second-to-last, games 3rd-to-last
        fpts = _safe_float(text_vals[-1]) if text_vals else None
        ppg = _safe_float(text_vals[-2]) if len(text_vals) > 1 else None
        games = _safe_int(text_vals[-3]) if len(text_vals) > 2 else None

        key = _normalize_name(name)
        suffix = f"_{scoring.lower()}_{year}"
        players[key] = {
            "name": name,
            "fp_id": fp_id,
            "position": position.upper(),
            f"fantasy_points{suffix}": fpts,
            f"points_per_game{suffix}": ppg,
            f"games_played{suffix}": games,
            f"finish_rank{suffix}": finish_rank,
        }

    logger.info(f"Parsed {len(players)} {position} players from stats ({year} {scoring})")
    return players


def _parse_adp_page(html: str) -> dict:
    """Parses the FantasyPros ADP page."""
    players = {}
    soup = BeautifulSoup(html, "html.parser")

    # Try embedded JSON first
    json_data = _extract_embedded_json(html, r"var\s+adpData")

    if json_data:
        for item in json_data:
            if not isinstance(item, dict):
                continue
            name = item.get("player_name", "")
            if not name:
                continue
            key = _normalize_name(name)
            players[key] = {
                "name": name,
                "fp_id": str(item.get("player_id", "")),
                "adp_2025": _safe_float(item.get("avg")),
                "ecr_2025": _safe_float(item.get("ecr")),
                "adp_best_2025": _safe_int(item.get("best")),
                "adp_worst_2025": _safe_int(item.get("worst")),
                "adp_std_dev_2025": _safe_float(item.get("std_dev")),
            }
        if players:
            return players

    # Fall back to HTML table
    table = soup.find("table", id="data") or soup.find("table")
    if not table:
        return players

    tbody = table.find("tbody")
    if not tbody:
        return players

    for row in tbody.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 4:
            continue

        fp_id = row.get("data-fp-id", "")
        name_link = row.find("a")
        name = name_link.get_text(strip=True) if name_link else cells[1].get_text(strip=True)

        if not name:
            continue

        text_vals = [c.get_text(strip=True) for c in cells]

        key = _normalize_name(name)
        players[key] = {
            "name": name,
            "fp_id": fp_id,
            "adp_2025": _safe_float(text_vals[2]) if len(text_vals) > 2 else None,
            "ecr_2025": _safe_float(text_vals[3]) if len(text_vals) > 3 else None,
            "adp_best_2025": _safe_int(text_vals[4]) if len(text_vals) > 4 else None,
            "adp_worst_2025": _safe_int(text_vals[5]) if len(text_vals) > 5 else None,
            "adp_std_dev_2025": _safe_float(text_vals[6]) if len(text_vals) > 6 else None,
        }

    logger.info(f"Parsed {len(players)} players from ADP page")
    return players


def fetch_rankings() -> dict:
    """
    Fetches all FantasyPros ranking pages and merges into a single dict
    keyed by normalized player name.
    """
    merged: dict[str, dict] = {}

    for key, url in RANKING_URLS.items():
        logger.info(f"Fetching rankings: {key} from {url}")
        try:
            html = _fetch_with_retry(url)
            rank_key = f"rank_{key}"
            page_data = _parse_rankings_page(html, rank_key)
            for norm_name, player in page_data.items():
                if norm_name not in merged:
                    merged[norm_name] = {
                        "name": player["name"],
                        "fp_id": player.get("fp_id", ""),
                        "team": player.get("team", ""),
                        "position": player.get("position", ""),
                    }
                merged[norm_name][rank_key] = player.get(rank_key)
        except Exception as e:
            logger.error(f"Failed to fetch rankings {key}: {e}")
        time.sleep(2)

    return merged


def fetch_stats() -> dict:
    """
    Fetches all FantasyPros stats pages (2024 + 2025) and merges into
    a single dict keyed by normalized player name.
    """
    merged: dict[str, dict] = {}

    # 2025 stats
    for key, url in STATS_URLS_2025.items():
        pos, scoring = key.rsplit("_", 1)
        logger.info(f"Fetching 2025 stats: {pos.upper()} {scoring}")
        try:
            html = _fetch_with_retry(url)
            page_data = _parse_stats_page(html, pos, 2025, scoring)
            for norm_name, player in page_data.items():
                if norm_name not in merged:
                    merged[norm_name] = {
                        "name": player["name"],
                        "fp_id": player.get("fp_id", ""),
                        "position": player.get("position", ""),
                    }
                merged[norm_name].update({
                    k: v for k, v in player.items()
                    if k not in ("name", "fp_id", "position")
                })
        except Exception as e:
            logger.error(f"Failed to fetch 2025 stats {key}: {e}")
        time.sleep(2)

    # 2024 stats (half PPR only for trend)
    for key, url in STATS_URLS_2024.items():
        pos, scoring = key.rsplit("_", 1)
        logger.info(f"Fetching 2024 stats: {pos.upper()} {scoring}")
        try:
            html = _fetch_with_retry(url)
            page_data = _parse_stats_page(html, pos, 2024, scoring)
            for norm_name, player in page_data.items():
                if norm_name not in merged:
                    merged[norm_name] = {
                        "name": player["name"],
                        "fp_id": player.get("fp_id", ""),
                        "position": player.get("position", ""),
                    }
                merged[norm_name].update({
                    k: v for k, v in player.items()
                    if k not in ("name", "fp_id", "position")
                })
        except Exception as e:
            logger.error(f"Failed to fetch 2024 stats {key}: {e}")
        time.sleep(2)

    return merged


def fetch_adp() -> dict:
    """Fetches FantasyPros ADP page."""
    logger.info("Fetching ADP data")
    try:
        html = _fetch_with_retry(ADP_URL)
        return _parse_adp_page(html)
    except Exception as e:
        logger.error(f"Failed to fetch ADP: {e}")
        return {}


def compute_value_signals(player: dict) -> dict:
    """Computes ecr_vs_adp and value_vs_adp derived fields."""
    adp = player.get("adp_2025")
    ecr = player.get("ecr_2025")
    finish = player.get("finish_rank_half_2025")

    player["ecr_vs_adp_2025"] = round(ecr - adp, 2) if ecr and adp else None

    if finish and adp:
        player["value_vs_adp_2025"] = round(finish - adp, 2)
    else:
        player["value_vs_adp_2025"] = None

    return player


def hash_fp_player(player: dict) -> str:
    """Hashes a FantasyPros player record for change detection."""
    watchlist = {
        "rank_standard_2026": player.get("rank_rank_standard_2026"),
        "rank_half_ppr_2026": player.get("rank_rank_half_ppr_2026"),
        "rank_ppr_2026": player.get("rank_rank_ppr_2026"),
        "adp_2025": player.get("adp_2025"),
        "finish_rank_half_2025": player.get("finish_rank_half_2025"),
    }
    content = json.dumps(watchlist, sort_keys=True)
    return hashlib.md5(content.encode()).hexdigest()


def find_changed_fp_players(old_players: list, new_players: list) -> list:
    """Detects changed FantasyPros players using hash comparison."""
    old_hashes = {}
    for p in old_players:
        key = _normalize_name(p.get("name", ""))
        old_hashes[key] = hash_fp_player(p)

    changed = []
    for player in new_players:
        key = _normalize_name(player.get("name", ""))
        new_hash = hash_fp_player(player)
        if key not in old_hashes or old_hashes[key] != new_hash:
            changed.append(player)
    return changed


def save_rankings(rankings: dict) -> None:
    """Saves rankings data to data/raw/fantasypros_rankings.json."""
    RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
    file_path = RAW_DATA_PATH / "fantasypros_rankings.json"
    data = list(rankings.values())
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved {len(data)} player rankings to {file_path}")


def save_stats(stats: dict) -> None:
    """Saves stats data to data/raw/fantasypros_stats.json."""
    RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
    file_path = RAW_DATA_PATH / "fantasypros_stats.json"
    data = list(stats.values())
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved {len(data)} player stats to {file_path}")


def load_existing_rankings() -> list:
    """Loads previously saved rankings data."""
    file_path = RAW_DATA_PATH / "fantasypros_rankings.json"
    if not file_path.exists():
        return []
    with open(file_path) as f:
        return json.load(f)


def load_existing_stats() -> list:
    """Loads previously saved stats data."""
    file_path = RAW_DATA_PATH / "fantasypros_stats.json"
    if not file_path.exists():
        return []
    with open(file_path) as f:
        return json.load(f)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    logger.info("Fetching FantasyPros rankings...")
    rankings = fetch_rankings()
    save_rankings(rankings)

    logger.info("Fetching FantasyPros stats...")
    stats = fetch_stats()
    save_stats(stats)

    logger.info("Fetching ADP data...")
    adp = fetch_adp()

    # merge ADP into stats
    for norm_name, adp_data in adp.items():
        if norm_name in stats:
            stats[norm_name].update(adp_data)
        else:
            stats[norm_name] = adp_data
    save_stats(stats)

    logger.info("FantasyPros ingestion complete")
