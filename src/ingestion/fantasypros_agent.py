import json
import hashlib
import logging
from pathlib import Path

import nflreadpy as nflr

logger = logging.getLogger(__name__)

RAW_DATA_PATH = Path(__file__).parent.parent.parent / "data" / "raw"

# page_type values from load_ff_rankings(type="draft")
_PAGE_REDRAFT_OVERALL  = "redraft-overall"
_PAGE_DYNASTY_OVERALL  = "dynasty-overall"
_POS_PAGES = {
    "QB": "redraft-qb",
    "RB": "redraft-rb",
    "WR": "redraft-wr",
    "TE": "redraft-te",
}


def _normalize_name(name: str) -> str:
    """Lowercases and strips common suffixes for name-based matching."""
    name = name.lower().strip()
    for suffix in [" jr.", " sr.", " iii", " iv", " ii", " jr", " sr", " v"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name.strip()


def _safe_int(val) -> int | None:
    try:
        return int(round(float(val)))
    except (ValueError, TypeError):
        return None


def _safe_float(val) -> float | None:
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def fetch_rankings() -> dict:
    """
    Fetches FantasyPros consensus rankings via nflreadpy.load_ff_rankings().
    Data is sourced from nflverse's maintained pipeline — no scraping required.

    Returns a dict keyed by normalized player name with fields:
        name, fp_id, position, team,
        rank_standard_2026, rank_half_ppr_2026, rank_ppr_2026, rank_dynasty_2026,
        adp_2025, ecr_2025, adp_best_2025, adp_worst_2025, adp_std_dev_2025
    """
    try:
        df = nflr.load_ff_rankings(type="draft").to_pandas()
    except Exception as e:
        logger.error("load_ff_rankings failed: %s", e)
        return {}

    merged: dict[str, dict] = {}

    # --- Redraft overall: ECR used for all three scoring formats as best proxy ---
    redraft = df[df["page_type"] == _PAGE_REDRAFT_OVERALL].copy()
    redraft = redraft.sort_values("ecr")
    logger.info("Loaded %d redraft-overall rankings (scraped %s)",
                len(redraft), redraft["scrape_date"].iloc[0] if len(redraft) else "?")

    for row in redraft.to_dict("records"):
        name = str(row.get("player", "") or "").strip()
        if not name:
            continue
        key = _normalize_name(name)
        ecr = _safe_float(row.get("ecr"))
        rank = _safe_int(ecr) if ecr is not None else None

        merged[key] = {
            "name":               name,
            "fp_id":              str(row.get("player_filename") or row.get("id") or ""),
            "position":           str(row.get("pos") or ""),
            "team":               str(row.get("team") or ""),
            "rank_standard_2026": rank,
            "rank_half_ppr_2026": rank,
            "rank_ppr_2026":      rank,
            "rank_dynasty_2026":  None,
            # ADP fields — use ECR as proxy (highly correlated in draft settings)
            "adp_2025":           round(ecr, 2) if ecr is not None else None,
            "ecr_2025":           ecr,
            "adp_best_2025":      _safe_int(row.get("best")),
            "adp_worst_2025":     _safe_int(row.get("worst")),
            "adp_std_dev_2025":   _safe_float(row.get("sd")),
        }

    # --- Dynasty overall: only updates dynasty rank ---
    dynasty = df[df["page_type"] == _PAGE_DYNASTY_OVERALL].copy()
    dynasty = dynasty.sort_values("ecr")
    logger.info("Loaded %d dynasty-overall rankings", len(dynasty))

    for row in dynasty.to_dict("records"):
        name = str(row.get("player", "") or "").strip()
        if not name:
            continue
        key = _normalize_name(name)
        rank = _safe_int(_safe_float(row.get("ecr")))
        if key in merged:
            merged[key]["rank_dynasty_2026"] = rank
        else:
            merged[key] = {
                "name":               name,
                "fp_id":              str(row.get("player_filename") or row.get("id") or ""),
                "position":           str(row.get("pos") or ""),
                "team":               str(row.get("team") or ""),
                "rank_standard_2026": None,
                "rank_half_ppr_2026": None,
                "rank_ppr_2026":      None,
                "rank_dynasty_2026":  rank,
                "adp_2025":           None,
                "ecr_2025":           None,
                "adp_best_2025":      None,
                "adp_worst_2025":     None,
                "adp_std_dev_2025":   None,
            }

    # --- Position-specific pages: extract position rank (RB1, WR3, etc.) ---
    for pos, page_type in _POS_PAGES.items():
        pos_page = df[df["page_type"] == page_type].sort_values("ecr")
        for pos_rank, row in enumerate(pos_page.to_dict("records"), start=1):
            name = str(row.get("player", "") or "").strip()
            if not name:
                continue
            key = _normalize_name(name)
            if key in merged:
                merged[key]["pos_rank_half_ppr_2026"] = pos_rank
        logger.info("Loaded %d %s position ranks", len(pos_page), pos)

    logger.info("fetch_rankings: %d total players", len(merged))
    return merged


def fetch_stats() -> dict:
    """
    Historical per-player stats (fantasy points, finish ranks, games played)
    are now sourced from nflreadpy.load_player_stats() in nfl_data_agent.py.
    This stub exists so the orchestrator import does not break.
    """
    logger.info("fetch_stats: stats now come from nflreadpy — returning empty dict")
    return {}


def fetch_adp() -> dict:
    """
    ADP is now embedded in fetch_rankings() output (ecr used as proxy).
    This stub exists so the orchestrator import does not break.
    """
    logger.info("fetch_adp: ADP embedded in rankings — returning empty dict")
    return {}


def hash_fp_player(player: dict) -> str:
    """Hashes a FantasyPros player record for change detection."""
    watchlist = {
        "rank_standard_2026":  player.get("rank_standard_2026"),
        "rank_half_ppr_2026":  player.get("rank_half_ppr_2026"),
        "rank_ppr_2026":       player.get("rank_ppr_2026"),
        "rank_dynasty_2026":   player.get("rank_dynasty_2026"),
        "adp_2025":            player.get("adp_2025"),
    }
    content = json.dumps(watchlist, sort_keys=True)
    return hashlib.md5(content.encode()).hexdigest()


def find_changed_fp_players(old_players: list, new_players: list) -> list:
    """Detects changed FantasyPros players using hash comparison."""
    old_hashes = {
        _normalize_name(p.get("name", "")): hash_fp_player(p)
        for p in old_players
    }
    changed = []
    for player in new_players:
        key = _normalize_name(player.get("name", ""))
        if key not in old_hashes or old_hashes[key] != hash_fp_player(player):
            changed.append(player)
    return changed


def save_rankings(rankings: dict) -> None:
    """Saves rankings data to data/raw/fantasypros_rankings.json."""
    RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
    file_path = RAW_DATA_PATH / "fantasypros_rankings.json"
    data = list(rankings.values())
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Saved %d player rankings to %s", len(data), file_path)


def save_stats(stats: dict) -> None:
    """Saves stats data to data/raw/fantasypros_stats.json."""
    RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
    file_path = RAW_DATA_PATH / "fantasypros_stats.json"
    data = list(stats.values())
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Saved %d player stats to %s", len(data), file_path)


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
    rankings = fetch_rankings()
    save_rankings(rankings)
    logger.info("FantasyPros rankings ingestion complete (%d players)", len(rankings))
