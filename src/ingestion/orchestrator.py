import json
import logging
from pathlib import Path

from src.ingestion.sleeper_agent import (
    fetch_players,
    filter_players,
    find_changed_players,
    save_raw_data,
    load_existing_players,
)
from src.ingestion.fantasypros_agent import (
    fetch_rankings,
    fetch_stats,
    fetch_adp,
    find_changed_fp_players,
    save_rankings,
    save_stats,
    load_existing_rankings,
    load_existing_stats,
    _normalize_name,
)

logger = logging.getLogger(__name__)

PROCESSED_PATH = Path(__file__).parent.parent.parent / "data" / "processed"
RAW_PATH = Path(__file__).parent.parent.parent / "data" / "raw"


def _load_sleeper_players() -> list[dict]:
    """Loads cached Sleeper players or returns empty list."""
    path = RAW_PATH / "sleeper_players.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def _calculate_injury_risk(games_missed_2024: int, games_missed_2025: int,
                            injury_body_part_2025: str | None) -> str:
    """Scores injury risk as low/medium/high."""
    missed_25 = games_missed_2025 or 0
    missed_24 = games_missed_2024 or 0
    total_missed = missed_24 + missed_25
    soft_tissue_2025 = bool(injury_body_part_2025) and any(
        part in (injury_body_part_2025 or "").lower()
        for part in ["hamstring", "acl", "pcl", "achilles", "knee", "quad"]
    )

    # Missing 5+ games in a single season is immediately high risk
    if missed_25 >= 5 or total_missed >= 9 or soft_tissue_2025:
        return "high"
    elif total_missed >= 3:
        return "medium"
    return "low"


def _calculate_trend(finish_2024: int | None, finish_2025: int | None) -> str:
    """Determines performance trend from 2024 → 2025."""
    if not finish_2024 or not finish_2025:
        return "unknown"
    diff = finish_2024 - finish_2025  # positive = improved (lower rank number = better)
    if diff >= 10:
        return "improving"
    elif diff <= -10:
        return "declining"
    return "stable"


def _calculate_sleeper_signal(ecr_vs_adp: float | None, finish_2024: int | None,
                               finish_2025: int | None, games_2025: int | None) -> bool:
    """True if a player is undervalued relative to expert consensus."""
    if ecr_vs_adp is None:
        return False
    improving = (finish_2024 and finish_2025 and finish_2024 > finish_2025)
    return ecr_vs_adp < -3 and improving and (games_2025 or 0) >= 14


def _calculate_bust_signal(ecr_vs_adp: float | None, games_missed_2024: int,
                            games_missed_2025: int, trend: str, age: int | None) -> bool:
    """True if a player is likely being overdrafted."""
    if ecr_vs_adp is not None and ecr_vs_adp > 3:
        return True

    missed_25 = games_missed_2025 or 0
    missed_24 = games_missed_2024 or 0
    total_missed = missed_24 + missed_25

    # Missed more than half a season in 2025 alone is a bust signal regardless of 2024
    if missed_25 >= 9:
        return True
    # Combined 2-season missed games threshold
    if total_missed > 8:
        return True
    # Declining production AND entering the fragile side of a career
    if trend == "declining" and (age or 0) >= 29:
        return True
    return False


def _merge_player(sleeper: dict, fp_rankings: dict, fp_stats: dict) -> dict:
    """
    Merges a Sleeper player document with FantasyPros rankings and stats
    into a single rich document.
    """
    name = sleeper.get("full_name", "")
    norm = _normalize_name(name)

    rk = fp_rankings.get(norm, {})
    st = fp_stats.get(norm, {})

    games_played_2025 = st.get("games_played_half_2025")
    games_played_2024 = st.get("games_played_half_2024")
    games_missed_2025 = (17 - games_played_2025) if games_played_2025 else None
    games_missed_2024 = (17 - games_played_2024) if games_played_2024 else None

    adp = st.get("adp_2025")
    ecr = st.get("ecr_2025")
    ecr_vs_adp = round(ecr - adp, 2) if ecr and adp else None

    finish_2025 = st.get("finish_rank_half_2025")
    finish_2024 = st.get("finish_rank_half_2024")
    value_vs_adp = round(finish_2025 - adp, 2) if finish_2025 and adp else None

    trend = _calculate_trend(finish_2024, finish_2025)
    injury_risk = _calculate_injury_risk(
        games_missed_2024 or 0,
        games_missed_2025 or 0,
        sleeper.get("injury_body_part"),
    )
    sleeper_signal = _calculate_sleeper_signal(
        ecr_vs_adp, finish_2024, finish_2025, games_played_2025
    )
    bust_signal = _calculate_bust_signal(
        ecr_vs_adp,
        games_missed_2024 or 0,
        games_missed_2025 or 0,
        trend,
        sleeper.get("age"),
    )

    return {
        # identity
        "player_id": sleeper.get("player_id"),
        "fp_id": rk.get("fp_id") or st.get("fp_id", ""),
        "full_name": name,
        "position": sleeper.get("position"),
        "team": sleeper.get("team"),
        "age": sleeper.get("age"),
        "years_exp": sleeper.get("years_exp"),

        # current status
        "status": sleeper.get("status"),
        "depth_chart_order": sleeper.get("depth_chart_order"),
        "injury_status": sleeper.get("injury_status"),
        "injury_body_part": sleeper.get("injury_body_part"),
        "injury_start_date": sleeper.get("injury_start_date"),
        "practice_participation": sleeper.get("practice_participation"),

        # 2026 draft projections
        "rank_standard_2026": rk.get("rank_rank_standard_2026"),
        "rank_half_ppr_2026": rk.get("rank_rank_half_ppr_2026"),
        "rank_ppr_2026": rk.get("rank_rank_ppr_2026"),
        "rank_dynasty_2026": rk.get("rank_rank_dynasty_2026"),

        # 2025 ADP & value
        "adp_2025": adp,
        "ecr_2025": ecr,
        "ecr_vs_adp_2025": ecr_vs_adp,
        "adp_best_2025": st.get("adp_best_2025"),
        "adp_worst_2025": st.get("adp_worst_2025"),
        "adp_std_dev_2025": st.get("adp_std_dev_2025"),
        "value_vs_adp_2025": value_vs_adp,

        # 2025 performance
        "fantasy_points_std_2025": st.get("fantasy_points_std_2025"),
        "fantasy_points_half_ppr_2025": st.get("fantasy_points_half_2025"),
        "fantasy_points_ppr_2025": st.get("fantasy_points_ppr_2025"),
        "points_per_game_2025": st.get("points_per_game_half_2025"),
        "games_played_2025": games_played_2025,
        "games_missed_2025": games_missed_2025,
        "finish_rank_std_2025": st.get("finish_rank_std_2025"),
        "finish_rank_half_ppr_2025": finish_2025,
        "finish_rank_ppr_2025": st.get("finish_rank_ppr_2025"),

        # 2024 performance (trend)
        "fantasy_points_half_ppr_2024": st.get("fantasy_points_half_2024"),
        "finish_rank_half_ppr_2024": finish_2024,
        "games_played_2024": games_played_2024,
        "games_missed_2024": games_missed_2024,

        # calculated signals
        "injury_risk_score": injury_risk,
        "trend": trend,
        "sleeper_signal": sleeper_signal,
        "bust_signal": bust_signal,
    }


def build_player_documents(league_format: str = "redraft") -> list[dict]:
    """
    Merges Sleeper + FantasyPros data into rich player documents.
    Returns list of documents ready for embedding.
    """
    logger.info("Loading Sleeper player data...")
    sleeper_players = _load_sleeper_players()
    if not sleeper_players:
        logger.warning("No Sleeper data found — run sleeper_agent first")
        return []

    logger.info("Loading FantasyPros rankings...")
    rankings_list = load_existing_rankings()
    fp_rankings = {_normalize_name(p.get("name", "")): p for p in rankings_list}

    logger.info("Loading FantasyPros stats...")
    stats_list = load_existing_stats()
    fp_stats = {_normalize_name(p.get("name", "")): p for p in stats_list}

    logger.info(f"Merging {len(sleeper_players)} Sleeper players with FantasyPros data...")
    documents = []
    matched = 0
    for player in sleeper_players:
        doc = _merge_player(player, fp_rankings, fp_stats)
        doc["league_format"] = league_format
        documents.append(doc)
        norm = _normalize_name(player.get("full_name", ""))
        if norm in fp_rankings or norm in fp_stats:
            matched += 1

    logger.info(f"Merged {len(documents)} documents, {matched} with FantasyPros data")
    return documents


def save_player_documents(documents: list[dict]) -> None:
    """Saves merged player documents to data/processed/player_documents.json."""
    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
    file_path = PROCESSED_PATH / "player_documents.json"
    with open(file_path, "w") as f:
        json.dump(documents, f, indent=2)
    logger.info(f"Saved {len(documents)} player documents to {file_path}")


def load_player_documents() -> list[dict]:
    """Loads previously processed player documents."""
    file_path = PROCESSED_PATH / "player_documents.json"
    if not file_path.exists():
        return []
    with open(file_path) as f:
        return json.load(f)


def refresh_all(league_format: str = "redraft", force: bool = False) -> None:
    """
    Full pipeline refresh:
    1. Fetch fresh Sleeper data and detect changes
    2. Fetch fresh FantasyPros data
    3. Merge into player documents
    4. Update only changed players in Chroma (or all if force=True)
    5. Save updated raw and processed data
    """
    from src.vectorstore.chroma_store import build_vectorstore, update_players, get_collection_count

    logger.info("=== Starting full data refresh ===")

    # Step 1: Sleeper
    logger.info("Fetching fresh Sleeper data...")
    old_sleeper = _load_sleeper_players()
    raw_players = fetch_players()
    new_sleeper = filter_players(raw_players, league_format)
    changed_sleeper = find_changed_players(old_sleeper, new_sleeper) if not force else new_sleeper
    logger.info(f"Sleeper: {len(new_sleeper)} players, {len(changed_sleeper)} changed")
    save_raw_data(new_sleeper)

    # Step 2: FantasyPros rankings
    logger.info("Fetching fresh FantasyPros rankings...")
    try:
        new_rankings = fetch_rankings()
        old_rankings = load_existing_rankings()
        save_rankings(new_rankings)
    except Exception as e:
        logger.error(f"FantasyPros rankings fetch failed: {e} — using cached data")
        new_rankings = {_normalize_name(p.get("name", "")): p for p in load_existing_rankings()}

    # Step 3: FantasyPros stats + ADP
    logger.info("Fetching fresh FantasyPros stats and ADP...")
    try:
        new_stats = fetch_stats()
        adp_data = fetch_adp()
        for norm_name, adp in adp_data.items():
            if norm_name in new_stats:
                new_stats[norm_name].update(adp)
            else:
                new_stats[norm_name] = adp
        save_stats(new_stats)
    except Exception as e:
        logger.error(f"FantasyPros stats fetch failed: {e} — using cached data")
        new_stats = {_normalize_name(p.get("name", "")): p for p in load_existing_stats()}

    # Step 4: Merge all into player documents
    logger.info("Merging all data sources into player documents...")
    documents = build_player_documents(league_format)
    save_player_documents(documents)

    # Step 5: Update Chroma
    collection_count = get_collection_count()
    if collection_count == 0 or force:
        logger.info("Building vectorstore from scratch...")
        build_vectorstore(documents)
    else:
        # Only re-embed players whose Sleeper data changed
        changed_ids = {p["player_id"] for p in changed_sleeper}
        changed_docs = [d for d in documents if d.get("player_id") in changed_ids]
        logger.info(f"Updating {len(changed_docs)} changed players in Chroma...")
        update_players(changed_docs)

    logger.info("=== Refresh complete ===")
