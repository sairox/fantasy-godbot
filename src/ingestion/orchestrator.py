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
    save_rankings,
    save_stats,
    load_existing_rankings,
    load_existing_stats,
    _normalize_name,
)
from src.ingestion.nfl_data_agent import (
    build_nfl_data,
    save_nfl_data,
    load_nfl_data,
    find_changed_nfl_players,
    build_name_gsis_lookup,
    save_name_gsis_lookup,
    load_name_gsis_lookup,
    _norm_name,
    fetch_rosters,
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

    if missed_25 >= 5 or total_missed >= 9 or soft_tissue_2025:
        return "high"
    elif total_missed >= 3:
        return "medium"
    return "low"


def _calculate_trend(finish_2024: int | None, finish_2025: int | None) -> str:
    """Determines performance trend from 2024 → 2025."""
    if not finish_2024 or not finish_2025:
        return "unknown"
    diff = finish_2024 - finish_2025  # positive = improved (lower rank = better)
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

    if missed_25 >= 9:
        return True
    if total_missed > 8:
        return True
    if trend == "declining" and (age or 0) >= 29:
        return True
    return False


def _merge_player(sleeper: dict, fp_rankings: dict, fp_stats: dict,
                  nfl_data: dict[str, dict],
                  name_gsis_lookup: dict[str, str] | None = None) -> dict:
    """
    Merges Sleeper + FantasyPros + nfl-data-py into a single rich document.
    nfl_data is the authoritative source for games missed and advanced NGS stats.
    name_gsis_lookup is used as fallback when Sleeper gsis_id is missing.
    """
    name = sleeper.get("full_name", "")
    norm = _normalize_name(name)
    gsis_id = sleeper.get("gsis_id") or ""

    # Fallback: match by normalized name when Sleeper gsis_id is missing
    if not gsis_id and name_gsis_lookup:
        gsis_id = name_gsis_lookup.get(_norm_name(name), "")
        if gsis_id:
            logger.debug("gsis_id resolved by name for %s -> %s", name, gsis_id)

    rk = fp_rankings.get(norm, {})
    st = fp_stats.get(norm, {})
    ngs = nfl_data.get(gsis_id, {})

    # --- Games missed: nfl_data is authoritative; fall back to FantasyPros calc ---
    fp_games_played_2025 = st.get("games_played_half_2025")
    fp_games_played_2024 = st.get("games_played_half_2024")

    games_missed_2025 = (
        ngs.get("games_missed_2025")
        or ((17 - fp_games_played_2025) if fp_games_played_2025 else None)
    )
    games_missed_2024 = (
        ngs.get("games_missed_2024")
        or ((17 - fp_games_played_2024) if fp_games_played_2024 else None)
    )
    games_played_2025 = (
        (17 - games_missed_2025) if games_missed_2025 is not None
        else fp_games_played_2025
    )
    games_played_2024 = (
        (17 - games_missed_2024) if games_missed_2024 is not None
        else fp_games_played_2024
    )

    # --- ADP / ECR ---
    adp = st.get("adp_2025")
    ecr = st.get("ecr_2025")
    ecr_vs_adp = round(ecr - adp, 2) if ecr and adp else None

    finish_2025 = st.get("finish_rank_half_2025")
    finish_2024 = st.get("finish_rank_half_2024")
    value_vs_adp = round(finish_2025 - adp, 2) if finish_2025 and adp else None

    # --- Signals ---
    trend = _calculate_trend(finish_2024, finish_2025)
    injury_type_2025 = ngs.get("injury_type_2025") or sleeper.get("injury_body_part")
    injury_risk = _calculate_injury_risk(
        games_missed_2024 or 0,
        games_missed_2025 or 0,
        injury_type_2025,
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
        "gsis_id": gsis_id,
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
        "rank_standard_2026": rk.get("rank_standard_2026"),
        "rank_half_ppr_2026": rk.get("rank_half_ppr_2026"),
        "rank_ppr_2026": rk.get("rank_ppr_2026"),
        "rank_dynasty_2026": rk.get("rank_dynasty_2026"),

        # 2025 ADP & value
        "adp_2025": adp,
        "ecr_2025": ecr,
        "ecr_vs_adp_2025": ecr_vs_adp,
        "adp_best_2025": st.get("adp_best_2025"),
        "adp_worst_2025": st.get("adp_worst_2025"),
        "adp_std_dev_2025": st.get("adp_std_dev_2025"),
        "value_vs_adp_2025": value_vs_adp,

        # 2025 performance (FantasyPros where available, nflverse seasonal as fallback)
        "fantasy_points_std_2025": st.get("fantasy_points_std_2025") or ngs.get("fantasy_points_std_2025"),
        "fantasy_points_half_ppr_2025": st.get("fantasy_points_half_2025") or ngs.get("fantasy_points_half_ppr_2025"),
        "fantasy_points_ppr_2025": st.get("fantasy_points_ppr_2025") or ngs.get("fantasy_points_ppr_2025"),
        "points_per_game_2025": st.get("points_per_game_half_2025"),
        "games_played_2025": games_played_2025 or ngs.get("games_played_2025"),
        "games_missed_2025": games_missed_2025,
        "finish_rank_std_2025": st.get("finish_rank_std_2025"),
        "finish_rank_half_ppr_2025": finish_2025,
        "finish_rank_ppr_2025": st.get("finish_rank_ppr_2025"),
        "target_share_2025": ngs.get("target_share_2025"),
        "wopr_2025": ngs.get("wopr_2025"),

        # 2024 performance
        "fantasy_points_half_ppr_2024": st.get("fantasy_points_half_2024") or ngs.get("fantasy_points_half_ppr_2024"),
        "fantasy_points_std_2024": ngs.get("fantasy_points_std_2024"),
        "finish_rank_half_ppr_2024": finish_2024,
        "games_played_2024": games_played_2024 or ngs.get("games_played_2024"),
        "target_share_2024": ngs.get("target_share_2024"),
        "games_missed_2024": games_missed_2024,

        # NGS rushing stats — all positions (RBs, QBs, gadget WRs)
        "rush_attempts_2025":       ngs.get("rush_attempts_2025"),
        "rush_yards_2025":          ngs.get("rush_yards_2025"),
        "ypc_2025":                 ngs.get("ypc_2025"),
        "rush_tds_2025":            ngs.get("rush_tds_2025"),
        "ryoe_2025":                ngs.get("ryoe_2025"),
        "ryoe_per_att_2025":        ngs.get("ryoe_per_att_2025"),
        "rush_efficiency_2025":     ngs.get("rush_efficiency_2025"),
        "pct_vs_8_defenders_2025":  ngs.get("pct_vs_8_defenders_2025"),
        "rush_attempts_2024":       ngs.get("rush_attempts_2024"),
        "rush_yards_2024":          ngs.get("rush_yards_2024"),
        "ypc_2024":                 ngs.get("ypc_2024"),
        "rush_tds_2024":            ngs.get("rush_tds_2024"),
        "ryoe_2024":                ngs.get("ryoe_2024"),
        "ryoe_per_att_2024":        ngs.get("ryoe_per_att_2024"),
        "rush_efficiency_2024":     ngs.get("rush_efficiency_2024"),
        "pct_vs_8_defenders_2024":  ngs.get("pct_vs_8_defenders_2024"),

        # NGS receiving stats — all positions (RBs, WRs, TEs, pass-catching QBs)
        "targets_2025":             ngs.get("targets_2025"),
        "receptions_2025":          ngs.get("receptions_2025"),
        "rec_yards_2025":           ngs.get("rec_yards_2025"),
        "rec_tds_2025":             ngs.get("rec_tds_2025"),
        "catch_pct_2025":           ngs.get("catch_pct_2025"),
        "avg_separation_2025":      ngs.get("avg_separation_2025"),
        "air_yards_share_2025":     ngs.get("air_yards_share_2025"),
        "yac_above_expected_2025":  ngs.get("yac_above_expected_2025"),
        "avg_yac_2025":             ngs.get("avg_yac_2025"),
        "targets_2024":             ngs.get("targets_2024"),
        "receptions_2024":          ngs.get("receptions_2024"),
        "rec_yards_2024":           ngs.get("rec_yards_2024"),
        "rec_tds_2024":             ngs.get("rec_tds_2024"),
        "catch_pct_2024":           ngs.get("catch_pct_2024"),
        "avg_separation_2024":      ngs.get("avg_separation_2024"),
        "air_yards_share_2024":     ngs.get("air_yards_share_2024"),
        "yac_above_expected_2024":  ngs.get("yac_above_expected_2024"),
        "avg_yac_2024":             ngs.get("avg_yac_2024"),

        # NGS passing stats — QBs (and any dual-threat players in NGS passing data)
        "pass_attempts_2025":   ngs.get("pass_attempts_2025"),
        "pass_yards_2025":      ngs.get("pass_yards_2025"),
        "pass_tds_2025":        ngs.get("pass_tds_2025"),
        "interceptions_2025":   ngs.get("interceptions_2025"),
        "completion_pct_2025":  ngs.get("completion_pct_2025"),
        "cpoe_2025":            ngs.get("cpoe_2025"),
        "aggressiveness_2025":  ngs.get("aggressiveness_2025"),
        "time_to_throw_2025":   ngs.get("time_to_throw_2025"),
        "passer_rating_2025":   ngs.get("passer_rating_2025"),
        "pass_attempts_2024":   ngs.get("pass_attempts_2024"),
        "pass_yards_2024":      ngs.get("pass_yards_2024"),
        "pass_tds_2024":        ngs.get("pass_tds_2024"),
        "interceptions_2024":   ngs.get("interceptions_2024"),
        "completion_pct_2024":  ngs.get("completion_pct_2024"),
        "cpoe_2024":            ngs.get("cpoe_2024"),
        "aggressiveness_2024":  ngs.get("aggressiveness_2024"),
        "time_to_throw_2024":   ngs.get("time_to_throw_2024"),
        "passer_rating_2024":   ngs.get("passer_rating_2024"),

        # injury detail
        "injury_type_2025":    ngs.get("injury_type_2025"),
        "injury_type_2024":    ngs.get("injury_type_2024"),

        # calculated signals
        "injury_risk_score": injury_risk,
        "trend": trend,
        "sleeper_signal": sleeper_signal,
        "bust_signal": bust_signal,
    }


def _build_stub_players_from_nfl_data(nfl_data: dict[str, dict]) -> list[dict]:
    """
    Creates Sleeper-like player stubs from nfl_data (rosters + stats) when
    Sleeper is unavailable. Includes position/team/age from roster data.
    Only includes players with meaningful production.
    """
    stubs = []
    for gsis_id, data in nfl_data.items():
        name = data.get("player_display_name", "").strip()
        if not name:
            continue
        rush = (data.get("rush_attempts_2025") or 0) + (data.get("rush_attempts_2024") or 0)
        tgt  = (data.get("targets_2025") or 0)  + (data.get("targets_2024") or 0)
        pa   = (data.get("pass_attempts_2025") or 0) + (data.get("pass_attempts_2024") or 0)
        if rush < 50 and tgt < 30 and pa < 50:
            continue
        stubs.append({
            "player_id":            gsis_id,
            "full_name":            name,
            "gsis_id":              gsis_id,
            "position":             data.get("position"),
            "team":                 data.get("team"),
            "age":                  data.get("age"),
            "years_exp":            data.get("years_exp"),
            "status":               data.get("status") or "Active",
            "depth_chart_order":    data.get("depth_chart_order"),
            "injury_status":        None,
            "injury_body_part":     None,
            "injury_start_date":    None,
            "practice_participation": None,
            "search_rank":          None,
        })
    logger.info(f"Built {len(stubs)} player stubs from nfl_data")
    return stubs


def build_player_documents(league_format: str = "redraft") -> list[dict]:
    """
    Merges Sleeper + FantasyPros + nfl-data-py into rich player documents.
    Falls back to nfl_data stubs when Sleeper data is unavailable.
    """
    logger.info("Loading Sleeper player data...")
    sleeper_players = _load_sleeper_players()
    nfl_data = load_nfl_data()
    if not sleeper_players:
        logger.warning("No Sleeper data — building stubs from nfl_data")
        sleeper_players = _build_stub_players_from_nfl_data(nfl_data)
    if not sleeper_players:
        logger.warning("No player data available from any source")
        return []

    logger.info("Loading FantasyPros rankings...")
    rankings_list = load_existing_rankings()
    fp_rankings = {_normalize_name(p.get("name", "")): p for p in rankings_list}

    logger.info("Loading FantasyPros stats...")
    stats_list = load_existing_stats()
    fp_stats = {_normalize_name(p.get("name", "")): p for p in stats_list}

    name_gsis_lookup = load_name_gsis_lookup()

    direct = sum(1 for p in sleeper_players if p.get("gsis_id") in nfl_data)
    name_fallback = sum(
        1 for p in sleeper_players
        if not p.get("gsis_id")
        and name_gsis_lookup.get(_norm_name(p.get("full_name", ""))) in nfl_data
    )
    logger.info(
        f"NGS matched: {direct} by gsis_id, {name_fallback} by name fallback "
        f"/ {len(sleeper_players)} total"
    )

    logger.info(f"Merging {len(sleeper_players)} players...")
    documents = []
    fp_matched = 0
    for player in sleeper_players:
        doc = _merge_player(player, fp_rankings, fp_stats, nfl_data, name_gsis_lookup)
        doc["league_format"] = league_format
        documents.append(doc)
        norm = _normalize_name(player.get("full_name", ""))
        if norm in fp_rankings or norm in fp_stats:
            fp_matched += 1

    _fill_computed_rankings(documents)

    logger.info(f"Built {len(documents)} documents — {fp_matched} with FP data, "
                f"{direct + name_fallback} with NGS data")
    return documents


_REPLACEMENT_RANK = {"QB": 14, "RB": 36, "WR": 48, "TE": 14, "K": 14}


def _fill_computed_rankings(documents: list[dict]) -> None:
    """
    Computes VORP-based (Value Over Replacement Player) overall rankings from
    recent fantasy points (65% 2025, 35% 2024).  VORP normalises position
    scoring so elite RBs/WRs rank above fringe QBs, mirroring how actual
    fantasy drafts work.  Fills rank fields only when official FP data is absent.
    Modifies documents in-place.
    """
    def _raw_score(doc: dict) -> float:
        fp25 = doc.get("fantasy_points_half_ppr_2025") or 0
        fp24 = doc.get("fantasy_points_half_ppr_2024") or 0
        return fp25 * 0.65 + fp24 * 0.35

    # Build per-position replacement levels
    pos_scores: dict[str, list[float]] = {}
    for doc in documents:
        pos = doc.get("position") or "UNK"
        s = _raw_score(doc)
        if s > 0:
            pos_scores.setdefault(pos, []).append(s)

    replacement: dict[str, float] = {}
    for pos, scores in pos_scores.items():
        scores.sort(reverse=True)
        rep_idx = _REPLACEMENT_RANK.get(pos, 20) - 1
        replacement[pos] = scores[rep_idx] if len(scores) > rep_idx else (scores[-1] if scores else 0)

    # Compute VORP (negative = below replacement)
    def _vorp(doc: dict) -> float:
        pos = doc.get("position") or "UNK"
        s = _raw_score(doc)
        return s - replacement.get(pos, 0) if s > 0 else -9999

    scored = [(i, _vorp(doc)) for i, doc in enumerate(documents)]
    scored.sort(key=lambda x: x[1], reverse=True)

    has_fp_ranks = any(doc.get("rank_half_ppr_2026") for doc in documents)

    for computed_rank, (idx, v) in enumerate(scored, start=1):
        doc = documents[idx]
        doc["computed_rank_half_ppr"] = computed_rank if v > -9999 else 9999
        if not has_fp_ranks and v > -9999:
            if doc.get("rank_half_ppr_2026") is None:
                doc["rank_half_ppr_2026"] = computed_rank
            if doc.get("rank_standard_2026") is None:
                doc["rank_standard_2026"] = computed_rank
            if doc.get("rank_ppr_2026") is None:
                doc["rank_ppr_2026"] = computed_rank

    logger.info("Computed VORP rankings for %d players", len(documents))


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
    1. Sleeper: fetch fresh player data, detect changes
    2. FantasyPros: rankings + stats + ADP
    3. nfl-data-py: NGS stats + accurate injury/games-missed data
    4. Merge all into player documents
    5. Update Chroma (incremental or full rebuild)
    """
    from src.vectorstore.chroma_store import build_vectorstore, update_players, get_collection_count

    logger.info("=== Starting full data refresh ===")

    # Step 1: Sleeper
    logger.info("Fetching fresh Sleeper data...")
    old_sleeper = _load_sleeper_players()
    try:
        raw_players = fetch_players()
        new_sleeper = filter_players(raw_players, league_format)
        changed_sleeper = find_changed_players(old_sleeper, new_sleeper) if not force else new_sleeper
        logger.info(f"Sleeper: {len(new_sleeper)} players, {len(changed_sleeper)} changed")
        save_raw_data(new_sleeper)
    except Exception as e:
        logger.warning(f"Sleeper fetch failed ({e}) — using cached data ({len(old_sleeper)} players)")
        new_sleeper = old_sleeper
        changed_sleeper = old_sleeper if force else []

    # Step 2: FantasyPros rankings
    logger.info("Fetching fresh FantasyPros rankings...")
    try:
        new_rankings = fetch_rankings()
        if new_rankings:
            save_rankings(new_rankings)
        else:
            logger.warning("FantasyPros rankings returned empty — keeping cached data")
    except Exception as e:
        logger.error(f"FantasyPros rankings fetch failed: {e} — using cached data")

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
        if new_stats:
            save_stats(new_stats)
        else:
            logger.warning("FantasyPros stats returned empty — keeping cached data")
    except Exception as e:
        logger.error(f"FantasyPros stats fetch failed: {e} — using cached data")

    # Step 4: nfl-data-py (NGS + injuries)
    logger.info("Fetching fresh NFL data (NGS + injuries)...")
    try:
        old_nfl_data = load_nfl_data()
        new_nfl_data = build_nfl_data([2024, 2025])
        changed_gsis_ids = set(
            find_changed_nfl_players(list(old_nfl_data.values()), new_nfl_data)
        ) if not force else set(new_nfl_data.keys())
        save_nfl_data(new_nfl_data)
        lookup = build_name_gsis_lookup(new_nfl_data)
        save_name_gsis_lookup(lookup)
        logger.info(f"NFL data: {len(new_nfl_data)} players, {len(changed_gsis_ids)} changed, "
                    f"{len(lookup)} name->gsis entries")
    except Exception as e:
        logger.error(f"NFL data fetch failed: {e} — using cached data")
        changed_gsis_ids = set()

    # Step 5: Merge into player documents
    logger.info("Merging all sources into player documents...")
    documents = build_player_documents(league_format)
    save_player_documents(documents)

    # Step 6: Update Chroma
    collection_count = get_collection_count()
    if collection_count == 0 or force:
        logger.info("Building vectorstore from scratch...")
        build_vectorstore(documents)
    else:
        # Re-embed players whose Sleeper data OR NGS data changed
        changed_sleeper_ids = {p["player_id"] for p in changed_sleeper}
        changed_docs = [
            d for d in documents
            if d.get("player_id") in changed_sleeper_ids
            or d.get("gsis_id") in changed_gsis_ids
        ]
        logger.info(f"Updating {len(changed_docs)} changed players in Chroma...")
        update_players(changed_docs)

    logger.info("=== Refresh complete ===")
