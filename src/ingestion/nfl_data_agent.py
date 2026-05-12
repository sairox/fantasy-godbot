import json
import hashlib
import logging
from pathlib import Path

import nfl_data_py as nfl
import pandas as pd

logger = logging.getLogger(__name__)

RAW_DATA_PATH = Path(__file__).parent.parent.parent / "data" / "raw"

RUSH_COLS = [
    "player_gsis_id", "player_display_name", "player_position", "team_abbr",
    "rush_attempts", "rush_yards", "avg_rush_yards", "rush_touchdowns",
    "rush_yards_over_expected", "rush_yards_over_expected_per_att",
    "efficiency", "percent_attempts_gte_eight_defenders",
    "expected_rush_yards",
]

RECV_COLS = [
    "player_gsis_id", "player_display_name", "player_position", "team_abbr",
    "targets", "receptions", "catch_percentage", "yards", "rec_touchdowns",
    "avg_separation", "avg_cushion", "avg_intended_air_yards",
    "percent_share_of_intended_air_yards",
    "avg_yac", "avg_yac_above_expectation",
]

PASS_COLS = [
    "player_gsis_id", "player_display_name", "player_position", "team_abbr",
    "attempts", "completions", "completion_percentage", "pass_yards",
    "pass_touchdowns", "interceptions", "passer_rating",
    "expected_completion_percentage", "completion_percentage_above_expectation",
    "aggressiveness", "avg_time_to_throw", "avg_intended_air_yards",
]


def _safe_val(val):
    """Converts numpy types to plain Python scalars; None for NaN."""
    if val is None:
        return None
    try:
        import math
        if math.isnan(float(val)):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(val, "item"):
        return val.item()
    return val


def _row_to_dict(row: pd.Series, cols: list[str]) -> dict:
    return {c: _safe_val(row[c]) for c in cols if c in row.index}


def _norm_name(name: str) -> str:
    """Lowercases and strips common name suffixes for matching."""
    n = name.lower().strip()
    for suffix in [" jr.", " sr.", " iii", " iv", " ii", " jr", " sr", " v"]:
        if n.endswith(suffix):
            n = n[: -len(suffix)]
            break
    return n.strip()


def _fetch_ngs(stat_type: str, years: list[int]) -> pd.DataFrame:
    """Fetches NGS data for given stat type and filters to full-season rows (week=0)."""
    try:
        df = nfl.import_ngs_data(stat_type, years)
        return df[df["week"] == 0].copy()
    except Exception as e:
        logger.error(f"Failed to fetch NGS {stat_type} for {years}: {e}")
        return pd.DataFrame()


def fetch_rushing_stats(years: list[int] = [2024, 2025]) -> dict[str, dict]:
    """
    Returns NGS rushing stats keyed by gsis_id.
    Includes YPC, rush yards over expected, efficiency, % vs stacked box.
    """
    df = _fetch_ngs("rushing", years)
    if df.empty:
        return {}

    result: dict[str, dict] = {}
    for _, row in df.iterrows():
        gsis_id = str(row.get("player_gsis_id", ""))
        if not gsis_id:
            continue
        season = int(row["season"])
        suffix = f"_{season}"
        data = _row_to_dict(row, RUSH_COLS)

        if gsis_id not in result:
            result[gsis_id] = {
                "gsis_id": gsis_id,
                "player_display_name": str(row.get("player_display_name", "")),
            }

        result[gsis_id].update({
            f"rush_attempts{suffix}":          data.get("rush_attempts"),
            f"rush_yards{suffix}":             data.get("rush_yards"),
            f"ypc{suffix}":                    round(data["avg_rush_yards"], 3) if data.get("avg_rush_yards") else None,
            f"rush_tds{suffix}":               data.get("rush_touchdowns"),
            f"ryoe{suffix}":                   round(data["rush_yards_over_expected"], 1) if data.get("rush_yards_over_expected") else None,
            f"ryoe_per_att{suffix}":           round(data["rush_yards_over_expected_per_att"], 3) if data.get("rush_yards_over_expected_per_att") else None,
            f"rush_efficiency{suffix}":        round(data["efficiency"], 2) if data.get("efficiency") else None,
            f"pct_vs_8_defenders{suffix}":     round(data["percent_attempts_gte_eight_defenders"], 1) if data.get("percent_attempts_gte_eight_defenders") else None,
        })

    logger.info(f"Fetched NGS rushing stats for {len(result)} players ({years})")
    return result


def fetch_receiving_stats(years: list[int] = [2024, 2025]) -> dict[str, dict]:
    """
    Returns NGS receiving stats keyed by gsis_id.
    Includes separation, air yards share, YAC above expected.
    """
    df = _fetch_ngs("receiving", years)
    if df.empty:
        return {}

    result: dict[str, dict] = {}
    for _, row in df.iterrows():
        gsis_id = str(row.get("player_gsis_id", ""))
        if not gsis_id:
            continue
        season = int(row["season"])
        suffix = f"_{season}"
        data = _row_to_dict(row, RECV_COLS)

        if gsis_id not in result:
            result[gsis_id] = {
                "gsis_id": gsis_id,
                "player_display_name": str(row.get("player_display_name", "")),
            }

        result[gsis_id].update({
            f"targets{suffix}":             data.get("targets"),
            f"receptions{suffix}":          data.get("receptions"),
            f"rec_yards{suffix}":           data.get("yards"),
            f"rec_tds{suffix}":             data.get("rec_touchdowns"),
            f"catch_pct{suffix}":           round(data["catch_percentage"], 1) if data.get("catch_percentage") else None,
            f"avg_separation{suffix}":      round(data["avg_separation"], 2) if data.get("avg_separation") else None,
            f"air_yards_share{suffix}":     round(data["percent_share_of_intended_air_yards"], 1) if data.get("percent_share_of_intended_air_yards") else None,
            f"yac_above_expected{suffix}":  round(data["avg_yac_above_expectation"], 2) if data.get("avg_yac_above_expectation") else None,
            f"avg_yac{suffix}":             round(data["avg_yac"], 2) if data.get("avg_yac") else None,
        })

    logger.info(f"Fetched NGS receiving stats for {len(result)} players ({years})")
    return result


def fetch_passing_stats(years: list[int] = [2024, 2025]) -> dict[str, dict]:
    """
    Returns NGS passing stats keyed by gsis_id.
    Includes CPOE (completion % over expected), aggressiveness, time to throw.
    """
    df = _fetch_ngs("passing", years)
    if df.empty:
        return {}

    result: dict[str, dict] = {}
    for _, row in df.iterrows():
        gsis_id = str(row.get("player_gsis_id", ""))
        if not gsis_id:
            continue
        season = int(row["season"])
        suffix = f"_{season}"
        data = _row_to_dict(row, PASS_COLS)

        if gsis_id not in result:
            result[gsis_id] = {
                "gsis_id": gsis_id,
                "player_display_name": str(row.get("player_display_name", "")),
            }

        result[gsis_id].update({
            f"pass_attempts{suffix}":   data.get("attempts"),
            f"pass_yards{suffix}":      data.get("pass_yards"),
            f"pass_tds{suffix}":        data.get("pass_touchdowns"),
            f"interceptions{suffix}":   data.get("interceptions"),
            f"completion_pct{suffix}":  round(data["completion_percentage"], 1) if data.get("completion_percentage") else None,
            f"cpoe{suffix}":            round(data["completion_percentage_above_expectation"], 2) if data.get("completion_percentage_above_expectation") else None,
            f"aggressiveness{suffix}":  round(data["aggressiveness"], 1) if data.get("aggressiveness") else None,
            f"time_to_throw{suffix}":   round(data["avg_time_to_throw"], 2) if data.get("avg_time_to_throw") else None,
            f"passer_rating{suffix}":   round(data["passer_rating"], 1) if data.get("passer_rating") else None,
        })

    logger.info(f"Fetched NGS passing stats for {len(result)} players ({years})")
    return result


def fetch_games_missed(years: list[int] = [2024, 2025]) -> dict[str, dict]:
    """
    Computes accurate games missed per player per season from official
    NFL injury reports. A player missed a game if they were listed as 'Out'
    for that week. Keyed by gsis_id.
    """
    try:
        injuries = nfl.import_injuries(years)
    except Exception as e:
        logger.error(f"Failed to fetch injury data: {e}")
        return {}

    result: dict[str, dict] = {}

    for season in years:
        season_df = injuries[
            (injuries["season"] == season) &
            (injuries["report_status"] == "Out")
        ]
        missed = (
            season_df.groupby("gsis_id")["week"]
            .nunique()
            .reset_index()
            .rename(columns={"week": f"games_missed_{season}"})
        )

        # Also grab primary injury type for most recent Out week
        injury_types = (
            season_df.dropna(subset=["report_primary_injury"])
            .sort_values("week", ascending=False)
            .groupby("gsis_id")["report_primary_injury"]
            .first()
            .reset_index()
            .rename(columns={"report_primary_injury": f"injury_type_{season}"})
        )

        for _, row in missed.iterrows():
            gsis_id = str(row["gsis_id"])
            if gsis_id not in result:
                result[gsis_id] = {"gsis_id": gsis_id}
            result[gsis_id][f"games_missed_{season}"] = int(row[f"games_missed_{season}"])

        for _, row in injury_types.iterrows():
            gsis_id = str(row["gsis_id"])
            if gsis_id not in result:
                result[gsis_id] = {"gsis_id": gsis_id}
            result[gsis_id][f"injury_type_{season}"] = row[f"injury_type_{season}"]

    logger.info(f"Computed games missed for {len(result)} players ({years})")
    return result


def build_nfl_data(years: list[int] = [2024, 2025]) -> dict[str, dict]:
    """
    Fetches and merges all NGS stats and injury data into a single dict
    keyed by gsis_id. This is what gets merged into player documents.
    """
    logger.info(f"Building NFL data for years {years}...")

    rushing  = fetch_rushing_stats(years)
    receiving = fetch_receiving_stats(years)
    passing  = fetch_passing_stats(years)
    injuries = fetch_games_missed(years)

    merged: dict[str, dict] = {}

    for source in [rushing, receiving, passing, injuries]:
        for gsis_id, data in source.items():
            if gsis_id not in merged:
                merged[gsis_id] = {"gsis_id": gsis_id}
            merged[gsis_id].update({k: v for k, v in data.items() if k != "gsis_id"})

    logger.info(f"NFL data built for {len(merged)} players")
    return merged


def build_name_gsis_lookup(nfl_data: dict[str, dict]) -> dict[str, str]:
    """
    Builds a normalized_name -> gsis_id lookup from already-fetched NGS data.
    Used as fallback for Sleeper players whose gsis_id field is empty.
    """
    lookup: dict[str, str] = {}
    for gsis_id, data in nfl_data.items():
        name = data.get("player_display_name", "").strip()
        if name:
            lookup[_norm_name(name)] = gsis_id
    logger.info(f"Built name->gsis_id lookup with {len(lookup)} entries")
    return lookup


def save_name_gsis_lookup(lookup: dict[str, str]) -> None:
    RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
    file_path = RAW_DATA_PATH / "name_gsis_lookup.json"
    with open(file_path, "w") as f:
        json.dump(lookup, f)
    logger.info(f"Saved name->gsis_id lookup ({len(lookup)} entries)")


def load_name_gsis_lookup() -> dict[str, str]:
    file_path = RAW_DATA_PATH / "name_gsis_lookup.json"
    if not file_path.exists():
        return {}
    with open(file_path) as f:
        return json.load(f)


def hash_nfl_player(data: dict) -> str:
    """Hashes key performance fields for change detection."""
    watchlist = {
        "ypc_2025":           data.get("ypc_2025"),
        "ryoe_2025":          data.get("ryoe_2025"),
        "rec_yards_2025":     data.get("rec_yards_2025"),
        "games_missed_2025":  data.get("games_missed_2025"),
        "pass_yards_2025":    data.get("pass_yards_2025"),
    }
    content = json.dumps(watchlist, sort_keys=True)
    return hashlib.md5(content.encode()).hexdigest()


def find_changed_nfl_players(old: list[dict], new: dict[str, dict]) -> list[str]:
    """Returns list of gsis_ids whose stats changed since last fetch."""
    old_hashes = {p["gsis_id"]: hash_nfl_player(p) for p in old if "gsis_id" in p}
    changed = []
    for gsis_id, data in new.items():
        if gsis_id not in old_hashes or old_hashes[gsis_id] != hash_nfl_player(data):
            changed.append(gsis_id)
    return changed


def save_nfl_data(data: dict[str, dict]) -> None:
    """Saves NFL data to data/raw/nfl_data.json."""
    RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
    file_path = RAW_DATA_PATH / "nfl_data.json"
    with open(file_path, "w") as f:
        json.dump(list(data.values()), f, indent=2)
    logger.info(f"Saved NFL data for {len(data)} players to {file_path}")


def load_nfl_data() -> dict[str, dict]:
    """Loads previously saved NFL data, keyed by gsis_id."""
    file_path = RAW_DATA_PATH / "nfl_data.json"
    if not file_path.exists():
        return {}
    with open(file_path) as f:
        records = json.load(f)
    return {r["gsis_id"]: r for r in records if "gsis_id" in r}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data = build_nfl_data([2024, 2025])
    save_nfl_data(data)
    logger.info("Done.")
