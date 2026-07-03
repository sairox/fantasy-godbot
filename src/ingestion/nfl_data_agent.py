import json
import hashlib
import logging
from datetime import date
from pathlib import Path

import nflreadpy as nflr
import nfl_data_py as nfl  # retained only for fetch_games_missed (injury data)
import pandas as pd

logger = logging.getLogger(__name__)

RAW_DATA_PATH = Path(__file__).parent.parent.parent / "data" / "raw"


def _safe_val(val):
    """Converts numpy/polars types to plain Python scalars; None for NaN."""
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


def _norm_name(name: str) -> str:
    """Lowercases and strips common name suffixes for matching."""
    n = name.lower().strip()
    for suffix in [" jr.", " sr.", " iii", " iv", " ii", " jr", " sr", " v"]:
        if n.endswith(suffix):
            n = n[: -len(suffix)]
            break
    return n.strip()


def _compute_age(birth_date_str) -> int | None:
    """Computes age in years from a birth date string (YYYY-MM-DD)."""
    if not birth_date_str:
        return None
    try:
        bd = date.fromisoformat(str(birth_date_str)[:10])
        today = date.today()
        return today.year - bd.year - ((today.month, today.day) < (bd.month, bd.day))
    except Exception:
        return None


def fetch_rosters(years: list[int] = [2024, 2025]) -> dict[str, dict]:
    """
    Fetches weekly roster data from nflreadpy and collapses to the most
    recent entry per player.  Provides position, team, age, depth chart,
    status, and cross-platform IDs (Sleeper, ESPN, Yahoo, etc.).
    """
    frames = []
    for yr in sorted(years):
        try:
            frames.append(nflr.load_rosters_weekly([yr]).to_pandas())
        except Exception as e:
            logger.warning("Roster data unavailable for %s: %s", yr, e)
    if not frames:
        return {}

    df = pd.concat(frames, ignore_index=True)
    # Keep the latest week per player so newer team/status wins
    df = df.sort_values(["season", "week"], ascending=True)
    df = df.dropna(subset=["gsis_id"])
    df = df.drop_duplicates(subset=["gsis_id"], keep="last")

    result: dict[str, dict] = {}
    for row in df.to_dict("records"):
        gsis_id = str(row.get("gsis_id", "")).strip()
        if not gsis_id:
            continue
        name = str(row.get("full_name", "")).strip()
        if not name:
            continue
        result[gsis_id] = {
            "gsis_id":           gsis_id,
            "player_display_name": name,
            "position":          _safe_val(row.get("position")),
            "team":              _safe_val(row.get("team")),
            "age":               _compute_age(row.get("birth_date")),
            "years_exp":         _safe_val(row.get("years_exp")),
            "depth_chart_order": _safe_val(row.get("depth_chart_position")),
            "status":            _safe_val(row.get("status")),
            "sleeper_id":        _safe_val(row.get("sleeper_id")),
            "espn_id":           _safe_val(row.get("espn_id")),
            "yahoo_id":          _safe_val(row.get("yahoo_id")),
        }

    logger.info("Fetched rosters for %d players (%s)", len(result), years)
    return result


def fetch_seasonal_stats(years: list[int] = [2024, 2025]) -> dict[str, dict]:
    """
    Fetches regular-season totals from nflreadpy for all positions.
    Covers every position (including RBs that NGS receiving excludes),
    fantasy points in all formats, target_share and WOPR on the correct
    0-1 decimal scale, and games played.
    """
    try:
        df = nflr.load_player_stats(years, summary_level="reg").to_pandas()
    except Exception as e:
        logger.error("load_player_stats failed: %s", e)
        return {}

    result: dict[str, dict] = {}
    for row in df.to_dict("records"):
        gsis_id = str(row.get("player_id", "")).strip()
        if not gsis_id:
            continue
        season = int(row["season"])
        suffix = f"_{season}"

        if gsis_id not in result:
            result[gsis_id] = {
                "gsis_id": gsis_id,
                "player_display_name": str(row.get("player_display_name", "")),
            }

        fp_std  = _safe_val(row.get("fantasy_points"))
        fp_ppr  = _safe_val(row.get("fantasy_points_ppr"))
        fp_half = round((fp_std + fp_ppr) / 2, 1) if fp_std is not None and fp_ppr is not None else None

        tgt_sh = _safe_val(row.get("target_share"))   # already 0-1 decimal
        wopr   = _safe_val(row.get("wopr"))            # already 0-1 decimal

        result[gsis_id].update({
            f"rush_attempts{suffix}":           _safe_val(row.get("carries")),
            f"rush_yards{suffix}":              _safe_val(row.get("rushing_yards")),
            f"rush_tds{suffix}":                _safe_val(row.get("rushing_tds")),
            f"targets{suffix}":                 _safe_val(row.get("targets")),
            f"receptions{suffix}":              _safe_val(row.get("receptions")),
            f"rec_yards{suffix}":               _safe_val(row.get("receiving_yards")),
            f"rec_tds{suffix}":                 _safe_val(row.get("receiving_tds")),
            f"fantasy_points_std{suffix}":      fp_std,
            f"fantasy_points_ppr{suffix}":      fp_ppr,
            f"fantasy_points_half_ppr{suffix}": fp_half,
            f"games_played{suffix}":            _safe_val(row.get("games")),
            f"target_share{suffix}":            round(tgt_sh * 100, 1) if tgt_sh is not None else None,
            f"wopr{suffix}":                    round(float(wopr), 3) if wopr is not None else None,
        })

    logger.info("Fetched seasonal stats for %d players (%s)", len(result), years)
    return result


def fetch_rushing_stats(years: list[int] = [2024, 2025]) -> dict[str, dict]:
    """
    Returns NGS rushing stats keyed by gsis_id.
    Includes YPC, rush yards over expected, efficiency, % vs stacked box.
    """
    try:
        df = nflr.load_nextgen_stats(years, "rushing").to_pandas()
    except Exception as e:
        logger.error("load_nextgen_stats rushing failed: %s", e)
        return {}

    df = df[df["week"] == 0].copy()  # week=0 = full-season aggregate

    result: dict[str, dict] = {}
    for row in df.to_dict("records"):
        gsis_id = str(row.get("player_gsis_id", "")).strip()
        if not gsis_id:
            continue
        season = int(row["season"])
        suffix = f"_{season}"

        if gsis_id not in result:
            result[gsis_id] = {
                "gsis_id": gsis_id,
                "player_display_name": str(row.get("player_display_name", "")),
            }

        avg_rush = _safe_val(row.get("avg_rush_yards"))
        ryoe     = _safe_val(row.get("rush_yards_over_expected"))
        ryoe_att = _safe_val(row.get("rush_yards_over_expected_per_att"))
        eff      = _safe_val(row.get("efficiency"))
        pct8     = _safe_val(row.get("percent_attempts_gte_eight_defenders"))

        result[gsis_id].update({
            f"rush_attempts{suffix}":       _safe_val(row.get("rush_attempts")),
            f"rush_yards{suffix}":          _safe_val(row.get("rush_yards")),
            f"ypc{suffix}":                 round(avg_rush, 3) if avg_rush is not None else None,
            f"rush_tds{suffix}":            _safe_val(row.get("rush_touchdowns")),
            f"ryoe{suffix}":                round(ryoe, 1) if ryoe is not None else None,
            f"ryoe_per_att{suffix}":        round(ryoe_att, 3) if ryoe_att is not None else None,
            f"rush_efficiency{suffix}":     round(eff, 2) if eff is not None else None,
            f"pct_vs_8_defenders{suffix}":  round(pct8, 1) if pct8 is not None else None,
        })

    logger.info("Fetched NGS rushing stats for %d players (%s)", len(result), years)
    return result


def fetch_receiving_stats(years: list[int] = [2024, 2025]) -> dict[str, dict]:
    """
    Returns NGS receiving stats keyed by gsis_id.
    Note: NGS receiving covers WR/TE only — RB receiving comes from fetch_seasonal_stats.
    Includes separation, air yards share, YAC above expected.
    """
    try:
        df = nflr.load_nextgen_stats(years, "receiving").to_pandas()
    except Exception as e:
        logger.error("load_nextgen_stats receiving failed: %s", e)
        return {}

    df = df[df["week"] == 0].copy()

    result: dict[str, dict] = {}
    for row in df.to_dict("records"):
        gsis_id = str(row.get("player_gsis_id", "")).strip()
        if not gsis_id:
            continue
        season = int(row["season"])
        suffix = f"_{season}"

        if gsis_id not in result:
            result[gsis_id] = {
                "gsis_id": gsis_id,
                "player_display_name": str(row.get("player_display_name", "")),
            }

        catch_pct  = _safe_val(row.get("catch_percentage"))
        sep        = _safe_val(row.get("avg_separation"))
        air_sh     = _safe_val(row.get("percent_share_of_intended_air_yards"))
        yac_above  = _safe_val(row.get("avg_yac_above_expectation"))
        avg_yac    = _safe_val(row.get("avg_yac"))

        result[gsis_id].update({
            f"targets{suffix}":             _safe_val(row.get("targets")),
            f"receptions{suffix}":          _safe_val(row.get("receptions")),
            f"rec_yards{suffix}":           _safe_val(row.get("yards")),
            f"rec_tds{suffix}":             _safe_val(row.get("rec_touchdowns")),
            f"catch_pct{suffix}":           round(catch_pct, 1) if catch_pct is not None else None,
            f"avg_separation{suffix}":      round(sep, 2) if sep is not None else None,
            f"air_yards_share{suffix}":     round(air_sh, 1) if air_sh is not None else None,
            f"yac_above_expected{suffix}":  round(yac_above, 2) if yac_above is not None else None,
            f"avg_yac{suffix}":             round(avg_yac, 2) if avg_yac is not None else None,
        })

    logger.info("Fetched NGS receiving stats for %d players (%s)", len(result), years)
    return result


def fetch_passing_stats(years: list[int] = [2024, 2025]) -> dict[str, dict]:
    """
    Returns NGS passing stats keyed by gsis_id.
    Includes CPOE, aggressiveness, time to throw.
    """
    try:
        df = nflr.load_nextgen_stats(years, "passing").to_pandas()
    except Exception as e:
        logger.error("load_nextgen_stats passing failed: %s", e)
        return {}

    df = df[df["week"] == 0].copy()

    result: dict[str, dict] = {}
    for row in df.to_dict("records"):
        gsis_id = str(row.get("player_gsis_id", "")).strip()
        if not gsis_id:
            continue
        season = int(row["season"])
        suffix = f"_{season}"

        if gsis_id not in result:
            result[gsis_id] = {
                "gsis_id": gsis_id,
                "player_display_name": str(row.get("player_display_name", "")),
            }

        cpoe   = _safe_val(row.get("completion_percentage_above_expectation"))
        agg    = _safe_val(row.get("aggressiveness"))
        ttt    = _safe_val(row.get("avg_time_to_throw"))
        rating = _safe_val(row.get("passer_rating"))
        comp   = _safe_val(row.get("completion_percentage"))

        result[gsis_id].update({
            f"pass_attempts{suffix}":   _safe_val(row.get("attempts")),
            f"pass_yards{suffix}":      _safe_val(row.get("pass_yards")),
            f"pass_tds{suffix}":        _safe_val(row.get("pass_touchdowns")),
            f"interceptions{suffix}":   _safe_val(row.get("interceptions")),
            f"completion_pct{suffix}":  round(comp, 1) if comp is not None else None,
            f"cpoe{suffix}":            round(cpoe, 2) if cpoe is not None else None,
            f"aggressiveness{suffix}":  round(agg, 1) if agg is not None else None,
            f"time_to_throw{suffix}":   round(ttt, 2) if ttt is not None else None,
            f"passer_rating{suffix}":   round(rating, 1) if rating is not None else None,
        })

    logger.info("Fetched NGS passing stats for %d players (%s)", len(result), years)
    return result


def fetch_games_missed(years: list[int] = [2024, 2025]) -> dict[str, dict]:
    """
    Computes games missed per player per season from NFL injury reports.
    Still uses nfl_data_py because nflreadpy's 2025 injury feed is incomplete.
    """
    try:
        injuries = nfl.import_injuries(years)
    except Exception as e:
        logger.error("Failed to fetch injury data: %s", e)
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

    logger.info("Computed games missed for %d players (%s)", len(result), years)
    return result


def build_nfl_data(years: list[int] = [2024, 2025]) -> dict[str, dict]:
    """
    Fetches and merges rosters, seasonal stats, NGS stats, and injury data
    into a single dict keyed by gsis_id.
    Merge order: rosters -> seasonal -> NGS rushing -> NGS receiving ->
    NGS passing -> injuries. Later sources overwrite earlier ones so that
    higher-quality NGS metrics always win over basic seasonal counts.
    """
    logger.info("Building NFL data for years %s...", years)

    rosters   = fetch_rosters(years)
    seasonal  = fetch_seasonal_stats(years)
    rushing   = fetch_rushing_stats(years)
    receiving = fetch_receiving_stats(years)
    passing   = fetch_passing_stats(years)
    injuries  = fetch_games_missed(years)

    merged: dict[str, dict] = {}

    for source in [rosters, seasonal, rushing, receiving, passing, injuries]:
        for gsis_id, data in source.items():
            if gsis_id not in merged:
                merged[gsis_id] = {"gsis_id": gsis_id}
            for k, v in data.items():
                if k == "gsis_id":
                    continue
                # Seasonal only fills gaps — NGS/injuries always win
                if source is seasonal:
                    if k not in merged[gsis_id] or merged[gsis_id][k] is None:
                        merged[gsis_id][k] = v
                else:
                    if v is not None:
                        merged[gsis_id][k] = v

    logger.info("NFL data built for %d players", len(merged))
    return merged


def build_name_gsis_lookup(nfl_data: dict[str, dict]) -> dict[str, str]:
    """Builds a normalized_name -> gsis_id lookup from already-fetched data."""
    lookup: dict[str, str] = {}
    for gsis_id, data in nfl_data.items():
        name = data.get("player_display_name", "").strip()
        if name:
            lookup[_norm_name(name)] = gsis_id
    logger.info("Built name->gsis_id lookup with %d entries", len(lookup))
    return lookup


def save_name_gsis_lookup(lookup: dict[str, str]) -> None:
    RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
    with open(RAW_DATA_PATH / "name_gsis_lookup.json", "w") as f:
        json.dump(lookup, f)
    logger.info("Saved name->gsis_id lookup (%d entries)", len(lookup))


def load_name_gsis_lookup() -> dict[str, str]:
    path = RAW_DATA_PATH / "name_gsis_lookup.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def hash_nfl_player(data: dict) -> str:
    """Hashes key performance fields for change detection."""
    watchlist = {
        "ypc_2025":           data.get("ypc_2025"),
        "ryoe_2025":          data.get("ryoe_2025"),
        "rec_yards_2025":     data.get("rec_yards_2025"),
        "games_missed_2025":  data.get("games_missed_2025"),
        "pass_yards_2025":    data.get("pass_yards_2025"),
        "fantasy_points_half_ppr_2025": data.get("fantasy_points_half_ppr_2025"),
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
    RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
    file_path = RAW_DATA_PATH / "nfl_data.json"
    with open(file_path, "w") as f:
        json.dump(list(data.values()), f, indent=2)
    logger.info("Saved NFL data for %d players to %s", len(data), file_path)


def load_nfl_data() -> dict[str, dict]:
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
