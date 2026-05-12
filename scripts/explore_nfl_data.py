"""
Explores nfl-data-py to see what advanced stats are available.
Run with: uv run python scripts/explore_nfl_data.py
"""
import nfl_data_py as nfl
import pandas as pd

SAMPLE_PLAYERS = [
    "Christian McCaffrey",
    "Jonathan Taylor",
    "Bijan Robinson",
    "Tyreek Hill",
    "CeeDee Lamb",
    "Patrick Mahomes",
    "Ja'Marr Chase",
    "Justin Jefferson",
    "Derrick Henry",
    "Breece Hall",
    "James Cook",
    "Malik Nabers",
    "Puka Nacua",
]

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)


def section(title: str):
    print(f"\n{'='*60}\n{title}\n{'='*60}")


def show_player_rows(df: pd.DataFrame, name_col: str, cols_to_show: list[str]):
    for name in SAMPLE_PLAYERS:
        mask = df[name_col].str.contains(name.split()[0], case=False, na=False)
        rows = df[mask]
        if not rows.empty:
            r = rows.iloc[0]
            vals = {c: r[c] for c in cols_to_show if c in r.index and pd.notna(r[c])}
            print(f"  {name}: {vals}")
        else:
            print(f"  {name}: — not found")


# ── 1. Seasonal stats (main fantasy stats) ──────────────────────────────────
section("1. Seasonal stats (2025)")
try:
    seasonal = nfl.import_seasonal_data([2025])
    print(f"  Rows: {len(seasonal)}  |  Columns ({len(seasonal.columns)}):")
    print(f"  {list(seasonal.columns)}\n")
    key_cols = [
        "player_id", "player_name", "season", "season_type", "position",
        "games", "targets", "receptions", "receiving_yards", "receiving_tds",
        "carries", "rushing_yards", "rushing_tds",
        "receiving_epa", "rushing_epa",
        "target_share", "air_yards_share",
        "wopr",           # weighted opportunity rating
        "racr",           # receiver air conversion ratio
        "fantasy_points", "fantasy_points_ppr",
        "rushing_first_downs", "receiving_first_downs",
    ]
    show_player_rows(seasonal, "player_name", key_cols)
except Exception as e:
    print(f"  ERROR: {e}")


# ── 2. Weekly stats (game-by-game) ──────────────────────────────────────────
section("2. Weekly stats (2025)")
try:
    weekly = nfl.import_weekly_data([2025])
    print(f"  Rows: {len(weekly)}  |  Columns ({len(weekly.columns)}):")
    print(f"  {list(weekly.columns)}\n")
    sample_cols = [
        "player_name", "week", "position", "targets", "receptions",
        "receiving_yards", "carries", "rushing_yards",
        "target_share", "snap_pct",
        "fantasy_points", "fantasy_points_ppr",
        "receiving_epa", "rushing_epa",
    ]
    show_player_rows(weekly[weekly["week"] == 1], "player_name", sample_cols)
except Exception as e:
    print(f"  ERROR: {e}")


# ── 3. NGS passing stats (advanced QB) ──────────────────────────────────────
section("3. NGS Passing advanced stats (2025)")
try:
    ngs_pass = nfl.import_ngs_data("passing", [2025])
    print(f"  Columns: {list(ngs_pass.columns)}\n")
    show_player_rows(ngs_pass, "player_display_name", list(ngs_pass.columns))
except Exception as e:
    print(f"  ERROR: {e}")


# ── 4. NGS rushing stats (advanced RB) ──────────────────────────────────────
section("4. NGS Rushing advanced stats (2025)")
try:
    ngs_rush = nfl.import_ngs_data("rushing", [2025])
    print(f"  Columns: {list(ngs_rush.columns)}\n")
    show_player_rows(ngs_rush, "player_display_name", list(ngs_rush.columns))
except Exception as e:
    print(f"  ERROR: {e}")


# ── 5. NGS receiving stats (advanced WR/TE) ─────────────────────────────────
section("5. NGS Receiving advanced stats (2025)")
try:
    ngs_recv = nfl.import_ngs_data("receiving", [2025])
    print(f"  Columns: {list(ngs_recv.columns)}\n")
    show_player_rows(ngs_recv, "player_display_name", list(ngs_recv.columns))
except Exception as e:
    print(f"  ERROR: {e}")


# ── 6. Injuries ──────────────────────────────────────────────────────────────
section("6. Injuries (2025)")
try:
    injuries = nfl.import_injuries([2025])
    print(f"  Columns: {list(injuries.columns)}\n")
    show_player_rows(injuries, "full_name", list(injuries.columns)[:10])
except Exception as e:
    print(f"  ERROR: {e}")
