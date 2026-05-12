"""
Audits what correct 2024/2025 data is available from nfl-data-py.
Shows accurate stats for fantasy-relevant players.
Run with: uv run python scripts/audit_nfl_data.py
"""
import nfl_data_py as nfl
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 300)


def section(title):
    print(f"\n{'='*70}\n{title}\n{'='*70}")


# ── 1. What years does seasonal data exist for? ──────────────────────────────
section("Seasonal data availability by year")
for yr in [2022, 2023, 2024, 2025]:
    try:
        df = nfl.import_seasonal_data([yr])
        print(f"  {yr}: {len(df)} rows — columns: {list(df.columns[:8])}...")
    except Exception as e:
        print(f"  {yr}: ERROR — {e}")

# ── 2. Weekly data availability ───────────────────────────────────────────────
section("Weekly data availability by year")
for yr in [2024, 2025]:
    try:
        df = nfl.import_weekly_data([yr])
        print(f"  {yr}: {len(df)} rows, {len(df.columns)} cols")
        if len(df):
            print(f"       weeks: {sorted(df['week'].unique())[:5]}...")
            print(f"       cols: {list(df.columns[:12])}")
    except Exception as e:
        print(f"  {yr}: ERROR — {e}")

# ── 3. NGS Rushing — accurate 2025 stats for top RBs ────────────────────────
section("NGS Rushing 2025 — top fantasy RBs (week=0 = full season)")
try:
    rush = nfl.import_ngs_data("rushing", [2024, 2025])
    rush25 = rush[(rush["season"] == 2025) & (rush["week"] == 0)].copy()
    rush24 = rush[(rush["season"] == 2024) & (rush["week"] == 0)].copy()

    TOP_RBS = [
        "Christian McCaffrey", "Jonathan Taylor", "Bijan Robinson",
        "Derrick Henry", "Breece Hall", "James Cook", "Saquon Barkley",
        "De'Von Achane", "Jahmyr Gibbs", "Kyren Williams", "Josh Jacobs",
        "Tony Pollard", "Rhamondre Stevenson", "Joe Mixon",
    ]

    cols_25 = ["player_display_name", "team_abbr", "rush_attempts",
               "rush_yards", "avg_rush_yards", "rush_touchdowns",
               "rush_yards_over_expected", "rush_yards_over_expected_per_att",
               "efficiency", "percent_attempts_gte_eight_defenders"]

    print("\n  2025 Season:")
    for name in TOP_RBS:
        row = rush25[rush25["player_display_name"] == name]
        if not row.empty:
            r = row.iloc[0]
            print(f"  {name} ({r['team_abbr']}): "
                  f"{r['rush_attempts']} att, {r['rush_yards']} yds, "
                  f"{r['avg_rush_yards']:.2f} YPC, {int(r['rush_touchdowns'])} TD | "
                  f"RYOE/att: {r['rush_yards_over_expected_per_att']:.2f}")
        else:
            print(f"  {name}: — not found in 2025")

    print("\n  2024 Season (for trend comparison):")
    for name in TOP_RBS[:8]:
        row = rush24[rush24["player_display_name"] == name]
        if not row.empty:
            r = row.iloc[0]
            print(f"  {name}: {r['rush_attempts']} att, {r['rush_yards']} yds, "
                  f"{r['avg_rush_yards']:.2f} YPC | RYOE/att: {r['rush_yards_over_expected_per_att']:.2f}")
        else:
            print(f"  {name}: — not found in 2024")
except Exception as e:
    print(f"  ERROR: {e}")

# ── 4. NGS Receiving — accurate 2025 stats for top WRs/TEs ──────────────────
section("NGS Receiving 2025 — top fantasy WRs/TEs (week=0 = full season)")
try:
    recv = nfl.import_ngs_data("receiving", [2024, 2025])
    recv25 = recv[(recv["season"] == 2025) & (recv["week"] == 0)].copy()
    recv24 = recv[(recv["season"] == 2024) & (recv["week"] == 0)].copy()

    TOP_WRS = [
        "Ja'Marr Chase", "CeeDee Lamb", "Justin Jefferson", "Tyreek Hill",
        "Puka Nacua", "Amon-Ra St. Brown", "Davante Adams", "Stefon Diggs",
        "Mike Evans", "Malik Nabers", "Marvin Harrison",
        "Sam LaPorta", "Brock Bowers", "Mark Andrews", "Travis Kelce",
    ]

    print("\n  2025 Season:")
    for name in TOP_WRS:
        row = recv25[recv25["player_display_name"] == name]
        if not row.empty:
            r = row.iloc[0]
            print(f"  {name} ({r['team_abbr']}, {r['player_position']}): "
                  f"{int(r['receptions'])} rec / {int(r['targets'])} tgt "
                  f"({r['catch_percentage']:.1f}%), {int(r['yards'])} yds, "
                  f"{int(r['rec_touchdowns'])} TD | "
                  f"sep: {r['avg_separation']:.2f}, "
                  f"YAC+: {r['avg_yac_above_expectation']:.2f}")
        else:
            print(f"  {name}: — not found in 2025")
except Exception as e:
    print(f"  ERROR: {e}")

# ── 5. NGS Passing — accurate 2025 stats for top QBs ────────────────────────
section("NGS Passing 2025 — top fantasy QBs (week=0 = full season)")
try:
    passing = nfl.import_ngs_data("passing", [2024, 2025])
    pass25 = passing[(passing["season"] == 2025) & (passing["week"] == 0)].copy()

    TOP_QBS = [
        "Patrick Mahomes", "Josh Allen", "Lamar Jackson", "Joe Burrow",
        "Jalen Hurts", "Dak Prescott", "Tua Tagovailoa", "Justin Herbert",
        "C.J. Stroud", "Brock Purdy", "Jordan Love", "Anthony Richardson",
    ]

    print("\n  2025 Season:")
    for name in TOP_QBS:
        row = pass25[pass25["player_display_name"] == name]
        if not row.empty:
            r = row.iloc[0]
            print(f"  {name} ({r['team_abbr']}): "
                  f"{int(r['attempts'])} att, {int(r['pass_yards'])} yds, "
                  f"{int(r['pass_touchdowns'])} TD, {int(r['interceptions'])} INT | "
                  f"cmp%: {r['completion_percentage']:.1f} "
                  f"(xCmp%: {r['expected_completion_percentage']:.1f}, "
                  f"diff: {r['completion_percentage_above_expectation']:+.1f}) | "
                  f"aggr: {r['aggressiveness']:.1f}")
        else:
            print(f"  {name}: — not found in 2025")
except Exception as e:
    print(f"  ERROR: {e}")

# ── 6. Injuries — compute actual games missed ────────────────────────────────
section("Injuries 2025 — games missed (Out/IR) per player")
try:
    injuries = nfl.import_injuries([2024, 2025])
    print(f"  Total injury report rows: {len(injuries)}")
    print(f"  report_status values: {injuries['report_status'].dropna().unique()[:10]}")

    # Games missed = weeks where report_status == "Out"
    out_2025 = injuries[
        (injuries["season"] == 2025) &
        (injuries["report_status"].isin(["Out", "Did Not Participate In Practice"]))
    ].copy()

    out_2024 = injuries[
        (injuries["season"] == 2024) &
        (injuries["report_status"].isin(["Out", "Did Not Participate In Practice"]))
    ].copy()

    # Group by player gsis_id and count distinct weeks
    missed_2025 = (
        out_2025.groupby(["gsis_id", "full_name", "position"])["week"]
        .nunique()
        .reset_index()
        .rename(columns={"week": "games_missed_2025"})
        .sort_values("games_missed_2025", ascending=False)
    )

    missed_2024 = (
        out_2024.groupby(["gsis_id", "full_name", "position"])["week"]
        .nunique()
        .reset_index()
        .rename(columns={"week": "games_missed_2024"})
        .sort_values("games_missed_2024", ascending=False)
    )

    print("\n  Most games missed in 2025 (top 20):")
    print(missed_2025.head(20).to_string(index=False))

    print("\n  Most games missed in 2024 (top 15):")
    print(missed_2024[missed_2024["position"].isin(["QB","RB","WR","TE"])].head(15).to_string(index=False))

    # Check specific fantasy players
    print("\n  Fantasy-relevant players — 2025 games missed:")
    FANTASY_NAMES = [
        "Christian McCaffrey", "Jonathan Taylor", "Bijan Robinson",
        "Tyreek Hill", "CeeDee Lamb", "Patrick Mahomes",
        "Ja'Marr Chase", "Justin Jefferson", "Derrick Henry",
        "Breece Hall", "James Cook", "Puka Nacua", "Malik Nabers",
        "Jahmyr Gibbs", "Saquon Barkley", "Josh Allen", "Lamar Jackson",
    ]
    for name in FANTASY_NAMES:
        row = missed_2025[missed_2025["full_name"] == name]
        if not row.empty:
            print(f"  {name}: {int(row.iloc[0]['games_missed_2025'])} games missed")
        else:
            print(f"  {name}: 0 games missed (no Out weeks)")

except Exception as e:
    print(f"  ERROR: {e}")
    import traceback; traceback.print_exc()
