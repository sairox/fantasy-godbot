"""
Diagnoses the data pipeline: checks sleeper, nfl_data, and Chroma
for key fantasy players. Run with: uv run python scripts/diagnose_data.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

RAW = ROOT / "data" / "raw"

KEY_PLAYERS = [
    "Christian McCaffrey", "Bijan Robinson", "Ja'Marr Chase",
    "CeeDee Lamb", "Justin Jefferson", "Saquon Barkley",
    "Jahmyr Gibbs", "Patrick Mahomes", "Josh Allen", "Travis Kelce",
]


def section(title):
    print(f"\n{'='*60}\n{title}\n{'='*60}")


# 1. Sleeper players
section("1. Sleeper players — key players present?")
sleeper_path = RAW / "sleeper_players.json"
if not sleeper_path.exists():
    print("  sleeper_players.json not found — run refresh first")
    sleeper = []
else:
    sleeper = json.load(open(sleeper_path))
    print(f"  Total players: {len(sleeper)}")
    for name in KEY_PLAYERS:
        match = next((p for p in sleeper if p.get("full_name") == name), None)
        if match:
            gsis = match.get("gsis_id") or "MISSING"
            status = match.get("status", "?")
            depth = match.get("depth_chart_order", "?")
            sr = match.get("search_rank", "?")
            print(f"  ✓ {name} | gsis_id={gsis} | status={status} | depth={depth} | search_rank={sr}")
        else:
            print(f"  ✗ {name} — NOT in sleeper_players.json")

# 2. NFL data (NGS)
section("2. NFL data (NGS) — gsis_id cross-match")
nfl_path = RAW / "nfl_data.json"
if not nfl_path.exists():
    print("  nfl_data.json not found — run refresh first")
else:
    nfl_records = json.load(open(nfl_path))
    nfl_by_gsis = {r["gsis_id"]: r for r in nfl_records if "gsis_id" in r}
    print(f"  Total NGS records: {len(nfl_by_gsis)}")
    for name in KEY_PLAYERS:
        sp = next((p for p in sleeper if p.get("full_name") == name), None)
        if not sp:
            print(f"  — {name}: not in Sleeper, skipping")
            continue
        gsis = sp.get("gsis_id") or ""
        ngs = nfl_by_gsis.get(gsis, {})
        if ngs:
            rush = ngs.get("rush_attempts_2025", "—")
            tgt  = ngs.get("targets_2025", "—")
            att  = ngs.get("pass_attempts_2025", "—")
            gm   = ngs.get("games_missed_2025", "—")
            print(f"  ✓ {name} (gsis={gsis}): rush_att={rush}, targets={tgt}, pass_att={att}, games_missed={gm}")
        else:
            print(f"  ✗ {name} (gsis={gsis or 'MISSING'}): NO NGS data — gsis_id not matched")

# 3. Chroma vectorstore
section("3. Chroma vectorstore — key players indexed?")
try:
    from src.vectorstore.chroma_store import get_vectorstore
    vs = get_vectorstore()
    result = vs.get(include=["metadatas"])
    all_names = [m.get("name", "") for m in result.get("metadatas", [])]
    print(f"  Total documents in Chroma: {len(all_names)}")
    for name in KEY_PLAYERS:
        first = name.split()[0].lower()
        last  = name.split()[-1].lower()
        found = any(first in n.lower() and last in n.lower() for n in all_names)
        print(f"  {'✓' if found else '✗'} {name}")
except Exception as e:
    print(f"  ERROR loading Chroma: {e}")

# 4. Check if RBs appear in NGS receiving data at all (by name scan)
section("4. NGS receiving data — do RBs appear at all?")
if nfl_path.exists():
    rb_names = ["Christian McCaffrey", "Bijan Robinson", "Saquon Barkley",
                "Jahmyr Gibbs", "De'Von Achane", "James Cook"]
    nfl_records = json.load(open(nfl_path))
    # Build name->record lookup from nfl_data
    nfl_by_name: dict = {}
    for r in nfl_records:
        dn = r.get("player_display_name", "").strip()
        if dn:
            nfl_by_name[dn.lower()] = r
    for name in rb_names:
        rec = nfl_by_name.get(name.lower())
        if rec:
            tgt25 = rec.get("targets_2025", "—")
            tgt24 = rec.get("targets_2024", "—")
            rush25 = rec.get("rush_attempts_2025", "—")
            print(f"  {name}: rush_att_2025={rush25}, targets_2025={tgt25}, targets_2024={tgt24}")
        else:
            print(f"  {name}: NOT found in nfl_data.json by display name")

# 5. Check FantasyPros rankings merge for Bijan
section("5. FantasyPros data — does Bijan appear in rankings/stats?")
processed_path = ROOT / "data" / "processed" / "player_documents.json"
if processed_path.exists():
    docs_all = json.load(open(processed_path))
    for name in ["Bijan Robinson", "Ja'Marr Chase", "Christian McCaffrey"]:
        doc = next((d for d in docs_all if d.get("full_name") == name), None)
        if doc:
            print(f"  {name}:")
            print(f"    rank_half_ppr_2026 = {doc.get('rank_half_ppr_2026')}")
            print(f"    adp_2025           = {doc.get('adp_2025')}")
            print(f"    fantasy_points_half_ppr_2025 = {doc.get('fantasy_points_half_ppr_2025')}")
            print(f"    targets_2025       = {doc.get('targets_2025')}")
            print(f"    rush_attempts_2025 = {doc.get('rush_attempts_2025')}")
        else:
            print(f"  {name}: not found in player_documents.json")
else:
    print("  player_documents.json not found")

# 6. Sample Chroma document
section("6. Sample Chroma document — Bijan Robinson")
try:
    from src.vectorstore.chroma_store import get_vectorstore
    vs = get_vectorstore()
    docs = vs.similarity_search("Bijan Robinson", k=1)
    if docs:
        print(docs[0].page_content[:2000])
    else:
        print("  Not found in Chroma")
except Exception as e:
    print(f"  ERROR: {e}")
