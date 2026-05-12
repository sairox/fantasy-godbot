import httpx
import json
import hashlib
from pathlib import Path

def fetch_players() -> dict:
    url = "https://api.sleeper.app/v1/players/nfl"
    response = httpx.get(url, timeout=30.0)
    response.raise_for_status()
    return response.json()

def filter_players(players: dict, league_format: str = "redraft") -> list:
    RELEVANT_POSITIONS = ["QB", "WR", "RB", "TE", "K", "DEF"]
    
    RELEVANT_STATUSES = [
        "Active",
        "Inactive", 
        "Injured Reserve",
        "Non Football Injury",
        "Physically Unable to Perform"
    ]
    
    # dynasty also includes practice squad
    if league_format == "dynasty":
        RELEVANT_STATUSES.append("Practice Squad")
    
    filtered = []
    
    for player_id, player in players.items():
        # check 1: position
        positions = player.get("fantasy_positions") or []
        if not any(pos in RELEVANT_POSITIONS for pos in positions):
            continue
            
        # check 2: status
        if player.get("status") not in RELEVANT_STATUSES:
            continue
        
        # check 3: depth chart
        depth = player.get("depth_chart_order")
        if depth and depth > 5:
            continue
            
        # check 4: search rank (9999999 means irrelevant)
        if player.get("search_rank") == 9999999:
            continue
        
        # player passed all checks
        filtered.append({
            "player_id": player_id,
            "full_name": player.get("full_name"),
            "position": player.get("position"),
            "fantasy_positions": positions,
            "team": player.get("team"),
            "age": player.get("age"),
            "years_exp": player.get("years_exp"),
            "status": player.get("status"),
            "depth_chart_order": player.get("depth_chart_order"),
            "injury_status": player.get("injury_status"),
            "injury_body_part": player.get("injury_body_part"),
            "injury_start_date": player.get("injury_start_date"),
            "practice_participation": player.get("practice_participation"),
            "search_rank": player.get("search_rank"),
            "gsis_id": player.get("gsis_id"),
        })
    
    return filtered

def hash_player(player: dict) -> str:
    # only hash the fields that affect fantasy relevance
    watchlist = {
        "status": player.get("status"),
        "team": player.get("team"),
        "depth_chart_order": player.get("depth_chart_order"),
        "injury_status": player.get("injury_status"),
    }
    # convert to string and hash it
    content = json.dumps(watchlist, sort_keys=True)
    return hashlib.md5(content.encode()).hexdigest()

def find_changed_players(old_players: list, new_players: list) -> list:
    # build a lookup of old hashes by player_id
    old_hashes = {p["player_id"]: hash_player(p) for p in old_players}
    
    changed = []
    for player in new_players:
        player_id = player["player_id"]
        new_hash = hash_player(player)
        
        # player is new OR their hash changed
        if player_id not in old_hashes or old_hashes[player_id] != new_hash:
            changed.append(player)
    
    return changed

def save_raw_data(players: list) -> None:
    # 1. define the path using pathlib
    RAW_DATA_PATH = Path(__file__).parent.parent.parent / "data" / "raw"
    
    # 2. make sure the directory exists
    RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
    
    # 3. write the file
    file_path = RAW_DATA_PATH / "sleeper_players.json"
    with open(file_path, "w") as f:
        json.dump(players, f, indent=2)
    
    print(f"Saved {len(players)} players to {file_path}")

def load_existing_players() -> list:
    """Loads previously saved Sleeper player data."""
    RAW_DATA_PATH = Path(__file__).parent.parent.parent / "data" / "raw"
    file_path = RAW_DATA_PATH / "sleeper_players.json"
    if not file_path.exists():
        return []
    with open(file_path) as f:
        return json.load(f)


if __name__ == "__main__":
    print("Fetching players from Sleeper API...")
    raw_players = fetch_players()
    print(f"Total players fetched: {len(raw_players)}")
    
    filtered = filter_players(raw_players)
    print(f"Fantasy relevant players: {len(filtered)}")
    
    save_raw_data(filtered)