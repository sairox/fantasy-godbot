"""
Exploratory scrape of Pro Football Reference to see what stats are available.
Run with: uv run python scripts/explore_pfr.py
"""
import cloudscraper
import time
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0",
}

EXPLORE_URLS = {
    "rushing_2025":   "https://www.pro-football-reference.com/years/2025/rushing.htm",
    "receiving_2025": "https://www.pro-football-reference.com/years/2025/receiving.htm",
    "passing_2025":   "https://www.pro-football-reference.com/years/2025/passing.htm",
}

SAMPLE_PLAYERS = [
    "christian mccaffrey",
    "jonathan taylor",
    "bijan robinson",
    "tyreek hill",
    "ceedee lamb",
    "patrick mahomes",
]


_scraper = None

def get_scraper():
    global _scraper
    if _scraper is None:
        _scraper = cloudscraper.create_scraper(
            browser={"browser": "chrome", "platform": "windows", "mobile": False}
        )
    return _scraper

def fetch(url: str) -> str:
    scraper = get_scraper()
    r = scraper.get(url, timeout=30)
    print(f"  Status: {r.status_code}")
    r.raise_for_status()
    return r.text


def parse_stats_table(html: str, url_key: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")

    # PFR embeds tables inside comments for some pages — unwrap them
    from bs4 import Comment
    comments = soup.find_all(string=lambda text: isinstance(text, Comment))
    for comment in comments:
        if "<table" in comment:
            frag = BeautifulSoup(comment, "html.parser")
            for tbl in frag.find_all("table"):
                soup.body.append(tbl)

    # Find the main stats table
    table = (
        soup.find("table", id="rushing") or
        soup.find("table", id="receiving") or
        soup.find("table", id="passing") or
        soup.find("table", class_="sortable")
    )
    if not table:
        print(f"  [!] No table found for {url_key}")
        return []

    # Extract headers
    headers = []
    for th in table.select("thead tr th, thead tr td"):
        stat = th.get("data-stat", th.get_text(strip=True))
        headers.append(stat)

    print(f"\n  Columns ({len(headers)}): {headers}\n")

    # Parse rows
    rows = []
    for tr in table.select("tbody tr"):
        if "thead" in tr.get("class", []) or tr.get("class") == ["thead"]:
            continue
        cells = tr.find_all(["td", "th"])
        if not cells:
            continue
        row = {}
        for i, cell in enumerate(cells):
            if i < len(headers):
                key = headers[i]
                # player links have data-append-csv (PFR player ID)
                link = cell.find("a")
                if link and "players" in link.get("href", ""):
                    row["pfr_id"] = link["href"].split("/")[-1].replace(".htm", "")
                row[key] = cell.get_text(strip=True)
        if row:
            rows.append(row)

    return rows


def find_sample_players(rows: list[dict], sample_names: list[str]) -> None:
    """Prints stats for our sample fantasy players."""
    name_key = next(
        (k for k in (rows[0].keys() if rows else []) if "player" in k.lower()),
        None
    )
    if not name_key:
        print("  [!] Could not identify player name column")
        return

    for target in sample_names:
        match = next(
            (r for r in rows if target.lower() in r.get(name_key, "").lower()),
            None
        )
        if match:
            print(f"  ✓ {match.get(name_key)}")
            for k, v in match.items():
                if v and v not in ("", "—", "0"):
                    print(f"      {k}: {v}")
        else:
            print(f"  ✗ {target} — not found")


def main():
    for url_key, url in EXPLORE_URLS.items():
        print(f"\n{'='*60}")
        print(f"Fetching: {url_key}")
        print(f"URL: {url}")
        print("="*60)
        try:
            html = fetch(url)
            print(f"  Response: {len(html):,} bytes")
            rows = parse_stats_table(html, url_key)
            print(f"  Rows parsed: {len(rows)}")
            if rows:
                find_sample_players(rows, SAMPLE_PLAYERS)
        except Exception as e:
            print(f"  [ERROR] {e}")
        time.sleep(3)


if __name__ == "__main__":
    main()
