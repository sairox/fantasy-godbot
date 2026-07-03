# Fantasy GodBot — Backend

AI-powered fantasy football draft assistant. FastAPI backend + RAG pipeline (ChromaDB + Claude).

## Commands

```bash
# Dev server
uv run python main.py          # http://localhost:8000

# Refresh all player data (requires REFRESH_API_KEY header)
uv run python refresh.py       # or POST /refresh with X-Api-Key header

# Tests
uv run pytest                  # full suite
uv run pytest tests/test_api.py -v   # API tests only

# Lint
ruff check src/
```

## Architecture

```
User → POST /chat → chain.py (LangChain LCEL) → ChromaDB retrieval → Claude claude-sonnet-4-5
                                                  ↑
                                         orchestrator.py (refresh pipeline)
                                                  ↑
                           fantasypros_agent.py + nfl_data_agent.py (nflreadpy)
```

**Data flow:**
1. `orchestrator.py` fetches from `fantasypros_agent.py` (FP rankings via `nflreadpy.load_ff_rankings`) and `nfl_data_agent.py` (stats via `nflreadpy.load_player_stats`)
2. Documents built in `chroma_store.py._player_to_text()` — this is the text the LLM reads
3. ChromaDB persists to `data/chroma/` (gitignored); embeddings use `all-MiniLM-L6-v2` via ONNX

**Key files:**
- `src/api/routes.py` — FastAPI endpoints (`/chat`, `/evaluate-pick`, `/player/{name}`, `/refresh`, `/health`)
- `src/chatbot/chain.py` — LangChain chain, system prompt, nickname map, round-range routing
- `src/vectorstore/chroma_store.py` — document templates, ADP display format, Chroma CRUD
- `src/ingestion/orchestrator.py` — refresh pipeline, VORP ranking computation
- `src/ingestion/fantasypros_agent.py` — FP rankings via nflreadpy
- `src/ingestion/nfl_data_agent.py` — seasonal stats, rosters, NGS via nflreadpy

## Environment Variables

All secrets live in `.env` — never commit them. Required:

```
ANTHROPIC_API_KEY=       # Claude API
REFRESH_API_KEY=         # protects POST /refresh
CORS_ORIGINS=            # comma-separated, default: http://localhost:3000,http://localhost:5173
```

## Critical Invariants

- **ADP/ECR values are overall pick numbers** (e.g. 3.77 = 4th overall), NOT round.pick notation. The system prompt and ADP display both enforce this — don't change the format.
- **`data/` is gitignored** — never commit ChromaDB files or raw JSON. The data directory is rebuilt via `/refresh`.
- **API keys only in `.env`** — never hardcode or log them.
- **`fp_stats` is hardcoded to `{}`** in `orchestrator.py` — stale `fantasypros_stats.json` caused wrong stats (Bowers 289 pts vs actual 144.2). Do not restore the cache read without fixing the staleness bug first.
- **`max_tokens=2048`** in chain.py — do not lower; responses were truncating at 1024.
- **Position ranks** come from `redraft-rb/wr/qb/te` pages (not overall) and are stored as `pos_rank_half_ppr_2026` in Chroma metadata.

## Frontend Handoff

The React frontend lives in a separate repo (`fantasy-godbot-ui`). Backend exposes CORS for `localhost:3000` and `localhost:5173`. A streaming endpoint (`POST /chat/stream`) is planned but not yet implemented — add it here when the frontend needs it.

## Planned: PostgreSQL Chat History

Current chat memory is in-process only (`_chat_chains` dict in `routes.py`) — lost on restart. Planned migration:
- Add `conversations` table (`session_id`, `role`, `content`, `timestamp`)
- Replace in-memory dict with DB-backed message loader in `/chat`
- Use SQLAlchemy + alembic for migrations
- Enables persistent sessions across restarts and multiple server instances (required for AWS)

## Data Pipeline Notes

- `nflreadpy.load_ff_rankings(type="draft")` returns `page_type` values: `redraft-overall`, `dynasty-overall`, `redraft-rb`, `redraft-wr`, `redraft-qb`, `redraft-te`
- `nflreadpy.load_player_stats(years, summary_level="reg")` covers 2025 season (2479 players)
- `fetch_stats()` and `fetch_adp()` in `fantasypros_agent.py` are intentional no-ops — stats come from `nfl_data_agent.py`
- Injury data still uses `nfl_data_py.import_injuries()` — nflreadpy's injury feed is incomplete
