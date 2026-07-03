import logging
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

CHROMA_PERSIST_DIR = str(Path(__file__).parent.parent.parent / "data" / "chroma")
COLLECTION_NAME = "fantasy_players"

_embeddings = None
_vectorstore = None


class _ChromaONNXEmbeddings(Embeddings):
    """Wraps Chroma's bundled ONNX all-MiniLM-L6-v2 as a LangChain Embeddings."""

    def __init__(self):
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
        self._ef = DefaultEmbeddingFunction()

    @staticmethod
    def _to_list(vec) -> list[float]:
        return vec.tolist() if hasattr(vec, "tolist") else [float(x) for x in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._to_list(vec) for vec in self._ef(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self._to_list(self._ef([text])[0])


def _get_embeddings() -> Embeddings:
    """Returns cached embeddings instance (all-MiniLM-L6-v2 via Chroma ONNX)."""
    global _embeddings
    if _embeddings is None:
        logger.info("Loading embeddings model (all-MiniLM-L6-v2 ONNX)...")
        _embeddings = _ChromaONNXEmbeddings()
    return _embeddings


def _fmt(val, decimals: int = 1, suffix: str = "") -> str:
    """Formats a numeric value or returns 'N/A'."""
    if val is None:
        return "N/A"
    try:
        return f"{float(val):.{decimals}f}{suffix}"
    except (TypeError, ValueError):
        return str(val)


def _v(player: dict, key: str, default: str = "N/A") -> str:
    """Returns the player field as a string, or default if None/missing."""
    val = player.get(key)
    return str(val) if val is not None else default


def _player_to_text(player: dict) -> str:
    """
    Converts a merged player document to rich natural-language text for embedding.
    Semantic search works far better on prose than raw JSON.
    """
    name = player.get("full_name", "Unknown")
    pos = player.get("position") or ""
    pos_str = pos
    team = player.get("team") or "N/A"
    age = player.get("age") or "?"
    exp = player.get("years_exp") or "?"

    # rankings — overall rank and position-specific rank are different
    overall_half = player.get("rank_half_ppr_2026")
    pos_rank     = player.get("pos_rank_half_ppr_2026")
    rk_dyn       = player.get("rank_dynasty_2026") or "unranked"

    # Display: "RB4 (overall #10)" so bot knows both position rank and overall slot
    if pos_rank and overall_half:
        rk_half = f"{pos_str}{pos_rank} (overall #{overall_half})"
    elif pos_rank:
        rk_half = f"{pos_str}{pos_rank}"
    elif overall_half:
        rk_half = f"overall #{overall_half}"
    else:
        rk_half = "unranked"

    rk_std = rk_half  # standard ≈ half-PPR for display purposes
    rk_ppr = rk_half

    # 2025 performance
    fpts_25 = player.get("fantasy_points_half_ppr_2025") or "N/A"
    ppg_25 = player.get("points_per_game_2025") or "N/A"
    gp_25 = player.get("games_played_2025") or "N/A"
    gm_25 = player.get("games_missed_2025") or 0
    fin_25 = player.get("finish_rank_half_ppr_2025") or "N/A"

    # ADP — ECR is an OVERALL pick number (e.g. 3.77 = pick 4, round 1 in 12-team)
    adp_raw = player.get("adp_2025")
    ecr_vs_adp = player.get("ecr_vs_adp_2025")

    if adp_raw and adp_raw < 990:
        overall_pick = int(round(adp_raw))
        team_size = 12
        draft_round = (overall_pick - 1) // team_size + 1
        pick_in_round = (overall_pick - 1) % team_size + 1
        adp_line = (
            f"FP ADP: pick {overall_pick} overall "
            f"(round {draft_round}, pick {pick_in_round} in 12-team) | ECR {adp_raw:.2f}"
        )
        if ecr_vs_adp is not None:
            if ecr_vs_adp < -3:
                adp_line += f" — undervalued by {abs(ecr_vs_adp):.1f} spots vs ADP"
            elif ecr_vs_adp > 3:
                adp_line += f" — overvalued by {ecr_vs_adp:.1f} spots vs ADP"
    else:
        adp_line = "FP ADP: unranked"

    # 2024 performance
    fpts_24 = player.get("fantasy_points_half_ppr_2024") or "N/A"
    gp_24 = player.get("games_played_2024") or "N/A"
    gm_24 = player.get("games_missed_2024") or 0
    fin_24 = player.get("finish_rank_half_ppr_2024") or "N/A"

    # injury
    injury_risk = player.get("injury_risk_score", "unknown")
    injury_status = player.get("injury_status") or "None"
    injury_part = player.get("injury_body_part") or ""
    injury_type_25 = player.get("injury_type_2025") or ""
    injury_type_24 = player.get("injury_type_2024") or ""

    injury_line = f"Injury risk: {injury_risk.capitalize()}"
    if gm_24:
        inj_str = f" — missed {gm_24} games in 2024"
        if injury_type_24:
            inj_str += f" ({injury_type_24})"
        injury_line += inj_str
    if gm_25:
        inj_str = f", {gm_25} in 2025"
        if injury_type_25:
            inj_str += f" ({injury_type_25})"
        injury_line += inj_str
    if injury_part:
        injury_line += f" | Current: {injury_part}"

    # signals
    trend = player.get("trend", "unknown")
    sleeper_sig = "Yes" if player.get("sleeper_signal") else "No"
    bust_sig = "Yes" if player.get("bust_signal") else "No"

    trend_detail = ""
    if fin_24 != "N/A" and fin_25 != "N/A":
        trend_detail = f" ({pos_str}{fin_24} in 2024 → {pos_str}{fin_25} in 2025)"

    # depth / status
    depth = player.get("depth_chart_order")
    depth_str = f"Starter ({depth})" if depth == 1 else f"Depth #{depth}" if depth else "Unknown"
    status = player.get("status", "Unknown")
    practice = player.get("practice_participation") or "N/A"

    lines = [
        f"{name} | {pos_str} | {team} | Age: {age} | Experience: {exp} years",
        "",
        "2026 DRAFT RANKINGS:",
        f"Standard: {rk_std} | Half PPR: {rk_half} | PPR: {rk_ppr} | Dynasty: {pos_str}{rk_dyn}",
        "",
        "2025 PERFORMANCE:",
        f"Fantasy Points (Half PPR): {fpts_25} | Points Per Game: {ppg_25} | "
        f"Games: {gp_25} (missed {gm_25}) | Finish: {pos_str}{fin_25}" +
        (f" | Target share: {_fmt(player.get('target_share_2025'), 1)}%" if player.get("target_share_2025") else "") +
        (f" | WOPR: {_fmt(player.get('wopr_2025'), 3)}" if player.get("wopr_2025") else ""),
        adp_line,
        "",
        "2024 PERFORMANCE:",
        f"Fantasy Points (Half PPR): {fpts_24} | "
        f"Games: {gp_24} (missed {gm_24}) | Finish: {pos_str}{fin_24}" +
        (f" | Target share: {_fmt(player.get('target_share_2024'), 1)}%" if player.get("target_share_2024") else ""),
        "",
    ]

    # ── NGS advanced stats: show each section if any data exists for that stat type.
    # No position gating — CMC has receiving stats, Lamar has rushing stats, etc.

    if player.get("rush_attempts_2025") or player.get("rush_attempts_2024"):
        rush_lines = ["RUSHING STATS (NGS):"]
        if player.get("rush_attempts_2025"):
            rush_lines.append(
                f"2025: {_v(player,'rush_attempts_2025')} att, "
                f"{_v(player,'rush_yards_2025')} yds, "
                f"YPC {_fmt(player.get('ypc_2025'), 2)}, "
                f"Rush TDs {_v(player,'rush_tds_2025')}, "
                f"RYOE {_fmt(player.get('ryoe_2025'), 1)} ({_fmt(player.get('ryoe_per_att_2025'), 3)}/att), "
                f"Efficiency {_fmt(player.get('rush_efficiency_2025'), 2)}, "
                f"% vs 8+ defenders {_fmt(player.get('pct_vs_8_defenders_2025'), 1)}"
            )
        if player.get("rush_attempts_2024"):
            rush_lines.append(
                f"2024: {_v(player,'rush_attempts_2024')} att, "
                f"{_v(player,'rush_yards_2024')} yds, "
                f"YPC {_fmt(player.get('ypc_2024'), 2)}, "
                f"Rush TDs {_v(player,'rush_tds_2024')}, "
                f"RYOE {_fmt(player.get('ryoe_2024'), 1)} ({_fmt(player.get('ryoe_per_att_2024'), 3)}/att)"
            )
        rush_lines.append("")
        lines += rush_lines

    if player.get("targets_2025") or player.get("targets_2024"):
        rec_lines = ["RECEIVING STATS (NGS):"]
        if player.get("targets_2025") or player.get("receptions_2025"):
            rec_lines.append(
                f"2025: {_v(player,'targets_2025')} tgt, "
                f"{_v(player,'receptions_2025')} rec, "
                f"{_v(player,'rec_yards_2025')} yds, "
                f"Rec TDs {_v(player,'rec_tds_2025')}, "
                f"Catch% {_fmt(player.get('catch_pct_2025'), 1)}, "
                f"Avg separation {_fmt(player.get('avg_separation_2025'), 2)} yds, "
                f"Air yards share {_fmt(player.get('air_yards_share_2025'), 1)}%, "
                f"YAC above expected {_fmt(player.get('yac_above_expected_2025'), 2)}"
            )
        if player.get("targets_2024") or player.get("receptions_2024"):
            rec_lines.append(
                f"2024: {_v(player,'targets_2024')} tgt, "
                f"{_v(player,'receptions_2024')} rec, "
                f"{_v(player,'rec_yards_2024')} yds, "
                f"Rec TDs {_v(player,'rec_tds_2024')}, "
                f"Catch% {_fmt(player.get('catch_pct_2024'), 1)}, "
                f"Avg separation {_fmt(player.get('avg_separation_2024'), 2)} yds, "
                f"Air yards share {_fmt(player.get('air_yards_share_2024'), 1)}%"
            )
        rec_lines.append("")
        lines += rec_lines

    if player.get("pass_attempts_2025") or player.get("pass_attempts_2024"):
        pass_lines = ["PASSING STATS (NGS):"]
        if player.get("pass_attempts_2025"):
            pass_lines.append(
                f"2025: {_v(player,'pass_attempts_2025')} att, "
                f"{_v(player,'pass_yards_2025')} yds, "
                f"Pass TDs {_v(player,'pass_tds_2025')}, "
                f"INTs {_v(player,'interceptions_2025')}, "
                f"Comp% {_fmt(player.get('completion_pct_2025'), 1)} "
                f"(CPOE {_fmt(player.get('cpoe_2025'), 2)}), "
                f"Passer rating {_fmt(player.get('passer_rating_2025'), 1)}, "
                f"Aggressiveness {_fmt(player.get('aggressiveness_2025'), 1)}, "
                f"Time to throw {_fmt(player.get('time_to_throw_2025'), 2)}s"
            )
        if player.get("pass_attempts_2024"):
            pass_lines.append(
                f"2024: {_v(player,'pass_attempts_2024')} att, "
                f"{_v(player,'pass_yards_2024')} yds, "
                f"Pass TDs {_v(player,'pass_tds_2024')}, "
                f"INTs {_v(player,'interceptions_2024')}, "
                f"Comp% {_fmt(player.get('completion_pct_2024'), 1)} "
                f"(CPOE {_fmt(player.get('cpoe_2024'), 2)}), "
                f"Passer rating {_fmt(player.get('passer_rating_2024'), 1)}, "
                f"Aggressiveness {_fmt(player.get('aggressiveness_2024'), 1)}"
            )
        pass_lines.append("")
        lines += pass_lines

    lines += [
        "INJURY HISTORY:",
        injury_line,
        "",
        "SIGNALS:",
        f"Trend: {trend.capitalize()}{trend_detail}",
        f"ADP Value 2025: {'Undervalued (sleeper)' if player.get('sleeper_signal') else 'Overvalued (bust risk)' if player.get('bust_signal') else 'Fair value'}",
        f"Sleeper signal: {sleeper_sig} | Bust signal: {bust_sig}",
        "",
        f"STATUS: {status} | Depth Chart: {depth_str} | Practice: {practice}",
        f"Injury Status: {injury_status}",
    ]

    return "\n".join(lines)


def _player_to_document(player: dict) -> Document:
    """Converts a merged player dict into a LangChain Document."""
    text = _player_to_text(player)
    computed = player.get("computed_rank_half_ppr") or 9999
    metadata = {
        "player_id":            str(player.get("player_id", "")),
        "name":                 player.get("full_name", ""),
        "position":             player.get("position") or "",
        "team":                 player.get("team") or "",
        "league_format":        player.get("league_format", "redraft"),
        "adp_2025":             float(player.get("adp_2025") or 999.0),
        "rank_standard_2026":   int(player.get("rank_standard_2026") or computed),
        "rank_half_ppr_2026":   int(player.get("rank_half_ppr_2026") or computed),
        "rank_ppr_2026":        int(player.get("rank_ppr_2026") or computed),
        "rank_dynasty_2026":      int(player.get("rank_dynasty_2026") or 9999),
        "pos_rank_half_ppr_2026": int(player.get("pos_rank_half_ppr_2026") or 999),
        "computed_rank_half_ppr": int(computed),
        "fantasy_points_half_ppr_2025": float(player.get("fantasy_points_half_ppr_2025") or 0),
        "fantasy_points_half_ppr_2024": float(player.get("fantasy_points_half_ppr_2024") or 0),
    }
    return Document(page_content=text, metadata=metadata)


def get_vectorstore() -> Chroma:
    """Returns the persistent Chroma vectorstore (creates if needed)."""
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=_get_embeddings(),
            persist_directory=CHROMA_PERSIST_DIR,
        )
    return _vectorstore


def build_vectorstore(documents: list[dict]) -> Chroma:
    """
    Takes merged player documents, converts to LangChain Documents,
    embeds them with all-MiniLM-L6-v2, and stores in persistent Chroma.
    """
    global _vectorstore

    logger.info(f"Building vectorstore from {len(documents)} player documents...")
    lc_docs = [_player_to_document(p) for p in documents]

    # Build in batches to avoid memory issues
    batch_size = 100
    embeddings = _get_embeddings()

    Path(CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)

    vs = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )

    # Clear existing collection
    try:
        existing = vs.get()
        if existing["ids"]:
            vs.delete(ids=existing["ids"])
            logger.info(f"Cleared {len(existing['ids'])} existing documents")
    except Exception as e:
        logger.warning(f"Could not clear existing collection: {e}")

    for i in range(0, len(lc_docs), batch_size):
        batch = lc_docs[i:i + batch_size]
        ids = [f"player_{doc.metadata['player_id']}" for doc in batch]
        vs.add_documents(documents=batch, ids=ids)
        logger.info(f"Embedded batch {i // batch_size + 1}/{(len(lc_docs) - 1) // batch_size + 1}")

    _vectorstore = vs
    logger.info(f"Vectorstore built with {len(lc_docs)} documents")
    return vs


def update_players(changed_players: list[dict]) -> None:
    """Upserts only changed players into Chroma (incremental update)."""
    if not changed_players:
        logger.info("No changed players to update")
        return

    vs = get_vectorstore()
    logger.info(f"Updating {len(changed_players)} players in Chroma...")

    ids = [f"player_{player.get('player_id', '')}" for player in changed_players]
    lc_docs = [_player_to_document(player) for player in changed_players]

    # Batch delete + add — one round trip each instead of two per player
    try:
        vs.delete(ids=ids)
    except Exception:
        pass  # some may not exist yet

    batch_size = 100
    for i in range(0, len(lc_docs), batch_size):
        vs.add_documents(documents=lc_docs[i:i + batch_size], ids=ids[i:i + batch_size])

    logger.info(f"Updated {len(changed_players)} players successfully")


def get_collection_count() -> int:
    """Returns the number of documents currently in the Chroma collection."""
    try:
        vs = get_vectorstore()
        # Native count — avoids fetching every document just to count them
        return vs._collection.count()
    except Exception as e:
        logger.warning(f"Could not get collection count: {e}")
        return 0
