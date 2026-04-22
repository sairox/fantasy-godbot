import logging
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

CHROMA_PERSIST_DIR = str(Path(__file__).parent.parent.parent / "data" / "chroma")
COLLECTION_NAME = "fantasy_players"

_embeddings = None
_vectorstore = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    """Returns cached HuggingFace embeddings (all-MiniLM-L6-v2)."""
    global _embeddings
    if _embeddings is None:
        logger.info("Loading HuggingFace embeddings model (all-MiniLM-L6-v2)...")
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def _player_to_text(player: dict) -> str:
    """
    Converts a merged player document to rich natural-language text for embedding.
    Semantic search works far better on prose than raw JSON.
    """
    name = player.get("full_name", "Unknown")
    pos = player.get("position", "")
    team = player.get("team", "N/A")
    age = player.get("age", "?")
    exp = player.get("years_exp", "?")

    # rankings
    rk_std = player.get("rank_standard_2026") or "unranked"
    rk_half = player.get("rank_half_ppr_2026") or "unranked"
    rk_ppr = player.get("rank_ppr_2026") or "unranked"
    rk_dyn = player.get("rank_dynasty_2026") or "unranked"

    # 2025 performance
    fpts_25 = player.get("fantasy_points_half_ppr_2025") or "N/A"
    ppg_25 = player.get("points_per_game_2025") or "N/A"
    gp_25 = player.get("games_played_2025") or "N/A"
    gm_25 = player.get("games_missed_2025") or 0
    fin_25 = player.get("finish_rank_half_ppr_2025") or "N/A"

    # ADP
    adp = player.get("adp_2025") or "N/A"
    ecr_vs_adp = player.get("ecr_vs_adp_2025")
    val_vs_adp = player.get("value_vs_adp_2025")

    adp_line = f"Drafted at ADP {adp}"
    if ecr_vs_adp is not None:
        if ecr_vs_adp < -3:
            adp_line += f" — experts ranked {abs(ecr_vs_adp):.1f} spots higher (undervalued)"
        elif ecr_vs_adp > 3:
            adp_line += f" — experts ranked {ecr_vs_adp:.1f} spots lower (overvalued)"
        else:
            adp_line += " — close to expert consensus"
    if val_vs_adp is not None:
        if val_vs_adp < 0:
            adp_line += f". Outperformed ADP by {abs(val_vs_adp):.0f} spots."
        elif val_vs_adp > 0:
            adp_line += f". Underperformed ADP by {val_vs_adp:.0f} spots."

    # 2024 performance
    fpts_24 = player.get("fantasy_points_half_ppr_2024") or "N/A"
    gp_24 = player.get("games_played_2024") or "N/A"
    gm_24 = player.get("games_missed_2024") or 0
    fin_24 = player.get("finish_rank_half_ppr_2024") or "N/A"

    # injury
    injury_risk = player.get("injury_risk_score", "unknown")
    injury_status = player.get("injury_status") or "None"
    injury_part = player.get("injury_body_part") or ""

    injury_line = f"Injury risk: {injury_risk.capitalize()}"
    if gm_24:
        injury_line += f" — missed {gm_24} games in 2024"
    if gm_25:
        injury_line += f", {gm_25} in 2025"
    if injury_part:
        injury_line += f" ({injury_part})"

    # signals
    trend = player.get("trend", "unknown")
    sleeper_sig = "Yes" if player.get("sleeper_signal") else "No"
    bust_sig = "Yes" if player.get("bust_signal") else "No"

    trend_detail = ""
    if fin_24 != "N/A" and fin_25 != "N/A":
        trend_detail = f" ({pos}{fin_24} in 2024 → {pos}{fin_25} in 2025)"

    # depth / status
    depth = player.get("depth_chart_order")
    depth_str = f"Starter ({depth})" if depth == 1 else f"Depth #{depth}" if depth else "Unknown"
    status = player.get("status", "Unknown")
    practice = player.get("practice_participation") or "N/A"

    lines = [
        f"{name} | {pos} | {team} | Age: {age} | Experience: {exp} years",
        "",
        "2026 DRAFT RANKINGS:",
        f"Standard: {pos}{rk_std} (Overall: {rk_std}) | Half PPR: {pos}{rk_half} (Overall: {rk_half}) | PPR: {pos}{rk_ppr} (Overall: {rk_ppr}) | Dynasty: {pos}{rk_dyn}",
        "",
        "2025 PERFORMANCE:",
        f"Fantasy Points (Half PPR): {fpts_25} | Points Per Game: {ppg_25}",
        f"Games Played: {gp_25} of 17 (missed {gm_25}) | Finish: {pos}{fin_25}",
        adp_line,
        "",
        "2024 PERFORMANCE:",
        f"Fantasy Points (Half PPR): {fpts_24} | Points Per Game: N/A",
        f"Games Played: {gp_24} of 17 (missed {gm_24}) | Finish: {pos}{fin_24}",
        "",
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
    metadata = {
        "player_id": str(player.get("player_id", "")),
        "name": player.get("full_name", ""),
        "position": player.get("position", ""),
        "team": player.get("team", ""),
        "league_format": player.get("league_format", "redraft"),
        "adp_2025": player.get("adp_2025") or 999.0,
        "rank_half_ppr_2026": player.get("rank_half_ppr_2026") or 9999,
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

    for player in changed_players:
        player_id = str(player.get("player_id", ""))
        doc_id = f"player_{player_id}"

        # Delete old document
        try:
            vs.delete(ids=[doc_id])
        except Exception:
            pass  # may not exist yet

        # Add new document
        lc_doc = _player_to_document(player)
        vs.add_documents(documents=[lc_doc], ids=[doc_id])

    logger.info(f"Updated {len(changed_players)} players successfully")


def get_collection_count() -> int:
    """Returns the number of documents currently in the Chroma collection."""
    try:
        vs = get_vectorstore()
        result = vs.get()
        return len(result["ids"])
    except Exception as e:
        logger.warning(f"Could not get collection count: {e}")
        return 0
