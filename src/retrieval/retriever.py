import logging
from langchain_core.documents import Document
from src.vectorstore.chroma_store import get_vectorstore

logger = logging.getLogger(__name__)


def get_retriever(league_format: str = "redraft", position_filter: str | None = None):
    """
    Returns a LangChain retriever scoped to the given league format.
    Optionally filters by position using Chroma metadata filtering.
    k=8 documents by default.
    """
    vs = get_vectorstore()

    search_kwargs: dict = {"k": 12}
    if position_filter:
        search_kwargs["filter"] = {"position": position_filter.upper()}

    return vs.as_retriever(search_type="similarity", search_kwargs=search_kwargs)


def retrieve_player(player_name: str) -> str:
    """
    Directly retrieves a specific player's document by name similarity search.
    Used for questions like 'tell me about CMC'.
    Returns the page_content of the best match, or a not-found message.
    """
    vs = get_vectorstore()
    results = vs.similarity_search(player_name, k=3)

    if not results:
        return f"No information found for '{player_name}'."

    # Check if first result is a strong match
    top = results[0]
    name_lower = player_name.lower().replace(".", "").strip()

    # Try to find the best name match in results
    for doc in results:
        stored_name = doc.metadata.get("name", "").lower()
        # Accept if query is a substring of stored name or vice versa
        if name_lower in stored_name or stored_name in name_lower:
            return doc.page_content

    return top.page_content


_FORMAT_RANK_FIELD = {
    "redraft":  "rank_half_ppr_2026",
    "half_ppr": "rank_half_ppr_2026",
    "ppr":      "rank_ppr_2026",
    "dynasty":  "rank_dynasty_2026",
}


def retrieve_top_players(n: int = 12, league_format: str = "redraft") -> list[Document]:
    """
    Retrieves the top-N ranked players using 2026 expert consensus rank when
    available, falling back to stat-computed rank when FP data is absent.
    Uses metadata filtering so ranking queries always return the actual
    highest-ranked players rather than a semantic similarity approximation.
    """
    rank_field = _FORMAT_RANK_FIELD.get(league_format, "rank_half_ppr_2026")
    vs = get_vectorstore()
    fetch_n = min(n * 3, 90)

    def _fetch_by_field(field: str, limit: int) -> list[Document]:
        results = vs.get(
            where={field: {"$lte": limit}},
            include=["documents", "metadatas"],
        )
        docs = [
            Document(page_content=c, metadata=m)
            for c, m in zip(results.get("documents", []), results.get("metadatas", []))
        ]
        docs.sort(key=lambda d: d.metadata.get(field, 9999))
        return docs[:n]

    try:
        docs = _fetch_by_field(rank_field, fetch_n)
        # If official ranks returned fewer than half the requested players,
        # fall back to computed rank (FP data unavailable)
        if len(docs) < max(1, n // 2):
            logger.info("Official ranks sparse (%d results), using computed rank", len(docs))
            docs = _fetch_by_field("computed_rank_half_ppr", fetch_n)
        return docs
    except Exception as e:
        logger.warning("Rank filter failed (%s), using computed rank fallback", e)
        try:
            return _fetch_by_field("computed_rank_half_ppr", fetch_n)
        except Exception:
            return vs.similarity_search("top fantasy players first round picks", k=n)


def retrieve_by_adp_range(pick_number: int, window: int = 5) -> list[Document]:
    """
    Retrieves players whose 2025 ADP is within `window` picks of pick_number.
    Used for 'who are alternatives at pick X' logic.
    """
    vs = get_vectorstore()

    low = max(1.0, pick_number - window)
    high = float(pick_number + window)

    # Chroma supports $gte / $lte for numeric metadata filters
    try:
        results = vs.get(
            where={
                "$and": [
                    {"adp_2025": {"$gte": low}},
                    {"adp_2025": {"$lte": high}},
                ]
            },
            include=["documents", "metadatas"],
        )
        docs = [
            Document(page_content=content, metadata=meta)
            for content, meta in zip(results.get("documents", []), results.get("metadatas", []))
        ]
        # Draft order reads better for the LLM than Chroma's arbitrary order;
        # cap so a wide round-range query can't flood the context window
        docs.sort(key=lambda d: d.metadata.get("adp_2025", 999.0))
        return docs[:40]
    except Exception as e:
        logger.warning(f"ADP range filter failed ({e}), falling back to similarity search")
        query = f"players available around pick {pick_number} ADP {pick_number}"
        return vs.similarity_search(query, k=8)
