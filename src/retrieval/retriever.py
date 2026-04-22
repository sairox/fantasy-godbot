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

    search_kwargs: dict = {"k": 8}
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
    doc_name = top.metadata.get("name", "").lower()

    # Try to find the best name match in results
    for doc in results:
        stored_name = doc.metadata.get("name", "").lower()
        # Accept if query is a substring of stored name or vice versa
        if name_lower in stored_name or stored_name in name_lower:
            return doc.page_content

    return top.page_content


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
        docs = []
        for content, meta in zip(results.get("documents", []), results.get("metadatas", [])):
            docs.append(Document(page_content=content, metadata=meta))
        return docs
    except Exception as e:
        logger.warning(f"ADP range filter failed ({e}), falling back to similarity search")
        query = f"players available around pick {pick_number} ADP {pick_number}"
        return vs.similarity_search(query, k=8)
