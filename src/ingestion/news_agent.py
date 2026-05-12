"""
News ingestion agent — stub for v1.
Will scrape injury reports, beat reporter updates, and transaction news in v2.
"""
import logging

logger = logging.getLogger(__name__)


def fetch_news(player_name: str | None = None) -> list[dict]:
    """Stub: returns empty list until v2 news scraping is implemented."""
    logger.debug("news_agent.fetch_news called (stub — not yet implemented)")
    return []


def save_news(news: list[dict]) -> None:
    """Stub."""
    pass
