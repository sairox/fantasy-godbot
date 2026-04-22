"""
YouTube transcript agent — stub for v1.
Will ingest fantasy football YouTube analysis transcripts in v2.
"""
import logging

logger = logging.getLogger(__name__)


def fetch_transcripts(channel_ids: list[str] | None = None) -> list[dict]:
    """Stub: returns empty list until v2 YouTube ingestion is implemented."""
    logger.debug("youtube_agent.fetch_transcripts called (stub — not yet implemented)")
    return []


def save_transcripts(transcripts: list[dict]) -> None:
    """Stub."""
    pass
