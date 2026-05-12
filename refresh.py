"""
Top-level CLI script for running the full data refresh pipeline.

Usage:
    uv run python refresh.py
    uv run python refresh.py --format dynasty
    uv run python refresh.py --force
"""
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fantasy GodBot data refresh pipeline")
    parser.add_argument(
        "--format",
        default="redraft",
        choices=["redraft", "half_ppr", "ppr", "dynasty"],
        help="League format (default: redraft)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-embed all players regardless of changes (default: incremental)",
    )
    args = parser.parse_args()

    from src.ingestion.orchestrator import refresh_all
    refresh_all(league_format=args.format, force=args.force)
