"""
Top-level CLI script for running the full data refresh pipeline.

Usage:
    uv run python refresh.py
    uv run python refresh.py --format half_ppr
    uv run python refresh.py --all
    uv run python refresh.py --force
"""
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

ALL_FORMATS = ["redraft", "half_ppr", "ppr", "dynasty"]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fantasy GodBot data refresh pipeline")
    parser.add_argument(
        "--format",
        default="redraft",
        choices=ALL_FORMATS,
        help="League format (default: redraft)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Refresh all four formats (redraft, half_ppr, ppr, dynasty)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-embed all players regardless of changes (default: incremental)",
    )
    args = parser.parse_args()

    from src.ingestion.orchestrator import refresh_all

    formats = ALL_FORMATS if args.all else [args.format]

    for fmt in formats:
        logging.getLogger(__name__).info(f"=== Refreshing format: {fmt} ===")
        refresh_all(league_format=fmt, force=args.force)
