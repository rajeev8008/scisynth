from __future__ import annotations

import argparse
import logging


def main() -> None:
    """Run SCISYNTH commands: API server or ingestion pipeline.

    Args:
        None.
    Returns:
        None.
    """
    parser = argparse.ArgumentParser(prog="scisynth")
    parser.add_argument(
        "command",
        nargs="?",
        default="serve",
        choices=("serve", "ingest"),
        help="serve API (default) or ingest configured dataset.",
    )
    parser.add_argument(
        "--source",
        dest="dataset_source",
        choices=("local", "arxiv", "huggingface"),
        default=None,
        help="override DATASET_SOURCE for ingest",
    )
    parser.add_argument("--dataset-id", dest="dataset_id", default=None, help="override DATASET_ID")
    parser.add_argument(
        "--profile",
        dest="dataset_profile",
        choices=("fixture", "full"),
        default=None,
        help="override DATASET_PROFILE for local source",
    )
    parser.add_argument("--arxiv-query", dest="arxiv_query", default=None)
    parser.add_argument("--arxiv-max-results", dest="arxiv_max_results", type=int, default=None)
    hf_choices = ("qasper", "scifact_corpus")
    parser.add_argument("--hf-preset", dest="hf_preset", choices=hf_choices, default=None)
    parser.add_argument("--hf-split", dest="hf_split", default=None)
    parser.add_argument("--hf-max-rows", dest="hf_max_rows", type=int, default=None)
    args = parser.parse_args()

    if args.command == "ingest":
        _run_ingestion_command(args)
        return
    _run_serve_command()


def _run_serve_command() -> None:
    """Start the FastAPI server with configured host/port.

    Args:
        None.
    Returns:
        None.
    """
    import uvicorn

    logging.basicConfig(level=logging.INFO)

    from scisynth.config import get_settings

    s = get_settings()
    uvicorn.run(
        "scisynth.api.main:app",
        host=s.api_host,
        port=s.api_port,
        factory=False,
    )


def _run_ingestion_command(args: argparse.Namespace) -> None:
    """Run ingestion and log summary counts and paths.

    Args:
        args: Parsed CLI arguments including optional overrides.
    Returns:
        None.
    """
    logging.basicConfig(level=logging.INFO)
    from scisynth.config import get_settings
    from scisynth.ingestion import run_ingestion

    base = get_settings()
    updates = {
        key: value
        for key, value in {
            "dataset_source": args.dataset_source,
            "dataset_id": args.dataset_id,
            "dataset_profile": args.dataset_profile,
            "arxiv_query": args.arxiv_query,
            "arxiv_max_results": args.arxiv_max_results,
            "hf_preset": args.hf_preset,
            "hf_split": args.hf_split,
            "hf_max_rows": args.hf_max_rows,
        }.items()
        if value is not None
    }
    stats = run_ingestion(base.model_copy(update=updates))
    logging.getLogger(__name__).info(
        "Ingested dataset=%s docs=%s chunks=%s output=%s",
        stats.dataset_id,
        stats.document_count,
        stats.chunk_count,
        stats.output_path,
    )
