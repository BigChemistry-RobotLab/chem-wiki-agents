import argparse
from collections.abc import Sequence
from pathlib import Path

from wiki_agents.logging_config import configure_logging
from wiki_agents.orchestrator.config import WikiOrchestratorConfig
from wiki_agents.orchestrator.wiki import WikiOrchestrator


def build_stage_parser(
    description: str, *, include_sources: bool,
) -> argparse.ArgumentParser:
    """Create the shared parser for one Wiki pipeline stage.

    Args:
        description: Short command description.
        include_sources: Whether to accept a source PDF directory.

    Returns:
        Parser with the shared stage arguments.

    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "repository_path",
        type=Path,
        help="Path to the local Wiki repository.",
    )
    if include_sources:
        parser.add_argument(
            "--sources-path",
            type=Path,
            help=(
                "Path to the folder that contains PDF sources. The default is "
                "the sources/ folder in the repository."
            ),
        )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Log counts, percentages, elapsed time, and periodic heartbeats.",
    )
    return parser


def create_orchestrator(
    parser: argparse.ArgumentParser,
    argv: Sequence[str] | None,
) -> WikiOrchestrator:
    """Create an orchestrator from shared command-line arguments.

    Args:
        parser: Parser for the selected stage.
        argv: Optional argument sequence.

    Returns:
        Configured Wiki orchestrator.

    """
    args = parser.parse_args(argv)
    configure_logging()
    return WikiOrchestrator(
        WikiOrchestratorConfig.from_env(
            repository_path=args.repository_path,
            sources_path=getattr(args, "sources_path", None),
        ),
        progress=args.progress,
    )
