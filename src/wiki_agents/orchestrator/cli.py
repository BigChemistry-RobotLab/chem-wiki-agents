import argparse
from collections.abc import Sequence
from pathlib import Path

from wiki_agents.logging_config import configure_logging
from wiki_agents.orchestrator.wiki import WikiOrchestrator, WikiOrchestratorConfig


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line wiki build workflow."""
    parser = argparse.ArgumentParser(description="Run the wiki build workflow.")
    parser.add_argument(
        "repository_path",
        type=Path,
        help="Path to the local wiki repository.",
    )
    parser.add_argument(
        "--sources-path",
        type=Path,
        help=(
            "Path to the folder containing PDF sources; defaults to the "
            "repository's sources/ folder."
        ),
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Log counts, percentages, elapsed time, and periodic heartbeats.",
    )
    args = parser.parse_args(argv)

    configure_logging()
    with WikiOrchestrator(
        WikiOrchestratorConfig.from_env(
            repository_path=args.repository_path,
            sources_path=args.sources_path,
        ),
        progress=args.progress,
    ) as orchestrator:
        orchestrator.run()
    return 0
