from collections.abc import Sequence

from wiki_agents.orchestrator.stages.common import (
    build_stage_parser,
    create_orchestrator,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Extract concepts and save authoritative extraction snapshots."""
    parser = build_stage_parser(
        "Extract and synchronize concepts from source PDFs.",
        include_sources=True,
    )
    with create_orchestrator(parser, argv) as orchestrator:
        orchestrator.extract_concepts()
    return 0
