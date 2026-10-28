from collections.abc import Sequence

from wiki_agents.orchestrator.stages.common import (
    build_stage_parser,
    create_orchestrator,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Write, validate, and publish Wiki pages from saved plans."""
    parser = build_stage_parser(
        "Write and publish Wiki pages from saved plans.",
        include_sources=False,
    )
    with create_orchestrator(parser, argv) as orchestrator:
        orchestrator.write_wiki()
    return 0
