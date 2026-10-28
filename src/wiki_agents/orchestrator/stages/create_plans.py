from collections.abc import Sequence

from wiki_agents.orchestrator.stages.common import (
    build_stage_parser,
    create_orchestrator,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Create candidate, relation, page, and graph plans."""
    parser = build_stage_parser(
        "Create Wiki plans from saved concept extractions.",
        include_sources=False,
    )
    with create_orchestrator(parser, argv) as orchestrator:
        orchestrator.create_plans()
    return 0
