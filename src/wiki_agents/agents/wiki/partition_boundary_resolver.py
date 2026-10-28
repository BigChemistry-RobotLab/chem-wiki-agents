"""Create the agent that resolves page-partition boundary relations."""

from agents import Agent, Model

from wiki_agents.agents.wiki.models import WikiRelationBatch


def create_wiki_partition_boundary_resolver_agent(
    *,
    model: str | Model | None = None,
) -> Agent[None]:
    """Create the agent that resolves relations across a page partition."""
    return Agent(
        name="Wiki page-partition boundary resolver",
        model=model,
        instructions="""Resolve every supplied candidate pair independently.

The candidates were in one provisional MERGE component, but the page planner put
them on different coherent pages. Return exactly one LINK or NONE result for each
pair and copy each sorted candidate ID pair exactly. Never return MERGE. LINK means
that the distinct pages need a meaningful direct Wiki link. For LINK, provide one or
two explicit directions that use only the two IDs in the result. NONE means that no
planned direct link is required. Use claims only to resolve semantic ambiguity. Do
not add scientific knowledge.
""",
        output_type=WikiRelationBatch,
    )
