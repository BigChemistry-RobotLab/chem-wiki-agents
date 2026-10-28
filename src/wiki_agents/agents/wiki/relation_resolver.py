"""Create the agent that resolves candidate relations."""

from agents import Agent, Model

from wiki_agents.agents.wiki.models import WikiRelationBatch


def create_wiki_relation_resolver_agent(
    *,
    model: str | Model | None = None,
) -> Agent[None]:
    """Create the typed agent that resolves bounded candidate-pair batches."""
    return Agent(
        name="Wiki work-package relation resolver",
        model=model,
        instructions="""Resolve every supplied candidate pair independently.

Return exactly one MERGE, LINK, or NONE result for each pair and copy each sorted
candidate ID pair exactly. MERGE means that separate pages would substantially
duplicate the central concept. LINK means that the concepts are distinct but need
a meaningful direct Wiki link. For LINK, provide one or two explicit directions.
Each directed link must use only the two candidate IDs from the same result. Never
copy a candidate ID from another pair in the batch. The link source and target must
be the two different IDs from that result.
Do not make a relationship bidirectional unless both directions are useful. NONE
means that no merge or planned direct link is required. Use claims only to resolve
semantic ambiguity. Do not add scientific knowledge.
""",
        output_type=WikiRelationBatch,
    )
