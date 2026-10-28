"""Utilities for constructing embedding records."""

import json
from uuid import NAMESPACE_URL, uuid5

from wiki_agents.concept.models import Concept, ConceptCandidate
from wiki_agents.document.models import SourceRef


def make_concept_candidate(
    *,
    source: SourceRef,
    concept: Concept,
) -> ConceptCandidate:
    """Create a candidate with a deterministic ID.

    Args:
        source: Source document reference.
        concept: Extracted concept.

    Returns:
        A validated concept candidate with a deterministic ID.

    """
    canonical = json.dumps(
        concept.model_dump(mode="json", exclude_none=False),
        sort_keys=True,
        separators=(",", ":"),
    )
    point_id = uuid5(
        NAMESPACE_URL,
        f"concept:{source.source_id}:{source.sha256}:{canonical}",
    )
    return ConceptCandidate(
        candidate_id=point_id,
        source=source,
        concept=concept,
    )
