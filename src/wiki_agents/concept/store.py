"""Define concept persistence and retrieval interfaces."""

from collections.abc import Sequence
from typing import Protocol

from wiki_agents.concept.models import ConceptCandidate, ConceptMatch
from wiki_agents.document.models import SourceRef


class ConceptStore(Protocol):
    """Persistence and similarity operations used by an extraction pipeline."""

    def replace_source_candidates(
        self,
        source: SourceRef,
        candidates: Sequence[ConceptCandidate],
    ) -> None: ...

    def top_k_similar_concepts(
        self,
        candidate: ConceptCandidate,
        *,
        top_k: int,
    ) -> list[ConceptMatch]: ...
