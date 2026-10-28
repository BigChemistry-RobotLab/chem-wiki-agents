from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from wiki_agents.concept.models import (
    ConceptExtraction,
    ConceptExtractionRetryState,
)
from wiki_agents.concept.store import ConceptStore


class DocumentParser(Protocol):
    """Define the document parser interface."""

    def __call__(self, path: str | Path) -> list[str]:
        """Parse one source document into anchored pages."""
        ...


class ConceptExtractor(Protocol):
    """Define the concept extractor interface."""

    def __call__(
        self,
        anchored_text: str,
        /,
        *,
        retry_state: ConceptExtractionRetryState | None = None,
    ) -> ConceptExtraction:
        """Extract and review concepts from one anchored chunk."""
        ...


class ConceptStoreProvider(Protocol):
    """Define the concept store provider interface."""

    def __call__(self) -> ConceptStore:
        """Return the configured concept store."""
        ...


@dataclass(frozen=True, slots=True)
class ConceptExtractionConfig:
    """Configure a concept extraction pipeline."""

    checkpoints_path: Path
    extractor_model: str
    critic_model: str
    fixer_model: str
    num_concept_pages: int
    document_parser: DocumentParser
    concept_extractor: ConceptExtractor
    concept_store_provider: ConceptStoreProvider
    similar_concepts_top_k: int = 5
    parser_version: str = "pymupdf"
    prompt_version: str = "concept-extraction-v1"
    schema_version: str = "concept-candidate-v1"

    def __post_init__(self) -> None:
        """Validate extraction settings."""
        if self.num_concept_pages < 1:
            raise ValueError("num_concept_pages must be at least 1")
        if self.similar_concepts_top_k < 1:
            raise ValueError("similar_concepts_top_k must be at least 1")
        if not self.parser_version:
            raise ValueError("parser_version must not be empty")
        if not self.prompt_version:
            raise ValueError("prompt_version must not be empty")
        if not self.schema_version:
            raise ValueError("schema_version must not be empty")
