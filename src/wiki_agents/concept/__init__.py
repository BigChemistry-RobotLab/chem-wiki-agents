from typing import TYPE_CHECKING

from wiki_agents.concept.config import ConceptExtractionConfig
from wiki_agents.concept.exceptions import ConceptExtractionExhausted
from wiki_agents.concept.models import (
    ConceptChunkWork,
    ExtractedSourceConcepts,
)
from wiki_agents.concept.store import ConceptStore
from wiki_agents.concept.utils import normalize_filename_stem

if TYPE_CHECKING:
    from wiki_agents.concept.pipeline import ConceptExtractionPipeline
    from wiki_agents.concept.workflow import ConceptExtractionWorkflow

__all__ = [
    "ConceptChunkWork",
    "ConceptExtractionConfig",
    "ConceptExtractionExhausted",
    "ConceptExtractionPipeline",
    "ConceptExtractionWorkflow",
    "ConceptStore",
    "ExtractedSourceConcepts",
    "normalize_filename_stem",
]


def __getattr__(name: str) -> object:
    """Load public concept APIs without eagerly importing the whole pipeline."""
    if name == "ConceptExtractionPipeline":
        from wiki_agents.concept.pipeline import (  # ruff: ignore[import-outside-top-level]
            ConceptExtractionPipeline,
        )

        return ConceptExtractionPipeline
    if name == "ConceptExtractionWorkflow":
        from wiki_agents.concept.workflow import (  # ruff: ignore[import-outside-top-level]
            ConceptExtractionWorkflow,
        )

        return ConceptExtractionWorkflow
    raise AttributeError(name)
