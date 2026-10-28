"""Embedding providers, formats, and concept vector storage."""

from wiki_agents.embedding.formats import (
    CONCEPT_DOCUMENT_FORMAT,
    QUERY_FORMAT,
    WORK_PACKAGE_FORMAT,
    format_concept_document,
    format_query,
    format_work_package,
)
from wiki_agents.embedding.provider import (
    EmbeddingProvider,
    OpenAICompatibleEmbeddingProvider,
)
from wiki_agents.embedding.store import ConceptVectorStore
from wiki_agents.embedding.utils import make_concept_candidate
from wiki_agents.embedding.work_package_store import (
    WorkPackageMatch,
    WorkPackageVectorStore,
)

__all__ = [
    "CONCEPT_DOCUMENT_FORMAT",
    "QUERY_FORMAT",
    "WORK_PACKAGE_FORMAT",
    "ConceptVectorStore",
    "EmbeddingProvider",
    "OpenAICompatibleEmbeddingProvider",
    "WorkPackageMatch",
    "WorkPackageVectorStore",
    "format_concept_document",
    "format_query",
    "format_work_package",
    "make_concept_candidate",
]
