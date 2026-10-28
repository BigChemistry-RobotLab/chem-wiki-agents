"""Text formats used to create concept and query embeddings."""

from wiki_agents.agents.wiki.models import WikiPageIntent
from wiki_agents.concept.models import Concept

# TODO: Parameterize these formats for the embedding provider and store so that
# they can be changed without code changes.
# At the moment, these are specific to the EmbeddingGemma 300M model.
CONCEPT_DOCUMENT_FORMAT = "title: {title} | text: {text}"
QUERY_FORMAT = "task: search result | query: {text}"
WORK_PACKAGE_FORMAT = "subject: {subject}\nscope: {scope}\nwork: {work}"


def format_concept_document(concept: Concept) -> str:
    """Format a concept for storage.

    Args:
        concept: Extracted concept to format.

    Returns:
        Text for the document embedding.

    """
    return CONCEPT_DOCUMENT_FORMAT.format(
        title=concept.title,
        text=concept.description,
    )


def format_query(text: str) -> str:
    """Format text for a search query.

    Args:
        text: Concept or claim text.

    Returns:
        Text for the query embedding.

    """
    return QUERY_FORMAT.format(text=text)


def format_work_package(intent: WikiPageIntent) -> str:
    """Format only the normalized semantic intent for similarity search."""
    return WORK_PACKAGE_FORMAT.format(
        subject=intent.canonical_subject,
        scope=intent.scope,
        work="; ".join(intent.work_summary),
    )
