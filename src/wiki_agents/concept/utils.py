import re
from collections.abc import Sequence
from hashlib import sha256
from typing import Literal

from wiki_agents.concept.models import ConceptExtractionChunk
from wiki_agents.document.models import SourceRef


def normalize_filename_stem(stem: str) -> str:
    """Normalize a source stem without interpreting domain identifiers."""
    normalized = re.sub(r"[^a-z0-9._-]+", "_", stem.lower())
    normalized = re.sub(r"_+", "_", normalized).strip("._-")
    if not normalized:
        raise ValueError(f"PDF filename has no usable normalized stem: {stem!r}")
    return normalized


def concept_chunk_id(
    source: SourceRef,
    page_numbers: Sequence[int],
    anchored_text: str,
) -> str:
    """Build a source-scoped identifier for an anchored page chunk."""
    pages = f"{page_numbers[0]:04d}-{page_numbers[-1]:04d}"
    content_hash = sha256(anchored_text.encode()).hexdigest()[:16]
    return f"{source.source_id}:pages-{pages}:{content_hash}"


def page_range(page_numbers: Sequence[int]) -> str:
    """Format ordered page numbers for concise operational messages."""
    if len(page_numbers) == 1:
        return str(page_numbers[0])
    return f"{page_numbers[0]}-{page_numbers[-1]}"


def concept_extraction_status(
    chunks: Sequence[ConceptExtractionChunk],
) -> Literal["succeeded", "partial", "failed"]:
    """Summarize chunk outcomes as a document-level extraction status."""
    failures = sum(chunk.status == "failed" for chunk in chunks)
    if failures == 0:
        return "succeeded"
    if failures == len(chunks):
        return "failed"
    return "partial"
