from collections.abc import Sequence
from hashlib import file_digest
from pathlib import Path

from wiki_agents.concept.models import (
    ConceptChunkWork,
    ConceptExtractionChunk,
    ExtractedSourceConcepts,
)
from wiki_agents.concept.utils import (
    concept_chunk_id,
    concept_extraction_status,
    normalize_filename_stem,
)
from wiki_agents.document.models import AnchoredPage, SourceRef
from wiki_agents.embedding import make_concept_candidate


def build_concept_chunk_work(
    source: SourceRef,
    anchored_pages: Sequence[str],
    *,
    pages_per_chunk: int,
) -> list[ConceptChunkWork]:
    """Turn anchored pages into stable, independently executable work items."""
    work_items = []
    for start_index in range(0, len(anchored_pages), pages_per_chunk):
        pages = anchored_pages[start_index : start_index + pages_per_chunk]
        first_page = start_index + 1
        numbered_pages = [
            (first_page + page_offset, page)
            for page_offset, page in enumerate(pages)
            if page.strip()
        ]
        if not numbered_pages:
            continue

        page_numbers = [page_number for page_number, _ in numbered_pages]
        anchored_text = "\n\n".join(page for _, page in numbered_pages)
        work_items.append(
            ConceptChunkWork(
                chunk_id=concept_chunk_id(source, page_numbers, anchored_text),
                page_numbers=page_numbers,
                anchored_text=anchored_text,
            ),
        )
    return work_items


def make_source_ref(source_path: Path) -> SourceRef:
    """Build a stable source reference for one document."""
    paper_id = f"paper_{normalize_filename_stem(source_path.stem)}"
    with source_path.open("rb") as source_file:
        source_sha256 = file_digest(source_file, "sha256").hexdigest()
    return SourceRef(
        source_id=paper_id,
        sha256=source_sha256,
        resource=f"/references/papers/{paper_id}.md",
    )


def build_extracted_source(
    source: SourceRef,
    anchored_pages: Sequence[str],
    chunk_results: list[ConceptExtractionChunk],
) -> ExtractedSourceConcepts:
    """Build the public result from parsed pages and chunk outcomes."""
    candidates = [
        make_concept_candidate(source=source, concept=concept)
        for chunk in chunk_results
        if chunk.status == "succeeded" and chunk.extraction is not None
        for concept in chunk.extraction.concepts
    ]
    pages = [
        AnchoredPage(page_number=page_number, text=text)
        for page_number, text in enumerate(anchored_pages, start=1)
    ]
    return ExtractedSourceConcepts(
        source=source,
        pages=pages,
        candidates=candidates,
        concept_extraction_status=concept_extraction_status(chunk_results),
        concept_extraction_chunks=chunk_results,
    )
