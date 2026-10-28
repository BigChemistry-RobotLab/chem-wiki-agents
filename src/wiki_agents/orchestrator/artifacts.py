import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from operator import itemgetter
from pathlib import Path
from typing import cast

import pymupdf

from wiki_agents.concept.models import ConceptCandidate, ExtractedSourceConcepts
from wiki_agents.document.models import AnchoredPage, SourceRef


@dataclass(frozen=True, slots=True)
class PipelineSnapshotMetadata:
    """Pipeline identity persisted alongside extracted candidates."""

    configuration: Mapping[str, object]
    fingerprint: str


def build_extraction_snapshot(
    extracted: ExtractedSourceConcepts,
    *,
    pipeline: PipelineSnapshotMetadata,
    original_filename: str,
    artifact_resource: str,
    extracted_at: str,
) -> dict[str, object]:
    """Build the authoritative, portable pre-wiki candidate snapshot."""
    candidate_ids = [str(item.candidate_id) for item in extracted.candidates]
    chunks = []
    candidate_offset = 0
    for chunk in extracted.concept_extraction_chunks:
        count = len(chunk.extraction.concepts) if chunk.extraction is not None else 0
        chunks.append(
            {
                "chunk_id": chunk.chunk_id,
                "page_numbers": chunk.page_numbers,
                "candidate_ids": candidate_ids[
                    candidate_offset : candidate_offset + count
                ],
            },
        )
        candidate_offset += count
    return {
        "schema_version": 1,
        "extracted_at": extracted_at,
        "paper_id": extracted.source.source_id,
        "source_sha256": extracted.source.sha256,
        "original_filename": original_filename,
        "reference_resource": extracted.source.resource,
        "artifact_resource": artifact_resource,
        "parser_version": f"pymupdf:{pymupdf.VersionBind}",
        "models": pipeline.configuration["models"],
        "chunk_size": pipeline.configuration["chunk_size"],
        "prompt_version": pipeline.configuration["prompt_version"],
        "candidate_schema_version": pipeline.configuration["schema_version"],
        "pipeline_fingerprint": pipeline.fingerprint,
        "chunks": chunks,
        "candidates": [item.model_dump(mode="json") for item in extracted.candidates],
    }


def load_extraction_snapshot(path: Path) -> ExtractedSourceConcepts:
    """Restore extracted candidates from an authoritative snapshot."""
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    candidates = [
        ConceptCandidate.model_validate(candidate)
        for candidate in snapshot["candidates"]
    ]
    source = SourceRef(
        source_id=snapshot["paper_id"],
        sha256=snapshot["source_sha256"],
        resource=snapshot["reference_resource"],
    )
    return ExtractedSourceConcepts(
        source=source,
        pages=[],
        candidates=candidates,
        concept_extraction_status="succeeded",
        concept_extraction_chunks=[],
    )


def load_paper_titles(repository_path: Path) -> dict[str, str]:
    """Load paper titles from all extraction snapshots in a repository."""
    titles: dict[str, str] = {}
    for path in sorted((repository_path / "extractions").glob("*.json")):
        snapshot = read_json_object(path)
        if snapshot is not None:
            paper_id = str(snapshot["paper_id"])
            artifact = str(snapshot["artifact_resource"])
            titles[paper_id] = Path(artifact).stem
    return titles


def read_json_object(path: Path) -> dict[str, object] | None:
    """Read a JSON object, returning ``None`` when the file does not exist."""
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return cast("dict[str, object]", value)


def render_reference(
    *,
    title: str,
    original_filename: str,
    source_sha256: str,
    generated_at: str,
    anchored_pages: Sequence[AnchoredPage],
) -> str:
    """Render one OKF reference document and its anchored extracted text."""
    artifact_resource = f"/references/papers/artifacts/{title}.pdf"
    frontmatter = (
        "---\n"
        "type: Reference\n"
        f"title: {json.dumps(title)}\n"
        f"resource: {artifact_resource}\n"
        "generated:\n"
        "  by: wiki-agents/0.1.0\n"
        f"  at: {json.dumps(generated_at)}\n"
        "status: stable\n"
        f"source_sha256: {json.dumps(source_sha256)}\n"
        f"original_filename: {json.dumps(original_filename)}\n"
        f"extraction_method: {json.dumps(f'pymupdf:{pymupdf.VersionBind}')}\n"
        "---"
    )
    body = "\n\n".join(page.text for page in anchored_pages)
    return f"{frontmatter}\n" if not body else f"{frontmatter}\n\n{body}\n"


def render_paper_index(titles: Mapping[str, str]) -> str:
    """Render the reserved paper index with deterministic title-sorted links."""
    links = [
        f"- [{title}]({paper_id}.md)"
        for paper_id, title in sorted(titles.items(), key=itemgetter(1, 0))
    ]
    return "# Papers\n" if not links else "# Papers\n\n" + "\n".join(links) + "\n"
