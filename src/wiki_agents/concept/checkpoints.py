import logging
from collections.abc import Sequence
from pathlib import Path

from wiki_agents.concept.models import (
    ConceptExtractionCheckpoint,
    ConceptExtractionChunk,
)
from wiki_agents.document.models import SourceRef

logger = logging.getLogger("wiki_agents.orchestrator")


class ConceptCheckpointStore:
    """Load and save concept extraction checkpoints."""

    def __init__(self, checkpoints_path: Path, fingerprint: str) -> None:
        """Create a checkpoint store for one pipeline configuration."""
        self._checkpoints_path = checkpoints_path
        self._fingerprint = fingerprint

    def path_for(self, source: SourceRef) -> Path:
        """Return the organized concept extraction checkpoint path."""
        directory = (
            self._checkpoints_path
            / "concepts"
            / source.source_id
            / source.sha256
            / self._fingerprint
        )
        directory.mkdir(parents=True, exist_ok=True)
        return directory / "extraction.json"

    def load(self, source: SourceRef) -> ConceptExtractionCheckpoint:
        """Load a source checkpoint or return an empty checkpoint."""
        path = self.path_for(source)
        if not path.exists():
            return ConceptExtractionCheckpoint(source=source, chunks=[])

        checkpoint = ConceptExtractionCheckpoint.model_validate_json(path.read_text())
        if (
            checkpoint.source.source_id != source.source_id
            or checkpoint.source.sha256 != source.sha256
        ):
            raise ValueError(f"Concept checkpoint source mismatch in {path}")
        return checkpoint

    def save(
        self,
        source: SourceRef,
        chunks: Sequence[ConceptExtractionChunk],
    ) -> None:
        """Save one source checkpoint with an atomic file replacement."""
        path = self.path_for(source)
        temporary_path = path.with_suffix(".json.tmp")
        checkpoint = ConceptExtractionCheckpoint(source=source, chunks=list(chunks))
        temporary_path.write_text(checkpoint.model_dump_json(indent=2) + "\n")
        temporary_path.replace(path)
        logger.debug("Saved %d concept chunk outcomes to %s", len(chunks), path)

    def save_incremental(
        self,
        source: SourceRef,
        current_results: Sequence[ConceptExtractionChunk],
        previous_chunks: dict[str, ConceptExtractionChunk],
    ) -> None:
        """Save current results and retain work that this run did not visit."""
        completed_ids = {chunk.chunk_id for chunk in current_results}
        unvisited_results = [
            chunk
            for chunk_id, chunk in previous_chunks.items()
            if chunk_id not in completed_ids
        ]
        self.save(source, [*current_results, *unvisited_results])
