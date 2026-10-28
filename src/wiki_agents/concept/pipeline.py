import json
import logging
from collections.abc import Sequence
from concurrent.futures import Executor
from hashlib import sha256
from pathlib import Path

from wiki_agents.concept.assembly import (
    build_concept_chunk_work,
    build_extracted_source,
    make_source_ref,
)
from wiki_agents.concept.checkpoints import ConceptCheckpointStore
from wiki_agents.concept.chunk_runner import ConceptChunkRunner
from wiki_agents.concept.config import ConceptExtractionConfig
from wiki_agents.concept.models import (
    ConceptCandidate,
    ConceptContext,
    ConceptExtraction,
    ConceptExtractionRetryState,
    ExtractedSourceConcepts,
)
from wiki_agents.concept.store import ConceptStore

# Keep operational messages under the established logger after this extraction.
logger = logging.getLogger("wiki_agents.orchestrator")


class ConceptExtractionPipeline:
    """Reusable concept extraction, checkpointing, and retrieval capability.

    The caller provides document parsing, one-chunk agent execution, and the
    backing concept store. This boundary lets different orchestrators share
    extraction without using inheritance.
    """

    def __init__(self, config: ConceptExtractionConfig) -> None:
        """Create a concept extraction pipeline from one configuration object."""
        self.config = config
        self._document_parser = config.document_parser
        self._concept_extractor = config.concept_extractor
        self._concept_store_provider = config.concept_store_provider
        self.metadata = self._pipeline_metadata()
        self.fingerprint = sha256(
            json.dumps(
                self.metadata,
                sort_keys=True,
                separators=(",", ":"),
            ).encode(),
        ).hexdigest()
        self._checkpoint_store = ConceptCheckpointStore(
            config.checkpoints_path,
            self.fingerprint,
        )

    def parse_document(self, path: str | Path) -> list[str]:
        """Parse a source document into anchored pages."""
        return self._document_parser(path)

    def run_concept_extraction(
        self,
        anchored_text: str,
        *,
        retry_state: ConceptExtractionRetryState | None = None,
    ) -> ConceptExtraction:
        """Extract and review concepts from one anchored chunk."""
        return self._concept_extractor(anchored_text, retry_state=retry_state)

    def _get_concept_store(self) -> ConceptStore:
        """Return the configured concept store."""
        return self._concept_store_provider()

    def extract_concepts(
        self,
        path: str | Path,
        *,
        executor: Executor | None = None,
    ) -> ExtractedSourceConcepts:
        """Parse one PDF and extract its concepts, reusing saved checkpoints."""
        source_path = Path(path)
        source = make_source_ref(source_path)
        anchored_pages = self.parse_document(source_path)
        work_items = build_concept_chunk_work(
            source,
            anchored_pages,
            pages_per_chunk=self.config.num_concept_pages,
        )

        logger.info(
            "Extracting concepts from %s (source %s): %d pages split into %d chunks "
            "of up to %d pages",
            source_path.name,
            source.source_id[:12],
            len(anchored_pages),
            len(work_items),
            self.config.num_concept_pages,
        )

        chunk_results = ConceptChunkRunner(
            self.run_concept_extraction,
            self._checkpoint_store,
        ).run(source, work_items, executor=executor)
        result = build_extracted_source(source, anchored_pages, chunk_results)
        succeeded_chunks = sum(chunk.status == "succeeded" for chunk in chunk_results)
        failed_chunks = len(chunk_results) - succeeded_chunks
        logger.info(
            "Concept extraction for %s (source %s) finished with status=%s: "
            "%d chunks succeeded, %d failed, %d concepts retained",
            source_path.name,
            source.source_id[:12],
            result.concept_extraction_status,
            succeeded_chunks,
            failed_chunks,
            len(result.candidates),
        )
        return result

    def update_concept_embeddings(
        self,
        extracted_sources: Sequence[ExtractedSourceConcepts],
    ) -> None:
        """Add or update embeddings after all corpus concepts are extracted."""
        store = self._get_concept_store()
        for extracted_source in extracted_sources:
            store.replace_source_candidates(
                extracted_source.source,
                extracted_source.candidates,
            )

    def prepare_wiki_input(self, candidate: ConceptCandidate) -> ConceptContext:
        """Prepare one extracted concept with corpus-wide similar concepts."""
        return ConceptContext(
            candidate=candidate,
            similar_concepts=self._get_concept_store().top_k_similar_concepts(
                candidate,
                top_k=self.config.similar_concepts_top_k,
            ),
        )

    def _pipeline_metadata(self) -> dict[str, object]:
        """Describe every configured input that can affect extraction output."""
        return {
            "parser": self.config.parser_version,
            "models": {
                "extractor": self.config.extractor_model,
                "critic": self.config.critic_model,
                "fixer": self.config.fixer_model,
            },
            "chunk_size": self.config.num_concept_pages,
            "prompt_version": self.config.prompt_version,
            "schema_version": self.config.schema_version,
        }
