import logging
from collections.abc import Sequence
from concurrent.futures import Executor, Future, as_completed

from wiki_agents.concept.checkpoints import ConceptCheckpointStore
from wiki_agents.concept.config import ConceptExtractor
from wiki_agents.concept.exceptions import ConceptExtractionExhausted
from wiki_agents.concept.models import ConceptChunkWork, ConceptExtractionChunk
from wiki_agents.concept.utils import page_range
from wiki_agents.document.models import SourceRef
from wiki_agents.tracing import (
    pdf_source_metadata,
    record_trace_event,
    traced_workflow,
)

logger = logging.getLogger("wiki_agents.orchestrator")


class ConceptChunkRunner:
    """Run concept chunks and preserve resumable progress."""

    def __init__(
        self,
        extractor: ConceptExtractor,
        checkpoint_store: ConceptCheckpointStore,
    ) -> None:
        """Create a runner with extraction and checkpoint dependencies."""
        self._extractor = extractor
        self._checkpoint_store = checkpoint_store

    def run(
        self,
        source: SourceRef,
        work_items: Sequence[ConceptChunkWork],
        *,
        executor: Executor | None = None,
    ) -> list[ConceptExtractionChunk]:
        """Run pending chunks and reuse successful results."""
        if executor is None:
            return self._run_sequentially(source, work_items)
        return self._run_concurrently(source, work_items, executor)

    def _run_sequentially(
        self,
        source: SourceRef,
        work_items: Sequence[ConceptChunkWork],
    ) -> list[ConceptExtractionChunk]:
        """Run pending chunks sequentially."""
        checkpoint = self._checkpoint_store.load(source)
        previous_chunks = {chunk.chunk_id: chunk for chunk in checkpoint.chunks}
        results: list[ConceptExtractionChunk] = []
        for chunk_number, work in enumerate(work_items, start=1):
            previous = previous_chunks.get(work.chunk_id)
            if previous is not None and previous.status == "succeeded":
                record_trace_event(
                    "Reuse concept extraction checkpoint",
                    {
                        "stage": "concept_extraction",
                        "message": "Reuse an accepted PDF chunk extraction.",
                        "chunk_id": work.chunk_id,
                        "page_numbers": work.page_numbers,
                        "attempt": previous.run_attempts,
                        "status": "reused",
                        **pdf_source_metadata([source]),
                    },
                )
                logger.info(
                    "Concept chunk %d/%d for pages %s reused its successful checkpoint",
                    chunk_number,
                    len(work_items),
                    page_range(work.page_numbers),
                )
                results.append(previous)
                continue

            self._log_chunk_start(chunk_number, len(work_items), work, previous)
            result = self._run_chunk(source, work, previous)
            results.append(result)
            self._checkpoint_store.save_incremental(
                source,
                results,
                previous_chunks,
            )

        # Remove checkpoint entries for chunks that are no longer in the document.
        self._checkpoint_store.save(source, results)
        return results

    def _run_concurrently(
        self,
        source: SourceRef,
        work_items: Sequence[ConceptChunkWork],
        executor: Executor,
    ) -> list[ConceptExtractionChunk]:
        """Run pending chunks concurrently and write checkpoints here."""
        checkpoint = self._checkpoint_store.load(source)
        previous_chunks = {chunk.chunk_id: chunk for chunk in checkpoint.chunks}
        results_by_id = {
            work.chunk_id: previous
            for work in work_items
            if (previous := previous_chunks.get(work.chunk_id)) is not None
            and previous.status == "succeeded"
        }
        futures: dict[Future[ConceptExtractionChunk], ConceptChunkWork] = {}

        for chunk_number, work in enumerate(work_items, start=1):
            previous = previous_chunks.get(work.chunk_id)
            if work.chunk_id in results_by_id:
                reused = results_by_id[work.chunk_id]
                record_trace_event(
                    "Reuse concept extraction checkpoint",
                    {
                        "stage": "concept_extraction",
                        "message": "Reuse an accepted PDF chunk extraction.",
                        "chunk_id": work.chunk_id,
                        "page_numbers": work.page_numbers,
                        "attempt": reused.run_attempts,
                        "status": "reused",
                        **pdf_source_metadata([source]),
                    },
                )
                logger.info(
                    "Concept chunk %d/%d for pages %s reused its successful checkpoint",
                    chunk_number,
                    len(work_items),
                    page_range(work.page_numbers),
                )
                continue

            self._log_chunk_start(chunk_number, len(work_items), work, previous)
            future = executor.submit(self._run_chunk, source, work, previous)
            futures[future] = work

        first_error: Exception | None = None
        for future in as_completed(futures):
            work = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                if first_error is None:
                    first_error = exc
                logger.exception(
                    "Concept extraction raised an unexpected error for pages %s",
                    page_range(work.page_numbers),
                )
                continue

            results_by_id[result.chunk_id] = result
            current_results = self._ordered_results(work_items, results_by_id)
            self._checkpoint_store.save_incremental(
                source,
                current_results,
                previous_chunks,
            )

        if first_error is not None:
            raise first_error

        results = self._ordered_results(work_items, results_by_id)
        # Remove checkpoint entries for chunks that are no longer in the document.
        self._checkpoint_store.save(source, results)
        return results

    @staticmethod
    def _ordered_results(
        work_items: Sequence[ConceptChunkWork],
        results_by_id: dict[str, ConceptExtractionChunk],
    ) -> list[ConceptExtractionChunk]:
        """Return available results in source-page order."""
        return [
            results_by_id[work.chunk_id]
            for work in work_items
            if work.chunk_id in results_by_id
        ]

    @staticmethod
    def _log_chunk_start(
        chunk_number: int,
        chunk_count: int,
        work: ConceptChunkWork,
        previous: ConceptExtractionChunk | None,
    ) -> None:
        """Log whether a chunk starts or resumes extraction."""
        if previous is not None and previous.retry_state is not None:
            logger.info(
                "Concept chunk %d/%d for pages %s is resuming attempt %d "
                "from its latest candidate",
                chunk_number,
                chunk_count,
                page_range(work.page_numbers),
                previous.run_attempts + 1,
            )
            return

        logger.info(
            "Concept chunk %d/%d for pages %s is starting extraction attempt %d",
            chunk_number,
            chunk_count,
            page_range(work.page_numbers),
            1 if previous is None else previous.run_attempts + 1,
        )

    def _run_chunk(
        self,
        source: SourceRef,
        work: ConceptChunkWork,
        previous: ConceptExtractionChunk | None,
    ) -> ConceptExtractionChunk:
        """Execute one chunk and convert expected exhaustion into data."""
        run_attempts = 1 if previous is None else previous.run_attempts + 1
        retry_state = previous.retry_state if previous is not None else None

        with traced_workflow(
            "Extract and review PDF chunk",
            metadata={
                "stage": "concept_extraction",
                "message": "Extract, critique, and repair concepts from one PDF chunk.",
                "chunk_id": work.chunk_id,
                "page_numbers": work.page_numbers,
                "attempt": run_attempts,
                "resumed": retry_state is not None,
                **pdf_source_metadata([source]),
            },
        ) as trace_metadata:
            try:
                if retry_state is None:
                    extraction = self._extractor(work.anchored_text)
                else:
                    extraction = self._extractor(
                        work.anchored_text,
                        retry_state=retry_state,
                    )
            except ConceptExtractionExhausted as exc:
                trace_metadata.update(
                    status="failed",
                    error_code=exc.error_code,
                )
                logger.warning(
                    "Concept extraction failed for pages %s on attempt %d: %s",
                    page_range(work.page_numbers),
                    run_attempts,
                    exc,
                )
                return ConceptExtractionChunk(
                    chunk_id=work.chunk_id,
                    page_numbers=work.page_numbers,
                    status="failed",
                    run_attempts=run_attempts,
                    error_code=exc.error_code,
                    error_message=str(exc),
                    retry_state=exc.retry_state,
                )

            trace_metadata.update(
                status="succeeded",
                concept_count=len(extraction.concepts),
            )
            result = ConceptExtractionChunk(
                chunk_id=work.chunk_id,
                page_numbers=work.page_numbers,
                status="succeeded",
                run_attempts=run_attempts,
                extraction=extraction,
            )
            logger.info(
                "Concept extraction succeeded for pages %s on attempt %d with %d concepts",
                page_range(work.page_numbers),
                run_attempts,
                len(extraction.concepts),
            )
            return result
