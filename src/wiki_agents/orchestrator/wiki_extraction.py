"""Coordinate corpus extraction for Wiki orchestration."""

import logging
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import cast

from docker import DockerClient
from docker import from_env as docker_from_env
from docker.errors import DockerException

from wiki_agents.concept.models import ConceptContext, ExtractedSourceConcepts
from wiki_agents.concept.pipeline import ConceptExtractionPipeline
from wiki_agents.concept.store import ConceptStore
from wiki_agents.concept.utils import normalize_filename_stem
from wiki_agents.document import pdf_to_okf_source
from wiki_agents.orchestrator.artifacts import (
    load_extraction_snapshot,
    read_json_object,
)
from wiki_agents.orchestrator.concept_store import ConceptStoreProvider
from wiki_agents.orchestrator.config import WikiOrchestratorConfig
from wiki_agents.orchestrator.ingestion import PaperIngestionService
from wiki_agents.progress import ProgressReporter
from wiki_agents.repository import WikiRepository
from wiki_agents.tracing import init_weave_tracing

logger = logging.getLogger("wiki_agents.orchestrator")


class WikiExtractionMixin:
    """Provide extraction coordination to the Wiki orchestrator."""

    config: WikiOrchestratorConfig
    repository_path: Path
    sources_path: Path
    ingestion: PaperIngestionService
    concept_pipeline: ConceptExtractionPipeline
    concept_store_provider: ConceptStoreProvider
    wiki_repository: WikiRepository
    progress: ProgressReporter

    @staticmethod
    def _parse_document(path: str | Path) -> list[str]:
        """Read a PDF without mutating the Git-tracked repository."""
        path = Path(path)
        if not path.is_file():
            raise ValueError(f"Path {path} is not a file")

        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Path {path} is not a PDF file")

        _, anchored_pages = pdf_to_okf_source(path)
        return anchored_pages

    def extract_concepts(self) -> list[ExtractedSourceConcepts]:
        """Extract the pending corpus concurrently and synchronize vector state.

        Returns:
            Authoritative extracted sources in snapshot path order.

        """
        init_weave_tracing()
        logger.info("Stage 1/10: extracting and synchronizing source concepts")
        self.wiki_repository.assert_clean()
        pending = self._pending_pdfs()
        self._validate_pending_names(pending)
        if any(self.ingestion.requires_extraction(path) for path in pending):
            self._assert_docker_running()

        self._extract_pending_concurrently(pending)

        return self.load_extracted_sources()

    def load_extracted_sources(self) -> list[ExtractedSourceConcepts]:
        """Load authoritative extraction snapshots and synchronize vector state.

        Returns:
            Extracted sources in snapshot path order.

        """
        # Snapshots are authoritative. Load and synchronize the complete corpus.
        extracted_sources: list[ExtractedSourceConcepts] = []
        snapshot_paths = sorted(
            (self.repository_path / "extractions").glob("*.json"),
        )
        with self.progress.task(
            "loading and synchronizing extraction snapshots",
            len(snapshot_paths),
        ) as progress:
            for snapshot_path in snapshot_paths:
                try:
                    extracted = load_extraction_snapshot(snapshot_path)
                except Exception:
                    logger.exception(
                        "Could not load extraction snapshot %s",
                        snapshot_path,
                    )
                    progress.advance()
                    continue

                extracted_sources.append(extracted)
                try:
                    self.concept_store_provider.get().replace_source_candidates(
                        extracted.source,
                        extracted.candidates,
                    )
                except Exception:
                    logger.exception(
                        "Could not synchronize Qdrant from %s",
                        snapshot_path,
                    )
                progress.advance()
        return extracted_sources

    def prepare_wiki_input(
        self,
        extracted_sources: Sequence[ExtractedSourceConcepts],
    ) -> list[ConceptContext]:
        """Build one corpus-wide context for each extracted concept.

        Args:
            extracted_sources: Sources to read in their required processing order.

        Returns:
            Wiki inputs in source and candidate order.

        """
        candidates = [
            candidate
            for extracted in extracted_sources
            for candidate in extracted.candidates
        ]
        contexts: list[ConceptContext] = []
        with self.progress.task(
            "preparing corpus candidate contexts",
            len(candidates),
        ) as progress:
            for candidate in candidates:
                contexts.append(self.concept_pipeline.prepare_wiki_input(candidate))
                progress.advance()
        return contexts

    def _snapshot_pdf_paths(self) -> dict[str, Path]:
        """Resolve sandbox PDF paths from authoritative snapshot resources."""
        paths: dict[str, Path] = {}
        for snapshot_path in sorted(
            (self.repository_path / "extractions").glob("*.json"),
        ):
            try:
                snapshot = read_json_object(snapshot_path)
            except Exception:
                logger.exception(
                    "Could not resolve a PDF from snapshot %s",
                    snapshot_path,
                )
                continue
            if snapshot is None:
                continue
            source_id = str(snapshot["paper_id"])
            resource = str(snapshot["artifact_resource"])
            if not resource.startswith("/"):
                raise ValueError(
                    f"Snapshot artifact resource is not bundle-rooted: {resource}",
                )
            artifact = self.repository_path / "wiki" / resource.removeprefix("/")
            if not artifact.is_file() or artifact.suffix.lower() != ".pdf":
                raise ValueError(f"Snapshot artifact PDF is missing: {artifact}")
            paths[source_id] = artifact
        return paths

    def _extract_pending_concurrently(
        self,
        pending: Sequence[Path],
    ) -> dict[Path, ExtractedSourceConcepts | None]:
        """Extract chunks and publish each completed PDF serially."""
        if not pending:
            return {}

        coordinator_count = min(self.config.num_extraction_workers, len(pending))
        with (
            ThreadPoolExecutor(max_workers=coordinator_count) as coordinators,
            ThreadPoolExecutor(
                max_workers=self.config.num_extraction_workers,
            ) as chunk_executor,
        ):
            futures: dict[Future[ExtractedSourceConcepts | None], Path] = {
                coordinators.submit(
                    self.ingestion.extract,
                    path,
                    executor=chunk_executor,
                ): path
                for path in pending
            }
            completed: dict[Path, ExtractedSourceConcepts | None] = {}
            with self.progress.task(
                "extracting and publishing PDFs",
                len(futures),
            ) as progress:
                for future in as_completed(futures):
                    path = futures[future]
                    try:
                        extracted = future.result()
                    except Exception:
                        logger.exception(
                            "Concept extraction failed for %s; continuing",
                            path,
                        )
                        completed[path] = None
                        progress.advance()
                        continue

                    completed[path] = extracted
                    if extracted is not None:
                        try:
                            self.wiki_repository.assert_clean()
                            self.ingestion.publish(path, extracted)
                        except Exception:
                            logger.exception(
                                "Ingestion failed for %s; continuing", path,
                            )
                    progress.advance()
        return {path: completed[path] for path in pending}

    @staticmethod
    def _assert_docker_running() -> None:
        """Require access to a responsive Docker daemon.

        Raises:
            RuntimeError: Docker is unavailable or its socket is not accessible.

        """
        client: DockerClient | None = None
        try:
            client = cast("DockerClient", docker_from_env())
            if not client.ping():
                raise RuntimeError("Docker daemon did not respond to its health check")
        except (DockerException, OSError) as exc:
            raise RuntimeError(
                "Docker daemon is not available. Start Docker and make its socket "
                "accessible.",
            ) from exc
        finally:
            if client is not None:
                client.close()

    def _pending_pdfs(self) -> list[Path]:
        """Return drop-folder inputs plus canonical artifacts needing inspection."""
        queued = [
            path
            for path in self.sources_path.iterdir()
            if path.is_file() and path.suffix.lower() == ".pdf"
        ]
        artifacts = self.repository_path / "wiki/references/papers/artifacts"
        legacy = [
            path
            for path in artifacts.iterdir()
            if path.is_file()
            and not path.is_symlink()
            and path.suffix.lower() == ".pdf"
        ]
        return sorted({path.absolute() for path in [*queued, *legacy]}, key=str)

    @staticmethod
    def _validate_pending_names(paths: Sequence[Path]) -> None:
        """Fail before extraction when distinct filenames map to one paper key."""
        paths_by_id: dict[str, list[Path]] = {}
        for path in paths:
            paper_id = f"paper_{normalize_filename_stem(path.stem)}"
            paths_by_id.setdefault(paper_id, []).append(path)
        collisions = {
            paper_id: {path.name for path in grouped_paths}
            for paper_id, grouped_paths in paths_by_id.items()
            if len(grouped_paths) > 1
        }
        if collisions:
            detail = "; ".join(
                f"{paper_id}: {', '.join(sorted(names))}"
                for paper_id, names in sorted(collisions.items())
            )
            raise ValueError(f"PDF filename normalization collision: {detail}")

    def _get_concept_store(self) -> ConceptStore:
        """Return the configured concept store."""
        return self.concept_store_provider.get()
