"""Publish extracted papers to the managed wiki repository."""

import json
import logging
import os
import shutil
from collections.abc import Mapping, Sequence
from concurrent.futures import Executor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path

from wiki_agents.concept.models import ExtractedSourceConcepts
from wiki_agents.concept.pipeline import ConceptExtractionPipeline
from wiki_agents.concept.utils import normalize_filename_stem
from wiki_agents.orchestrator.artifacts import (
    PipelineSnapshotMetadata,
    build_extraction_snapshot,
    load_extraction_snapshot,
    read_json_object,
    render_reference,
)
from wiki_agents.repository import WikiRepository

type _IngestionBackup = tuple[bytes | None, Path | None]

logger = logging.getLogger("wiki_agents.orchestrator")


@dataclass(frozen=True, slots=True)
class IngestionPaths:
    """Store the repository paths for one paper ingestion."""

    paper_id: str
    normalized_stem: str
    artifact: str
    reference: str
    snapshot: str

    @property
    def tracked(self) -> list[str]:
        """Paths that one ingestion commit can change."""
        return [self.reference, self.snapshot]

    @property
    def published(self) -> list[str]:
        """Paths that one ingestion can publish locally."""
        return [self.artifact, *self.tracked]


class PaperIngestionService:
    """Extract and transactionally publish papers."""

    def __init__(
        self,
        *,
        repository: WikiRepository,
        pipeline: ConceptExtractionPipeline,
        staging_path: Path,
    ) -> None:
        """Create an ingestion service.

        Args:
            repository: Managed wiki repository.
            pipeline: Concept extraction pipeline.
            staging_path: Directory for prepared publication files.

        """
        self.repository = repository
        self.repository_path = repository.path
        self.pipeline = pipeline
        self.staging_path = staging_path

    def ingest(self, source_path: Path) -> ExtractedSourceConcepts | None:
        """Extract and publish one complete paper."""
        extracted = self.extract(source_path)
        if extracted is not None:
            self.publish(source_path, extracted)
        return extracted

    def extract(
        self,
        source_path: Path,
        *,
        executor: Executor | None = None,
    ) -> ExtractedSourceConcepts | None:
        """Extract one PDF or load its current authoritative snapshot."""
        paths = self.paths_for(source_path)
        source_sha = sha256(source_path.read_bytes()).hexdigest()
        snapshot_path = self.repository_path / paths.snapshot
        existing = read_json_object(snapshot_path)

        if self._is_current(source_path, paths, source_sha, existing):
            return load_extraction_snapshot(snapshot_path)

        if executor is None:
            extracted = self.pipeline.extract_concepts(source_path)
        else:
            extracted = self.pipeline.extract_concepts(
                source_path,
                executor=executor,
            )
        if extracted.concept_extraction_status != "succeeded":
            logger.warning(
                "Paper %s remains incomplete; no files were published",
                paths.paper_id,
            )
            return None
        return extracted

    def requires_extraction(self, source_path: Path) -> bool:
        """Return whether a source needs concept extraction."""
        paths = self.paths_for(source_path)
        source_sha = sha256(source_path.read_bytes()).hexdigest()
        try:
            existing = read_json_object(self.repository_path / paths.snapshot)
        except (OSError, TypeError, ValueError):
            return True
        return not self._is_current(source_path, paths, source_sha, existing)

    def publish(
        self,
        source_path: Path,
        extracted: ExtractedSourceConcepts,
    ) -> None:
        """Publish one successful extraction to its authoritative snapshot."""
        paths = self.paths_for(source_path)
        source_sha = sha256(source_path.read_bytes()).hexdigest()
        if extracted.source.sha256 != source_sha:
            raise ValueError(f"Source changed during concept extraction: {source_path}")

        snapshot_path = self.repository_path / paths.snapshot
        existing = read_json_object(snapshot_path)
        if self._is_current(source_path, paths, source_sha, existing):
            return

        timestamp = self._timestamp(existing)
        original_filename = self._original_filename(source_path, existing)
        snapshot = build_extraction_snapshot(
            extracted,
            pipeline=PipelineSnapshotMetadata(
                configuration=self.pipeline.metadata,
                fingerprint=self.pipeline.fingerprint,
            ),
            original_filename=original_filename,
            artifact_resource=(
                f"/references/papers/artifacts/{paths.normalized_stem}.pdf"
            ),
            extracted_at=timestamp,
        )
        reference = render_reference(
            title=paths.normalized_stem,
            original_filename=original_filename,
            source_sha256=source_sha,
            generated_at=timestamp,
            anchored_pages=extracted.pages,
        )
        contents = {
            paths.reference: reference,
            paths.snapshot: json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
        }
        staged_root, prepared = self._stage(source_path, paths, contents)
        self._commit(
            paths,
            staged_root,
            prepared,
            is_update=existing is not None,
        )

    @staticmethod
    def paths_for(source_path: Path) -> IngestionPaths:
        """Build repository paths for a PDF."""
        normalized_stem = normalize_filename_stem(source_path.stem)
        paper_id = f"paper_{normalized_stem}"
        return IngestionPaths(
            paper_id=paper_id,
            normalized_stem=normalized_stem,
            artifact=f"wiki/references/papers/artifacts/{normalized_stem}.pdf",
            reference=f"wiki/references/papers/{paper_id}.md",
            snapshot=f"extractions/{paper_id}.json",
        )

    def _is_current(
        self,
        source_path: Path,
        paths: IngestionPaths,
        source_sha: str,
        existing: Mapping[str, object] | None,
    ) -> bool:
        """Return whether all current outputs exist for the source PDF."""
        artifact = self.repository_path / paths.artifact
        artifact_is_current = source_path.absolute() == artifact.absolute() or (
            artifact.is_symlink()
            and artifact.is_file()
            and artifact.resolve() == source_path.resolve()
        )
        return (
            existing is not None
            and existing.get("source_sha256") == source_sha
            and existing.get("pipeline_fingerprint") == self.pipeline.fingerprint
            and artifact_is_current
            and (self.repository_path / paths.reference).is_file()
        )

    @staticmethod
    def _timestamp(existing: Mapping[str, object] | None) -> str:
        """Create a UTC timestamp that differs from the previous timestamp."""
        timestamp = datetime.now(UTC).replace(microsecond=0)
        rendered = timestamp.isoformat().replace("+00:00", "Z")
        if existing is not None and existing.get("extracted_at") == rendered:
            rendered = (
                (timestamp + timedelta(seconds=1)).isoformat().replace("+00:00", "Z")
            )
        return rendered

    @staticmethod
    def _original_filename(
        source_path: Path,
        existing: Mapping[str, object] | None,
    ) -> str:
        """Keep the first filename when a canonical artifact is reprocessed."""
        if existing is not None and source_path.parent.name == "artifacts":
            return str(existing.get("original_filename", source_path.name))
        return source_path.name

    def _stage(
        self,
        source_path: Path,
        paths: IngestionPaths,
        contents: Mapping[str, str],
    ) -> tuple[Path, dict[str, Path]]:
        """Write all new ingestion outputs to the staging directory."""
        staged_root = self.staging_path / paths.paper_id
        if staged_root.exists():
            shutil.rmtree(staged_root)
        staged_root.mkdir(parents=True)
        prepared: dict[str, Path] = {}
        for relative, content in contents.items():
            staged = staged_root / relative
            staged.parent.mkdir(parents=True, exist_ok=True)
            staged.write_text(content)
            prepared[relative] = staged

        canonical_artifact = self.repository_path / paths.artifact
        if source_path != canonical_artifact:
            staged_artifact = staged_root / paths.artifact
            staged_artifact.parent.mkdir(parents=True, exist_ok=True)
            link_target = os.path.relpath(
                source_path.absolute(),
                start=canonical_artifact.parent,
            )
            staged_artifact.symlink_to(link_target)
            prepared[paths.artifact] = staged_artifact
        return staged_root, prepared

    def _commit(
        self,
        paths: IngestionPaths,
        staged_root: Path,
        prepared: Mapping[str, Path],
        *,
        is_update: bool,
    ) -> None:
        """Publish staged outputs in one commit and restore them on failure."""
        tracked = paths.tracked
        backups: dict[str, _IngestionBackup] = {}
        try:
            self._replace_files(paths.published, prepared, backups)
            verb = "update" if is_update else "add"
            self.repository.stage_and_commit(
                tracked,
                f"ingest: {verb} {paths.paper_id}",
            )
        except BaseException:
            self.repository.unstage(tracked)
            self._restore_backups(backups)
            raise
        finally:
            if staged_root.exists():
                shutil.rmtree(staged_root)

    def _replace_files(
        self,
        paths: Sequence[str],
        prepared: Mapping[str, Path],
        backups: dict[str, _IngestionBackup],
    ) -> None:
        """Replace repository files and retain their previous contents."""
        for relative in paths:
            target = self.repository_path / relative
            if target.is_symlink():
                backups[relative] = (None, target.readlink())
            elif target.exists():
                backups[relative] = (target.read_bytes(), None)
            else:
                backups[relative] = (None, None)
            staged = prepared.get(relative)
            if staged is not None:
                target.parent.mkdir(parents=True, exist_ok=True)
                staged.replace(target)

    def _restore_backups(
        self,
        backups: Mapping[str, _IngestionBackup],
    ) -> None:
        """Restore repository files after an ingestion commit fails."""
        for relative, (previous, link_target) in backups.items():
            target = self.repository_path / relative
            if target.exists() or target.is_symlink():
                target.unlink()
            if link_target is not None:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(link_target)
            elif previous is not None:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(previous)
