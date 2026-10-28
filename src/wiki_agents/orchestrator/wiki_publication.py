"""Create and publish the planned Wiki corpus."""

import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from threading import Lock
from uuid import UUID

from wiki_agents.agents.wiki.aggregator import build_page_contexts
from wiki_agents.agents.wiki.graph_validation import validate_graph_manifest
from wiki_agents.agents.wiki.indexes import render_concept_index_from_files
from wiki_agents.agents.wiki.models import WikiPageManifest, WikiPageRetryState
from wiki_agents.agents.wiki.validation import (
    WikiValidationIssue,
    normalize_wiki_bundle_metadata,
    utc_timestamp,
    validate_wiki_bundle,
)
from wiki_agents.agents.wiki.workflow import (
    WikiConceptStatus,
    WikiFinalizationWorkflow,
    WikiPageOutcome,
    WikiPageWorkflow,
    run_page_jobs_concurrently,
)
from wiki_agents.agents.wiki.workspace import read_host_wiki
from wiki_agents.concept.models import ConceptCandidate, ExtractedSourceConcepts
from wiki_agents.orchestrator.config import WikiOrchestratorConfig
from wiki_agents.orchestrator.mapping_checkpoints import (
    WikiCorpusCheckpoint,
    WikiMappingCheckpointStore,
    WikiPageAttempt,
    wiki_stage_fingerprint,
)
from wiki_agents.orchestrator.wiki_checkpoint_policy import _reusable_page_outcomes
from wiki_agents.orchestrator.wiki_outcomes import WikiCorpusOutcome
from wiki_agents.progress import ProgressReporter
from wiki_agents.repository import WikiRepository
from wiki_agents.tracing import (
    init_weave_tracing,
    pdf_source_metadata,
    record_trace_event,
    traced_workflow,
)

logger = logging.getLogger("wiki_agents.orchestrator")


class WikiPublicationMixin:
    """Provide atomic Wiki publication to the Wiki orchestrator."""

    config: WikiOrchestratorConfig
    repository_path: Path
    page_generation_config_fingerprint: str
    page_workflow: WikiPageWorkflow
    finalization_workflow: WikiFinalizationWorkflow
    wiki_mapping_checkpoints: WikiMappingCheckpointStore
    wiki_repository: WikiRepository
    progress: ProgressReporter
    load_extracted_sources: Callable[[], list[ExtractedSourceConcepts]]
    _snapshot_pdf_paths: Callable[[], dict[str, Path]]

    def write_wiki(
        self,
        extracted_sources: Sequence[ExtractedSourceConcepts] | None = None,
    ) -> WikiCorpusOutcome:
        """Create, finalize, validate, and publish the planned Wiki pages."""
        init_weave_tracing()
        if extracted_sources is None:
            extracted_sources = self.load_extracted_sources()
        with traced_workflow(
            "Generate and publish wiki corpus",
            metadata={
                "stage": "wiki_publication",
                "message": "Generate, review, finalize, validate, and commit wiki pages.",
                **pdf_source_metadata(
                    [extracted.source for extracted in extracted_sources],
                ),
            },
        ) as trace_metadata:
            outcome = self._write_wiki(extracted_sources)
            trace_metadata.update(
                status="completed" if outcome.commit_sha is not None else "failed",
                page_count=len(outcome.pages),
                commit_sha=outcome.commit_sha,
            )
            return outcome

    def _write_wiki(
        self,
        extracted_sources: Sequence[ExtractedSourceConcepts],
    ) -> WikiCorpusOutcome:
        """Execute atomic publication inside the correlated publication trace."""
        candidates = [
            candidate
            for extracted in extracted_sources
            for candidate in extracted.candidates
        ]
        candidates_by_id = {
            candidate.candidate_id: candidate for candidate in candidates
        }
        checkpoint = self.wiki_mapping_checkpoints.load_corpus()
        if checkpoint.final_graph is None:
            raise RuntimeError("Wiki plans are missing; run the planning stage first")
        manifest = checkpoint.final_graph.page_manifest
        source_pdfs = self._snapshot_pdf_paths()
        checkpoint, page_outcomes = self._create_pages(
            checkpoint,
            manifest,
            candidates,
            source_pdfs,
            final_ranking_fingerprint=checkpoint.final_graph.ranking_fingerprint,
        )
        failed_count = sum(
            outcome.status is not WikiConceptStatus.ACCEPTED
            for outcome in page_outcomes
        )
        if failed_count:
            logger.error(
                "Atomic Wiki publication stopped: %d page jobs failed",
                failed_count,
            )
            return WikiCorpusOutcome(tuple(page_outcomes), None)

        final_wiki, checkpoint = self._finalize_wiki(
            checkpoint,
            manifest,
            page_outcomes,
            candidates_by_id,
        )
        final_wiki = self._normalize_and_validate_wiki(final_wiki, manifest)
        commit_sha = self._publish_wiki(final_wiki, manifest)
        checkpoint = checkpoint.model_copy(update={"final_commit_sha": commit_sha})
        self.wiki_mapping_checkpoints.save_corpus(checkpoint)
        logger.info("Workflow complete: committed %d pages", len(manifest.pages))
        return WikiCorpusOutcome(tuple(page_outcomes), commit_sha)

    def _create_pages(  # ruff: ignore[too-many-locals] - Publication tracks reuse, persistence, and progress state.
        self,
        checkpoint: WikiCorpusCheckpoint,
        manifest: WikiPageManifest,
        candidates: Sequence[ConceptCandidate],
        source_pdfs: dict[str, Path],
        *,
        final_ranking_fingerprint: str,
    ) -> tuple[WikiCorpusCheckpoint, list[WikiPageOutcome]]:
        """Create page proposals and save their resumable state."""
        logger.info("Stage 7/10: creating and reviewing %d pages", len(manifest.pages))
        manifest_issues = validate_graph_manifest(manifest)
        if manifest_issues:
            raise RuntimeError(
                "Wiki graph manifest failed before page generation: "
                + "; ".join(issue.problem for issue in manifest_issues),
            )
        generation_fingerprint = wiki_stage_fingerprint(
            {
                "final_ranking_fingerprint": final_ranking_fingerprint,
                "writing_config_fingerprint": self.page_generation_config_fingerprint,
            },
        )
        if checkpoint.page_generation_fingerprint != generation_fingerprint:
            checkpoint = checkpoint.model_copy(
                update={
                    "page_generation_fingerprint": generation_fingerprint,
                    "page_attempts": [],
                    "accepted_page_ids": [],
                    "finalizer_state": {},
                    "final_commit_sha": None,
                },
            )
            self.wiki_mapping_checkpoints.save_corpus(checkpoint)

        saved_attempts = self.wiki_mapping_checkpoints.load_page_attempts(
            generation_fingerprint,
        )
        checkpoint = checkpoint.model_copy(
            update={
                "page_attempts": saved_attempts,
                "accepted_page_ids": [
                    attempt.page_id for attempt in saved_attempts if attempt.accepted
                ],
            },
        )

        contexts = build_page_contexts(manifest, candidates)
        reusable = _reusable_page_outcomes(
            checkpoint,
            contexts,
            generation_fingerprint,
            lambda page_id: self.wiki_mapping_checkpoints.load_page_proposal(
                generation_fingerprint,
                page_id,
            ),
        )
        if reusable:
            reusable_contexts = [
                context for context in contexts if context.page_job.page_id in reusable
            ]
            record_trace_event(
                "Reuse accepted wiki page checkpoint",
                {
                    "stage": "page_generation",
                    "message": "Reuse compatible accepted wiki page proposals.",
                    "status": "reused",
                    "page_ids": [
                        context.page_job.page_id for context in reusable_contexts
                    ],
                    "candidate_ids": [
                        str(candidate.candidate_id)
                        for context in reusable_contexts
                        for candidate in context.candidates
                    ],
                    **pdf_source_metadata(
                        [
                            candidate.source
                            for context in reusable_contexts
                            for candidate in context.candidates
                        ],
                    ),
                },
            )
            logger.info(
                "Reusing %d compatible accepted page proposals",
                len(reusable),
            )
        pending = [
            context for context in contexts if context.page_job.page_id not in reusable
        ]
        retry_states = {
            attempt.page_id: attempt.retry_state
            for attempt in saved_attempts
            if not attempt.accepted and attempt.retry_state is not None
        }
        if retry_states:
            logger.info(
                "Resuming %d failed page proposals from their latest review",
                len(retry_states),
            )
        outcome_lock = Lock()
        accepted_count = len(reusable)
        failed_count = 0
        with self.progress.task(
            self._page_progress_description(accepted_count, failed_count),
            len(contexts),
            completed=len(reusable),
        ) as progress:

            def checkpoint_proposal(
                page_id: str,
                state: WikiPageRetryState,
            ) -> None:
                """Save the latest proposal without advancing terminal progress."""
                self.wiki_mapping_checkpoints.save_page_attempt(
                    generation_fingerprint,
                    WikiPageAttempt(
                        page_id=page_id,
                        attempts=state.correction_runs + 1,
                        accepted=False,
                        last_error=(
                            "; ".join(finding.problem for finding in state.findings)
                            or "Proposal awaits review."
                        ),
                        retry_state=state,
                    ),
                )

            def checkpoint_and_report(outcome: WikiPageOutcome) -> None:
                """Save and report one terminal page outcome."""
                nonlocal accepted_count, failed_count
                self._checkpoint_page_outcome(generation_fingerprint, outcome)
                with outcome_lock:
                    if outcome.status is WikiConceptStatus.ACCEPTED:
                        accepted_count += 1
                        logger.info(
                            "Wiki page %s accepted after %d attempts "
                            "(accepted %d, failed %d, completed %d/%d)",
                            outcome.page_id,
                            outcome.correction_runs + 1,
                            accepted_count,
                            failed_count,
                            accepted_count + failed_count,
                            len(contexts),
                        )
                    else:
                        failed_count += 1
                        logger.warning(
                            "Wiki page %s failed category=%s after %d attempts: %s "
                            "(accepted %d, failed %d, completed %d/%d)",
                            outcome.page_id,
                            (
                                outcome.failure_category.value
                                if outcome.failure_category is not None
                                else "unknown"
                            ),
                            outcome.correction_runs + 1,
                            outcome.error,
                            accepted_count,
                            failed_count,
                            accepted_count + failed_count,
                            len(contexts),
                        )
                    progress.set_description(
                        self._page_progress_description(
                            accepted_count,
                            failed_count,
                        ),
                    )
                    progress.advance(1)

            created = (
                run_page_jobs_concurrently(
                    self.page_workflow,
                    pending,
                    page_manifest=manifest,
                    wiki_path=self.repository_path / "wiki",
                    source_pdfs=source_pdfs,
                    workers=self.config.wiki_page_workers,
                    checkpoint=checkpoint_and_report,
                    proposal_checkpoint=checkpoint_proposal,
                    retry_states=retry_states,
                )
                if pending
                else []
            )
        created_by_id = {outcome.page_id: outcome for outcome in created}
        outcomes = [
            reusable.get(context.page_job.page_id)
            or created_by_id[context.page_job.page_id]
            for context in contexts
        ]
        accepted_page_ids = [
            outcome.page_id
            for outcome in outcomes
            if outcome.status is WikiConceptStatus.ACCEPTED
        ]
        checkpoint = checkpoint.model_copy(
            update={
                "page_generation_fingerprint": generation_fingerprint,
                "page_attempts": [
                    WikiPageAttempt(
                        page_id=outcome.page_id,
                        attempts=outcome.correction_runs + 1,
                        accepted=outcome.status is WikiConceptStatus.ACCEPTED,
                        last_error=outcome.error,
                        retry_state=outcome.retry_state,
                    )
                    for outcome in outcomes
                ],
                "accepted_page_ids": accepted_page_ids,
            },
        )
        self.wiki_mapping_checkpoints.save_corpus(checkpoint)
        return checkpoint, outcomes

    @staticmethod
    def _page_progress_description(accepted: int, failed: int) -> str:
        """Return page progress text with terminal outcome counts."""
        return (
            f"creating and reviewing Wiki pages [accepted {accepted}, failed {failed}]"
        )

    def _checkpoint_page_outcome(
        self,
        generation_fingerprint: str,
        outcome: WikiPageOutcome,
    ) -> None:
        """Save one page outcome before its worker reports completion."""
        accepted = outcome.status is WikiConceptStatus.ACCEPTED
        if accepted:
            if outcome.creation is None or outcome.creation.changes[0].content is None:
                raise RuntimeError("an accepted page has no Markdown content")
            self.wiki_mapping_checkpoints.save_page_proposal(
                generation_fingerprint,
                outcome.page_id,
                outcome.creation.changes[0].content,
            )
        self.wiki_mapping_checkpoints.save_page_attempt(
            generation_fingerprint,
            WikiPageAttempt(
                page_id=outcome.page_id,
                attempts=outcome.correction_runs + 1,
                accepted=accepted,
                last_error=outcome.error,
                retry_state=outcome.retry_state,
            ),
        )

    def _finalize_wiki(
        self,
        checkpoint: WikiCorpusCheckpoint,
        manifest: WikiPageManifest,
        page_outcomes: Sequence[WikiPageOutcome],
        candidates_by_id: dict[UUID, ConceptCandidate],
    ) -> tuple[dict[str, bytes], WikiCorpusCheckpoint]:
        """Finalize the complete proposed Wiki and save finalizer state."""
        logger.info("Stage 8/10: finalizing the complete proposed Wiki")
        proposed_wiki = read_host_wiki(self.repository_path / "wiki")
        for outcome in page_outcomes:
            if outcome.creation is None:
                raise RuntimeError("an accepted page has no creation result")
            change = outcome.creation.changes[0]
            if change.content is None:
                raise RuntimeError("an accepted page has no content")
            proposed_wiki[change.path.removeprefix("wiki/")] = change.content

        provenance_manifest = {
            page.page_id: {
                "member_candidate_ids": [
                    str(item) for item in page.member_candidate_ids
                ],
                "source_ids": sorted(
                    {
                        candidates_by_id[item].source.source_id
                        for item in page.member_candidate_ids
                    },
                ),
            }
            for page in manifest.pages
        }
        final_wiki = proposed_wiki
        findings = validate_wiki_bundle(proposed_wiki, manifest)
        sources = [candidate.source for candidate in candidates_by_id.values()]
        if not findings:
            checkpoint = checkpoint.model_copy(
                update={"finalizer_state": {"runs": 0, "issues": 0}},
            )
            self.wiki_mapping_checkpoints.save_corpus(checkpoint)
            logger.info("Complete proposed Wiki is valid; skipping corpus finalizer")
            return proposed_wiki, checkpoint
        finalizer_run = 0
        completed_package_count = 0
        for finalizer_run in range(self.config.max_wiki_turns + 1):
            packages = self._finalizer_packages(
                findings,
                manifest,
                batch_size=self.config.wiki_finalizer_batch_size,
            )
            logger.info(
                "Corpus finalizer run %d: %d findings in %d work packages",
                finalizer_run + 1,
                len(findings),
                len(packages),
            )
            with self.progress.task(
                f"finalizing Wiki corpus run {finalizer_run + 1}",
                len(packages),
            ) as progress:
                for package_number, (editable_paths, package_findings) in enumerate(
                    packages,
                    start=1,
                ):
                    progress.set_description(
                        "finalizing Wiki corpus "
                        f"run {finalizer_run + 1}, package "
                        f"{package_number}/{len(packages)}",
                    )
                    logger.info(
                        "Finalizer package %d/%d: %d pages, %d findings",
                        package_number,
                        len(packages),
                        len(editable_paths),
                        len(package_findings),
                    )
                    package_manifest = self._finalizer_package_manifest(
                        manifest,
                        editable_paths,
                    )
                    package_wiki = self._finalizer_package_wiki(
                        final_wiki,
                        editable_paths,
                    )
                    package_provenance = {
                        page.page_id: provenance_manifest[page.page_id]
                        for page in package_manifest.pages
                    }
                    finalized_package = self.finalization_workflow.run_once(
                        package_wiki,
                        package_manifest,
                        package_provenance,
                        package_findings,
                        editable_paths,
                        sources=sources,
                        iteration=finalizer_run,
                        package_number=package_number,
                        package_count=len(packages),
                    )
                    final_wiki.update(finalized_package)
                    completed_package_count += 1
                    progress.advance()
            findings = validate_wiki_bundle(final_wiki, manifest)
            if not findings:
                break
            if finalizer_run >= self.config.max_wiki_turns:
                raise RuntimeError("corpus finalizer exhausted its correction runs")

        checkpoint = checkpoint.model_copy(
            update={
                "finalizer_state": {
                    "runs": finalizer_run + 1,
                    "issues": 0,
                    "packages": completed_package_count,
                },
            },
        )
        self.wiki_mapping_checkpoints.save_corpus(checkpoint)
        return final_wiki, checkpoint

    @staticmethod
    def _finalizer_packages(
        findings: Sequence[WikiValidationIssue],
        manifest: WikiPageManifest,
        *,
        batch_size: int,
    ) -> list[tuple[list[str], tuple[WikiValidationIssue, ...]]]:
        """Group deterministic findings into bounded page work packages."""
        findings_by_path: dict[str, list[WikiValidationIssue]] = {}
        manifest_paths = [page.target_path for page in manifest.pages]
        for finding in findings:
            target_path = next(
                (
                    path
                    for path in manifest_paths
                    if finding.target == path or finding.target.startswith(f"{path} ")
                ),
                None,
            )
            if target_path is None:
                raise RuntimeError(
                    "Corpus finalizer cannot edit an unscoped finding: "
                    f"{finding.target}: {finding.problem}",
                )
            findings_by_path.setdefault(target_path, []).append(finding)

        affected_paths = [path for path in manifest_paths if path in findings_by_path]
        packages = []
        for offset in range(0, len(affected_paths), batch_size):
            paths = affected_paths[offset : offset + batch_size]
            package_findings = tuple(
                finding for path in paths for finding in findings_by_path[path]
            )
            packages.append((paths, package_findings))
        return packages

    @staticmethod
    def _finalizer_package_manifest(
        manifest: WikiPageManifest,
        editable_paths: Sequence[str],
    ) -> WikiPageManifest:
        """Return manifest entries for one finalizer work package."""
        editable = set(editable_paths)
        return WikiPageManifest(
            pages=[page for page in manifest.pages if page.target_path in editable],
        )

    @staticmethod
    def _finalizer_package_wiki(
        wiki_files: dict[str, bytes],
        editable_paths: Sequence[str],
    ) -> dict[str, bytes]:
        """Return editable pages and shared non-concept context files."""
        editable = {path.removeprefix("wiki/") for path in editable_paths}
        return {
            path: content
            for path, content in wiki_files.items()
            if not path.removeprefix("wiki/").startswith("concepts/")
            or path.removeprefix("wiki/") in editable
            or path.removeprefix("wiki/") == "concepts/index.md"
        }

    @staticmethod
    def _normalize_and_validate_wiki(
        final_wiki: dict[str, bytes],
        manifest: WikiPageManifest,
    ) -> dict[str, bytes]:
        """Normalize metadata, create indexes, and validate the final Wiki."""
        logger.info("Stage 9/10: generating deterministic indexes and validating")
        final_timestamp = utc_timestamp()
        normalized_wiki = normalize_wiki_bundle_metadata(
            final_wiki,
            manifest,
            generated_at=final_timestamp,
            verified_at=final_timestamp,
        )
        concept_files = {
            path.removeprefix("concepts/"): content
            for path, content in normalized_wiki.items()
            if path.startswith("concepts/") and path.endswith(".md")
        }
        normalized_wiki["concepts/index.md"] = render_concept_index_from_files(
            concept_files,
        )
        issues = validate_wiki_bundle(
            normalized_wiki,
            manifest,
            require_indexes=True,
        )
        if issues:
            raise RuntimeError(
                "final Wiki failed deterministic bundle validation: "
                + "; ".join(issue.problem for issue in issues),
            )
        record_trace_event(
            "Validate complete wiki bundle",
            {
                "stage": "bundle_validation",
                "message": "Validate the normalized wiki and generated indexes.",
                "status": "accepted",
                "page_ids": [page.page_id for page in manifest.pages],
            },
        )
        return normalized_wiki

    def _publish_wiki(
        self,
        final_wiki: dict[str, bytes],
        manifest: WikiPageManifest,
    ) -> str:
        """Commit the complete Wiki graph atomically."""
        logger.info("Stage 10/10: committing the complete Wiki graph atomically")
        page_paths = {page.target_path.removeprefix("wiki/") for page in manifest.pages}
        files_to_commit = {
            f"wiki/{path}": content
            for path, content in final_wiki.items()
            if path == "concepts/index.md" or path in page_paths
        }
        with self.wiki_repository.begin_batch() as batch:
            batch.apply(files_to_commit)
            commit_sha = batch.commit("wiki: build concept graph")
        record_trace_event(
            "Publish complete wiki graph",
            {
                "stage": "wiki_commit",
                "message": "Commit the accepted wiki graph atomically.",
                "status": "committed",
                "commit_sha": commit_sha,
                "page_ids": [page.page_id for page in manifest.pages],
            },
        )
        return commit_sha
