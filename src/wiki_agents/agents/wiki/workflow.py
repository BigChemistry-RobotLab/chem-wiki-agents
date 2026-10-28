"""Run the deterministic Wiki creator and reviewer correction loop."""

from __future__ import annotations

import json
import logging
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

from wiki_agents.agents.wiki.creator import create_wiki_agent, run_wiki_agent
from wiki_agents.agents.wiki.finalizer import (
    create_wiki_finalizer_agent,
    run_wiki_finalizer_agent,
)
from wiki_agents.agents.wiki.models import (
    WikiChangeStatus,
    WikiCreationResult,
    WikiFileChange,
    WikiPageRetryState,
    WikiPlanDecision,
    WikiReview,
    WikiReviewDecision,
    WikiReviewFinding,
    WikiReviewFindingType,
)
from wiki_agents.agents.wiki.partition_boundary_resolver import (
    create_wiki_partition_boundary_resolver_agent,
)
from wiki_agents.agents.wiki.planner import (
    create_wiki_candidate_planner_agent,
    plan_candidates_concurrently,
)
from wiki_agents.agents.wiki.relation_resolver import (
    create_wiki_relation_resolver_agent,
)
from wiki_agents.agents.wiki.resilience import (
    WikiEditorOutputError,
    WikiInfrastructureUnavailableError,
    WikiPageFailureCategory,
    classify_wiki_transient_error,
    run_with_infrastructure_retries,
)
from wiki_agents.agents.wiki.resolver import resolve_relations_concurrently
from wiki_agents.agents.wiki.reviewer import (
    create_wiki_reviewer_agent,
    run_wiki_reviewer_agent,
)
from wiki_agents.agents.wiki.validation import (
    WikiValidationIssue,
    add_machine_verification,
    normalize_managed_metadata,
    utc_timestamp,
    validate_wiki_page_proposal,
)
from wiki_agents.agents.wiki.workspace import read_host_wiki
from wiki_agents.tracing import (
    pdf_source_metadata,
    record_trace_event,
    traced_workflow,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path
    from uuid import UUID

    from agents import Model

    from wiki_agents.agents.wiki.models import (
        WikiCandidatePlan,
        WikiPageContext,
        WikiPageManifest,
        WikiRelationResolution,
    )
    from wiki_agents.agents.wiki.resolver import WikiRelationCandidate
    from wiki_agents.concept.models import ConceptCandidate, ConceptContext
    from wiki_agents.document.models import SourceRef

logger = logging.getLogger("wiki_agents.orchestrator")
_INFRASTRUCTURE_FAILURE_LIMIT = 4


class WikiFinalizationWorkflow:
    """Own one serial correction step for a proposed corpus work package."""

    def __init__(
        self,
        *,
        model: str | Model | None,
        timeout_seconds: float,
    ) -> None:
        """Create the reusable corpus finalizer agent."""
        self.agent = create_wiki_finalizer_agent(model=model)
        self.timeout_seconds = timeout_seconds

    def run_once(  # ruff: ignore[too-many-arguments] - Trace provenance is explicit.
        self,
        wiki_files: Mapping[str, bytes],
        manifest: WikiPageManifest,
        provenance_manifest: Mapping[str, object],
        findings: Sequence[WikiValidationIssue],
        editable_paths: Sequence[str],
        *,
        sources: Sequence[SourceRef] = (),
        iteration: int = 0,
        package_number: int = 1,
        package_count: int = 1,
    ) -> dict[str, bytes]:
        """Run one serial finalizer work package."""
        with traced_workflow(
            "Finalize wiki corpus work package",
            metadata={
                "stage": "corpus_finalization",
                "message": "Validate and repair a proposed wiki work package.",
                "iteration": iteration,
                "page_ids": [page.page_id for page in manifest.pages],
                "finding_count": len(findings),
                "package_number": package_number,
                "package_count": package_count,
                **pdf_source_metadata(list(sources)),
            },
        ) as trace_metadata:
            result = run_wiki_finalizer_agent(
                self.agent,
                wiki_files,
                manifest,
                provenance_manifest,
                findings,
                editable_paths=editable_paths,
                timeout_seconds=self.timeout_seconds,
            )
            trace_metadata["status"] = "completed"
            return result


class WikiCandidatePlanningWorkflow:
    """Plan all candidates with one reusable stateless agent definition."""

    def __init__(self, *, model: str | Model | None) -> None:
        """Create the reusable candidate planner."""
        self.agent = create_wiki_candidate_planner_agent(model=model)

    def run(
        self,
        contexts: Sequence[ConceptContext],
        *,
        batch_size: int,
        workers: int,
        completed_plans: Sequence[WikiCandidatePlan] = (),
        checkpoint: Callable[[list[WikiCandidatePlan]], None] | None = None,
    ) -> list[WikiCandidatePlan]:
        """Plan pending candidate batches and checkpoint completed batches."""
        return plan_candidates_concurrently(
            self.agent,
            contexts,
            batch_size=batch_size,
            workers=workers,
            completed_plans=completed_plans,
            checkpoint=checkpoint,
        )


class WikiRelationResolutionWorkflow:
    """Resolve a sparse pair graph with one reusable typed agent."""

    def __init__(self, *, model: str | Model | None) -> None:
        """Create the reusable relation resolver."""
        self.agent = create_wiki_relation_resolver_agent(model=model)

    def run(  # ruff: ignore[too-many-arguments] - Resolution needs fixed corpus inputs, resume state, and bounded execution settings.
        self,
        pairs: Sequence[WikiRelationCandidate],
        plans: Mapping[UUID, WikiCandidatePlan],
        candidates: Mapping[UUID, ConceptCandidate],
        *,
        batch_size: int,
        workers: int,
        completed_relations: Sequence[WikiRelationResolution] = (),
        checkpoint: Callable[[list[WikiRelationResolution]], None] | None = None,
    ) -> list[WikiRelationResolution]:
        """Resolve pending relation batches and checkpoint valid responses."""
        return resolve_relations_concurrently(
            self.agent,
            pairs,
            plans,
            candidates,
            batch_size=batch_size,
            workers=workers,
            completed_relations=completed_relations,
            checkpoint=checkpoint,
        )


class WikiPartitionBoundaryWorkflow:
    """Resolve candidate relations that cross a refined page boundary."""

    def __init__(self, *, model: str | Model | None) -> None:
        """Create the reusable boundary relation resolver."""
        self.agent = create_wiki_partition_boundary_resolver_agent(model=model)

    def run(  # ruff: ignore[too-many-arguments] - Boundary resolution uses the shared relation runner.
        self,
        pairs: Sequence[WikiRelationCandidate],
        plans: Mapping[UUID, WikiCandidatePlan],
        candidates: Mapping[UUID, ConceptCandidate],
        *,
        batch_size: int,
        workers: int,
        completed_relations: Sequence[WikiRelationResolution] = (),
        checkpoint: Callable[[list[WikiRelationResolution]], None] | None = None,
    ) -> list[WikiRelationResolution]:
        """Resolve cut MERGE pairs as LINK or NONE decisions."""
        resolved = resolve_relations_concurrently(
            self.agent,
            pairs,
            plans,
            candidates,
            batch_size=batch_size,
            workers=workers,
            completed_relations=completed_relations,
            checkpoint=checkpoint,
        )
        if any(item.decision.value == "MERGE" for item in resolved):
            raise RuntimeError("a page-partition boundary cannot retain MERGE")
        return resolved


class WikiConceptStatus(StrEnum):
    """Describe the final outcome for one concept candidate."""

    ACCEPTED = "accepted"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class WikiPageOutcome:
    """Keep one independent page workflow result without publishing it."""

    page_id: str
    status: WikiConceptStatus
    creation: WikiCreationResult | None = None
    correction_runs: int = 0
    error: str | None = None
    retry_state: WikiPageRetryState | None = None
    failure_category: WikiPageFailureCategory | None = field(
        default=None,
        repr=False,
        compare=False,
    )


class WikiPageWorkflow:
    """Create, validate, review, and correct one fixed page job."""

    def __init__(
        self,
        *,
        creator_model: str | Model | None,
        reviewer_model: str | Model | None,
        max_correction_runs: int,
        transient_retries: int = 2,
    ) -> None:
        """Store the models used to create page-local agents."""
        if max_correction_runs < 0:
            raise ValueError("max_correction_runs must not be negative")
        if transient_retries < 0:
            raise ValueError("transient_retries must not be negative")
        self.creator_model = creator_model
        self.reviewer_model = reviewer_model
        self.max_correction_runs = max_correction_runs
        self.transient_retries = transient_retries

    def run(  # ruff: ignore[too-many-arguments] - Proposal persistence is a separate page lifecycle dependency.
        self,
        page_context: WikiPageContext,
        *,
        page_manifest: WikiPageManifest,
        wiki_path: Path,
        source_pdfs: Mapping[str, Path],
        retry_state: WikiPageRetryState | None = None,
        proposal_checkpoint: Callable[[str, WikiPageRetryState], None] | None = None,
    ) -> WikiPageOutcome:
        """Run one page lifecycle against the immutable baseline and manifest."""
        plan = page_context.page_job.plan
        with traced_workflow(
            "Create and review wiki page",
            metadata={
                "stage": "page_generation",
                "message": "Create, validate, review, and repair one wiki page.",
                "page_id": page_context.page_job.page_id,
                "target_path": plan.target_path or "",
                "candidate_ids": [
                    str(candidate.candidate_id) for candidate in page_context.candidates
                ],
                **pdf_source_metadata(
                    [candidate.source for candidate in page_context.candidates],
                ),
            },
        ) as trace_metadata:
            outcome = self._run_page(
                page_context,
                page_manifest=page_manifest,
                wiki_path=wiki_path,
                source_pdfs=source_pdfs,
                retry_state=retry_state,
                proposal_checkpoint=proposal_checkpoint,
            )
            trace_metadata.update(
                status=outcome.status.value,
                correction_runs=outcome.correction_runs,
                error=outcome.error,
            )
            return outcome

    def _run_page(  # ruff: ignore[too-many-arguments,complex-structure] - One page trace owns validation, review, correction, and recovery.
        self,
        page_context: WikiPageContext,
        *,
        page_manifest: WikiPageManifest,
        wiki_path: Path,
        source_pdfs: Mapping[str, Path],
        retry_state: WikiPageRetryState | None,
        proposal_checkpoint: Callable[[str, WikiPageRetryState], None] | None = None,
    ) -> WikiPageOutcome:
        """Execute the page correction loop inside its logical trace."""
        plan = page_context.page_job.plan
        page_id = page_context.page_job.page_id
        original_files = read_host_wiki(wiki_path)
        proposal: WikiCreationResult | None = None
        latest_state: WikiPageRetryState | None = retry_state
        correction_runs = retry_state.correction_runs if retry_state is not None else 0
        current_run_corrections = 0

        def save_state(state: WikiPageRetryState) -> None:
            """Save one resumable proposal state when configured."""
            nonlocal latest_state
            latest_state = state
            if proposal_checkpoint is not None:
                proposal_checkpoint(page_id, state)

        try:  # ruff: ignore[too-many-statements-in-try-clause] - A page worker owns its complete lifecycle.
            if retry_state is None:
                proposal = self._run_creator(
                    page_context,
                    page_manifest=page_manifest,
                    wiki_path=wiki_path,
                    source_pdfs=source_pdfs,
                    correction_count=correction_runs,
                )
                save_state(self._proposal_state(proposal, (), correction_runs))
            else:
                proposal = self._proposal_from_retry_state(page_context, retry_state)
                save_state(self._proposal_state(proposal, (), correction_runs))
            while True:
                try:
                    proposal = normalize_managed_metadata(
                        proposal,
                        generated_at=utc_timestamp(),
                    )
                    issues = validate_wiki_page_proposal(
                        proposal,
                        original_files=original_files,
                        page_context=page_context,
                        page_manifest=page_manifest,
                    )
                except ValueError as error:
                    issues = (
                        WikiValidationIssue(
                            WikiReviewFindingType.OKF,
                            plan.target_path or "wiki",
                            str(error),
                            "Write valid UTF-8 Markdown with OKF v0.2 frontmatter.",
                            "Deterministic OKF parser",
                        ),
                    )
                if issues:
                    findings = [issue.as_payload() for issue in issues]
                    record_trace_event(
                        "Deterministic page validation",
                        {
                            "stage": "page_validation",
                            "decision": "revise",
                            "finding_count": len(findings),
                            "findings": json.dumps(findings, sort_keys=True),
                        },
                    )
                else:
                    review = self._run_reviewer(
                        page_context,
                        proposal,
                        wiki_path=wiki_path,
                        source_pdfs=source_pdfs,
                        page_manifest=page_manifest,
                        correction_count=correction_runs,
                    )
                    record_trace_event(
                        "Wiki reviewer outcome",
                        {
                            "stage": "page_review",
                            "decision": review.decision.value,
                            "finding_count": len(review.findings),
                            "findings": json.dumps(
                                [
                                    finding.model_dump(mode="json")
                                    for finding in review.findings
                                ],
                                sort_keys=True,
                            ),
                        },
                    )
                    if review.decision is WikiReviewDecision.ACCEPT:
                        accepted = add_machine_verification(
                            proposal,
                            verified_at=utc_timestamp(),
                        )
                        final_issues = validate_wiki_page_proposal(
                            accepted,
                            original_files=original_files,
                            page_context=page_context,
                            page_manifest=page_manifest,
                        )
                        if not final_issues:
                            return WikiPageOutcome(
                                page_id,
                                WikiConceptStatus.ACCEPTED,
                                accepted,
                                correction_runs,
                            )
                        findings = [issue.as_payload() for issue in final_issues]
                    else:
                        findings = [
                            finding.model_dump(mode="json")
                            for finding in review.findings
                        ]
                save_state(self._proposal_state(proposal, findings, correction_runs))
                if current_run_corrections >= self.max_correction_runs:
                    return self._failed_review_outcome(
                        page_id,
                        proposal,
                        findings,
                        correction_runs,
                    )
                proposal = self._run_creator(
                    page_context,
                    wiki_path=wiki_path,
                    source_pdfs=source_pdfs,
                    previous_result=proposal,
                    correction_findings=findings,
                    page_manifest=page_manifest,
                    correction_count=correction_runs,
                )
                correction_runs += 1
                current_run_corrections += 1
                save_state(self._proposal_state(proposal, (), correction_runs))
        except Exception as error:  # ruff: ignore[blind-except] - A failed page must become a checkpointable result.
            classification = classify_wiki_transient_error(error)
            category = (
                WikiPageFailureCategory.INFRASTRUCTURE
                if classification.category != "other"
                else WikiPageFailureCategory.AGENT_OUTPUT
            )
            return WikiPageOutcome(
                page_id,
                WikiConceptStatus.FAILED,
                correction_runs=correction_runs,
                error=str(error),
                retry_state=latest_state,
                failure_category=category,
            )

    def _run_creator(  # ruff: ignore[too-many-arguments] - Page-local agents need the full page input.
        self,
        page_context: WikiPageContext,
        *,
        page_manifest: WikiPageManifest,
        wiki_path: Path,
        source_pdfs: Mapping[str, Path],
        correction_count: int,
        previous_result: WikiCreationResult | None = None,
        correction_findings: Sequence[Mapping[str, object]] | None = None,
    ) -> WikiCreationResult:
        """Run a creator in fresh sandboxes with bounded restart policies."""
        plan = page_context.page_job.plan

        def operation() -> WikiCreationResult:
            creator = create_wiki_agent(
                model=self.creator_model,
                transient_retries=self.transient_retries,
            )
            return run_wiki_agent(
                creator,
                page_context,
                plan,
                wiki_path=wiki_path,
                source_pdfs=source_pdfs,
                previous_result=previous_result,
                correction_findings=correction_findings,
                page_manifest=page_manifest,
            )

        for output_restart in range(2):
            try:
                return run_with_infrastructure_retries(
                    operation,
                    phase="creator",
                    retries=self.transient_retries,
                    correction_count=correction_count,
                )
            except WikiEditorOutputError as error:
                if output_restart or error.code not in {
                    "unchanged_target",
                    "wrong_action",
                    "wrong_file",
                }:
                    raise
                logger.warning(
                    "Wiki page retry phase=creator category=agent_output code=%s "
                    "status=None retry=1 delay=0 correction_count=%d",
                    error.code,
                    correction_count,
                )
        raise AssertionError("unreachable")

    def _run_reviewer(  # ruff: ignore[too-many-arguments] - Review needs explicit page inputs.
        self,
        page_context: WikiPageContext,
        proposal: WikiCreationResult,
        *,
        page_manifest: WikiPageManifest,
        wiki_path: Path,
        source_pdfs: Mapping[str, Path],
        correction_count: int,
    ) -> WikiReview:
        """Run a reviewer in a fresh sandbox after transient failures."""

        def operation() -> WikiReview:
            reviewer = create_wiki_reviewer_agent(
                model=self.reviewer_model,
                transient_retries=self.transient_retries,
            )
            return run_wiki_reviewer_agent(
                reviewer,
                page_context,
                proposal,
                wiki_path=wiki_path,
                source_pdfs=source_pdfs,
                page_manifest=page_manifest,
            )

        return run_with_infrastructure_retries(
            operation,
            phase="reviewer",
            retries=self.transient_retries,
            correction_count=correction_count,
        )

    @staticmethod
    def _proposal_from_retry_state(
        page_context: WikiPageContext,
        retry_state: WikiPageRetryState,
    ) -> WikiCreationResult:
        """Reconstruct a saved proposal without applying stale findings."""
        plan = page_context.page_job.plan
        if plan.target_path is None:
            raise RuntimeError("a resumable page plan has no target path")
        change_status = (
            WikiChangeStatus.ADDED
            if plan.decision is WikiPlanDecision.CREATE
            else WikiChangeStatus.MODIFIED
        )
        return WikiCreationResult(
            plan=plan,
            changes=(
                WikiFileChange(
                    path=plan.target_path,
                    status=change_status,
                    content=retry_state.content,
                ),
            ),
            raw_output=retry_state.raw_output,
        )

    @staticmethod
    def _proposal_state(
        proposal: WikiCreationResult,
        findings: Sequence[Mapping[str, object]],
        correction_runs: int,
    ) -> WikiPageRetryState:
        """Build resumable state for the latest valid editor proposal."""
        change = proposal.changes[0]
        if change.content is None:
            raise RuntimeError("a resumable page proposal has no Markdown content")
        return WikiPageRetryState(
            content=change.content,
            raw_output=proposal.raw_output,
            findings=[WikiReviewFinding.model_validate(item) for item in findings],
            correction_runs=correction_runs,
        )

    @staticmethod
    def _failed_review_outcome(
        page_id: str,
        proposal: WikiCreationResult,
        findings: Sequence[Mapping[str, object]],
        correction_runs: int,
    ) -> WikiPageOutcome:
        """Return a failed outcome that can resume from the reviewed proposal."""
        change = proposal.changes[0]
        if change.content is None:
            raise RuntimeError("a resumable page proposal has no Markdown content")
        typed_findings = [WikiReviewFinding.model_validate(item) for item in findings]
        return WikiPageOutcome(
            page_id,
            WikiConceptStatus.FAILED,
            correction_runs=correction_runs,
            error="; ".join(finding.problem for finding in typed_findings),
            retry_state=WikiPageRetryState(
                content=change.content,
                raw_output=proposal.raw_output,
                findings=typed_findings,
                correction_runs=correction_runs,
            ),
            failure_category=WikiPageFailureCategory.REVISION,
        )


def run_page_jobs_concurrently(  # ruff: ignore[too-many-arguments,complex-structure] - Bounded scheduling needs explicit shared inputs and circuit state.
    workflow: WikiPageWorkflow,
    contexts: Sequence[WikiPageContext],
    *,
    page_manifest: WikiPageManifest,
    wiki_path: Path,
    source_pdfs: Mapping[str, Path],
    workers: int = 4,
    progress: Callable[[int], None] | None = None,
    checkpoint: Callable[[WikiPageOutcome], None] | None = None,
    proposal_checkpoint: Callable[[str, WikiPageRetryState], None] | None = None,
    retry_states: Mapping[str, WikiPageRetryState] | None = None,
) -> list[WikiPageOutcome]:
    """Run bounded page workflows and stop after repeated infrastructure failures."""
    if workers < 1:
        raise ValueError("page workers must be positive")

    def run(context: WikiPageContext) -> WikiPageOutcome:
        outcome = workflow.run(
            context,
            page_manifest=page_manifest,
            wiki_path=wiki_path,
            source_pdfs=source_pdfs,
            retry_state=(retry_states or {}).get(context.page_job.page_id),
            proposal_checkpoint=proposal_checkpoint,
        )
        if checkpoint is not None:
            checkpoint(outcome)
        if progress is not None:
            progress(1)
        return outcome

    outcomes: dict[str, WikiPageOutcome] = {}
    pending_contexts = iter(contexts)
    consecutive_infrastructure = 0
    circuit_open = False
    with ThreadPoolExecutor(max_workers=workers) as executor:
        in_flight: dict[Future[WikiPageOutcome], WikiPageContext] = {}

        def submit_one() -> bool:
            """Submit one context when work remains."""
            try:
                context = next(pending_contexts)
            except StopIteration:
                return False
            in_flight[executor.submit(run, context)] = context
            return True

        for _ in range(min(workers, len(contexts))):
            submit_one()
        while in_flight:
            completed, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in completed:
                context = in_flight.pop(future)
                outcome = future.result()
                outcomes[context.page_job.page_id] = outcome
                if outcome.failure_category is WikiPageFailureCategory.INFRASTRUCTURE:
                    consecutive_infrastructure += 1
                else:
                    consecutive_infrastructure = 0
                if consecutive_infrastructure >= _INFRASTRUCTURE_FAILURE_LIMIT:
                    circuit_open = True
                    logger.error(
                        "Wiki page infrastructure circuit opened after %d "
                        "consecutive terminal outcomes",
                        consecutive_infrastructure,
                    )
            if not circuit_open:
                while len(in_flight) < workers and submit_one():
                    pass
    if circuit_open:
        raise WikiInfrastructureUnavailableError(
            "Wiki page infrastructure is unavailable after four consecutive failures",
        )
    return [outcomes[context.page_job.page_id] for context in contexts]
