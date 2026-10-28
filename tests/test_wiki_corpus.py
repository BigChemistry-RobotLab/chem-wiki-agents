"""Test full-corpus Wiki planning and deterministic aggregation."""

import asyncio
import json
from pathlib import Path
from threading import Barrier, Lock
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from uuid import UUID

import pytest
from agents.exceptions import ModelBehaviorError
from pydantic import ValidationError

if TYPE_CHECKING:
    from agents.sandbox.session import BaseSandboxSession

from wiki_agents.agents.wiki import finalizer as wiki_finalizer_module
from wiki_agents.agents.wiki import workflow as wiki_workflow_module
from wiki_agents.agents.wiki.aggregator import (
    WikiPageGroup,
    assign_target_paths,
    build_page_contexts,
    build_page_manifest,
    collapse_merge_relations,
    compile_page_definitions_concurrently,
    create_wiki_page_definition_agent,
    deterministic_page_id,
    singleton_page_definition,
)
from wiki_agents.agents.wiki.finalizer import (
    create_wiki_finalizer_agent,
    run_wiki_finalizer_agent,
    validate_finalizer_scope,
)
from wiki_agents.agents.wiki.models import (
    WikiCandidateBatchPlan,
    WikiCandidateDecision,
    WikiCandidatePlan,
    WikiChangeStatus,
    WikiCreationResult,
    WikiDirectedRelation,
    WikiFileChange,
    WikiPageContext,
    WikiPageDefinition,
    WikiPageDefinitionBatch,
    WikiPageDefinitionDraft,
    WikiPageIntent,
    WikiPageJob,
    WikiPageManifest,
    WikiPlan,
    WikiPlanDecision,
    WikiRelationBatch,
    WikiRelationDecision,
    WikiRelationResolution,
    WikiReview,
    WikiReviewDecision,
    WikiReviewFinding,
    WikiReviewFindingType,
)
from wiki_agents.agents.wiki.planner import (
    create_wiki_candidate_planner_agent,
    plan_candidates_concurrently,
)
from wiki_agents.agents.wiki.relation_resolver import (
    create_wiki_relation_resolver_agent,
)
from wiki_agents.agents.wiki.resilience import (
    WikiInfrastructureUnavailableError,
    WikiPageFailureCategory,
)
from wiki_agents.agents.wiki.resolver import (
    WikiRelationCandidate,
    resolve_relations_concurrently,
    retrieve_relation_candidates,
)
from wiki_agents.agents.wiki.workflow import (
    WikiConceptStatus,
    WikiPageOutcome,
    WikiPageWorkflow,
    WikiPartitionBoundaryWorkflow,
    run_page_jobs_concurrently,
)
from wiki_agents.concept.models import Claim, Concept, ConceptCandidate, ConceptContext
from wiki_agents.document.models import EvidenceSpan, SourceRef
from wiki_agents.embedding.formats import format_work_package
from wiki_agents.embedding.work_package_store import WorkPackageMatch
from wiki_agents.orchestrator.wiki_publication import WikiPublicationMixin


def _plan(number: int, subject: str) -> WikiCandidatePlan:
    return WikiCandidatePlan(
        candidate_id=UUID(int=number),
        decision=WikiCandidateDecision.CREATE,
        rationale=f"Represent {subject}.",
        page_intent=WikiPageIntent(
            canonical_subject=subject,
            scope=f"Scope for {subject}.",
            work_summary=[f"Explain {subject}."],
        ),
    )


def _candidate(number: int) -> ConceptCandidate:
    return ConceptCandidate(
        candidate_id=UUID(int=number),
        source=SourceRef(
            source_id=f"paper_{number}",
            sha256=f"{number:064x}",
            resource=f"/references/papers/paper_{number}.md",
        ),
        concept=Concept(
            title=f"Concept {number}",
            description=f"Description {number}.",
            claims=[
                Claim(
                    statement=f"Claim {number}.",
                    evidence=[
                        EvidenceSpan(
                            start_line_id="p0001-l0001",
                            end_line_id="p0001-l0002",
                        ),
                    ],
                ),
            ],
        ),
    )


def _page_context(number: int) -> WikiPageContext:
    """Return one page context for concurrency tests."""
    return WikiPageContext(
        page_job=WikiPageJob(
            page_id=f"page-{number}",
            plan=WikiPlan(
                decision=WikiPlanDecision.CREATE,
                rationale="Create the page.",
                target_path=f"wiki/concepts/page-{number}.md",
                title=f"Page {number}",
                work_summary=["Explain the concept."],
            ),
            member_candidate_ids=[UUID(int=number)],
        ),
        candidates=[_candidate(number)],
    )


def test_page_workers_checkpoint_each_outcome_before_starting_next() -> None:
    checkpointed: list[str] = []

    def run(context, **_kwargs):
        if context.page_job.page_id == "page-2":
            assert checkpointed == ["page-1"]
        return WikiPageOutcome(
            page_id=context.page_job.page_id,
            status=WikiConceptStatus.FAILED,
            error="Review failed.",
        )

    workflow = cast("WikiPageWorkflow", SimpleNamespace(run=run))
    outcomes = run_page_jobs_concurrently(
        workflow,
        [_page_context(1), _page_context(2)],
        page_manifest=WikiPageManifest(pages=[]),
        wiki_path=Path("wiki"),
        source_pdfs={},
        workers=1,
        checkpoint=lambda outcome: checkpointed.append(outcome.page_id),
    )

    assert checkpointed == ["page-1", "page-2"]
    assert [outcome.page_id for outcome in outcomes] == checkpointed


def test_page_scheduler_bounds_in_flight_work() -> None:
    barrier = Barrier(4)
    lock = Lock()
    active = 0
    maximum_active = 0

    def run(context, **_kwargs):
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        barrier.wait()
        with lock:
            active -= 1
        return WikiPageOutcome(
            page_id=context.page_job.page_id,
            status=WikiConceptStatus.ACCEPTED,
        )

    workflow = cast("WikiPageWorkflow", SimpleNamespace(run=run))
    outcomes = run_page_jobs_concurrently(
        workflow,
        [_page_context(number) for number in range(1, 9)],
        page_manifest=WikiPageManifest(pages=[]),
        wiki_path=Path("wiki"),
        source_pdfs={},
        workers=4,
    )

    assert len(outcomes) == 8
    assert maximum_active == 4


def test_page_scheduler_drains_after_infrastructure_circuit_opens() -> None:
    started: list[str] = []
    checkpointed: list[str] = []

    def run(context, **_kwargs):
        started.append(context.page_job.page_id)
        return WikiPageOutcome(
            page_id=context.page_job.page_id,
            status=WikiConceptStatus.FAILED,
            error="Docker is unavailable.",
            failure_category=WikiPageFailureCategory.INFRASTRUCTURE,
        )

    workflow = cast("WikiPageWorkflow", SimpleNamespace(run=run))
    with pytest.raises(WikiInfrastructureUnavailableError):
        run_page_jobs_concurrently(
            workflow,
            [_page_context(number) for number in range(1, 13)],
            page_manifest=WikiPageManifest(pages=[]),
            wiki_path=Path("wiki"),
            source_pdfs={},
            workers=4,
            checkpoint=lambda outcome: checkpointed.append(outcome.page_id),
        )

    assert 4 <= len(started) <= 7
    assert sorted(checkpointed) == sorted(started)


def test_failed_page_review_resumes_from_latest_proposal(monkeypatch, tmp_path) -> None:
    context = _page_context(1)
    plan = context.page_job.plan
    creator_calls = []
    creator_agents = []
    reviewer_agents = []
    accept_review = False

    def create_creator(**_kwargs):
        agent = object()
        creator_agents.append(agent)
        return agent

    def create_reviewer(**_kwargs):
        agent = object()
        reviewer_agents.append(agent)
        return agent

    monkeypatch.setattr(
        wiki_workflow_module,
        "create_wiki_agent",
        create_creator,
    )
    monkeypatch.setattr(
        wiki_workflow_module,
        "create_wiki_reviewer_agent",
        create_reviewer,
    )

    def create(_agent, _context, _plan, **kwargs):
        creator_calls.append((_agent, kwargs))
        content = f"proposal-{len(creator_calls)}".encode()
        return WikiCreationResult(
            plan=plan,
            changes=(
                WikiFileChange(
                    path=plan.target_path or "",
                    status=WikiChangeStatus.ADDED,
                    content=content,
                ),
            ),
            raw_output=f"creator-{len(creator_calls)}",
        )

    finding = WikiReviewFinding(
        finding_type=WikiReviewFindingType.LINK,
        target=plan.target_path or "wiki",
        problem="A manifest-planned target is not on disk yet.",
        required_change="Remove the planned link.",
        basis="Page-local filesystem check.",
    )

    def review(*_args: object, **_kwargs: object):
        if accept_review:
            return WikiReview(
                decision=WikiReviewDecision.ACCEPT,
                summary="The page is valid.",
            )
        return WikiReview(
            decision=WikiReviewDecision.REVISE,
            summary="The page needs a correction.",
            findings=[finding],
        )

    monkeypatch.setattr(wiki_workflow_module, "run_wiki_agent", create)
    monkeypatch.setattr(wiki_workflow_module, "run_wiki_reviewer_agent", review)
    monkeypatch.setattr(
        wiki_workflow_module,
        "normalize_managed_metadata",
        lambda proposal, **_kwargs: proposal,
    )
    monkeypatch.setattr(
        wiki_workflow_module,
        "add_machine_verification",
        lambda proposal, **_kwargs: proposal,
    )
    monkeypatch.setattr(
        wiki_workflow_module,
        "validate_wiki_page_proposal",
        lambda *_args, **_kwargs: (),
    )
    monkeypatch.setattr(wiki_workflow_module, "read_host_wiki", lambda _path: {})
    workflow = WikiPageWorkflow(
        creator_model=None,
        reviewer_model=None,
        max_correction_runs=1,
    )

    failed = workflow._run_page(
        context,
        page_manifest=WikiPageManifest(pages=[]),
        wiki_path=tmp_path,
        source_pdfs={},
        retry_state=None,
    )

    assert failed.status is WikiConceptStatus.FAILED
    assert failed.correction_runs == 1
    assert failed.retry_state is not None
    assert failed.retry_state.content == b"proposal-2"
    accept_review = True
    resumed_states = []

    accepted = workflow._run_page(
        context,
        page_manifest=WikiPageManifest(pages=[]),
        wiki_path=tmp_path,
        source_pdfs={},
        retry_state=failed.retry_state,
        proposal_checkpoint=lambda _page_id, state: resumed_states.append(state),
    )

    pending_review = workflow._run_page(
        context,
        page_manifest=WikiPageManifest(pages=[]),
        wiki_path=tmp_path,
        source_pdfs={},
        retry_state=failed.retry_state.model_copy(update={"findings": []}),
    )

    assert accepted.status is WikiConceptStatus.ACCEPTED
    assert pending_review.status is WikiConceptStatus.ACCEPTED
    assert accepted.correction_runs == 1
    assert resumed_states[0].findings == []
    assert len(creator_agents) == 2
    assert creator_agents[0] is not creator_agents[1]
    assert len(reviewer_agents) == 4
    assert len({id(agent) for agent in reviewer_agents}) == 4
    assert creator_calls[0][0] is not creator_calls[1][0]
    assert len(creator_calls) == 2
    assert creator_calls[-1][1]["previous_result"].changes[0].content == b"proposal-1"
    assert creator_calls[-1][1]["correction_findings"] == [
        finding.model_dump(mode="json"),
    ]


def test_candidate_plan_has_no_path_and_validates_intent() -> None:
    assert "target_path" not in WikiCandidatePlan.model_fields
    with pytest.raises(ValidationError, match="requires a page intent"):
        WikiCandidatePlan(
            candidate_id=UUID(int=1),
            decision=WikiCandidateDecision.CREATE,
            rationale="Useful knowledge.",
        )
    with pytest.raises(ValidationError, match="cannot have a page intent"):
        WikiCandidatePlan(
            candidate_id=UUID(int=1),
            decision=WikiCandidateDecision.SKIP,
            rationale="Not useful.",
            page_intent=WikiPageIntent(
                canonical_subject="Subject",
                scope="Scope.",
                work_summary=["Work."],
            ),
        )


def test_work_package_format_contains_only_semantic_intent() -> None:
    plan = _plan(1, "Life cycle assessment methodology")
    assert plan.page_intent is not None
    text = format_work_package(plan.page_intent)

    assert text == (
        "subject: Life cycle assessment methodology\n"
        "scope: Scope for Life cycle assessment methodology.\n"
        "work: Explain Life cycle assessment methodology."
    )
    assert str(plan.candidate_id) not in text
    assert plan.rationale not in text
    assert "references" not in text


def test_candidate_planning_checkpoints_successful_batches_and_resumes(
    monkeypatch,
) -> None:
    contexts = [
        ConceptContext(candidate=_candidate(number), similar_concepts=[])
        for number in range(1, 5)
    ]
    calls: list[list[UUID]] = []
    fail_last_batch = True

    def run_sync(_agent, payload):
        nonlocal fail_last_batch
        candidate_ids = [
            UUID(candidate["candidate_id"])
            for candidate in json.loads(payload)["candidates"]
        ]
        calls.append(candidate_ids)
        if fail_last_batch and candidate_ids == [UUID(int=3), UUID(int=4)]:
            raise RuntimeError("invalid candidate batch")
        output = WikiCandidateBatchPlan(
            items=[
                _plan(candidate_id.int, f"Subject {candidate_id.int}")
                for candidate_id in candidate_ids
            ],
        )
        return SimpleNamespace(final_output_as=lambda *_args, **_kwargs: output)

    monkeypatch.setattr("wiki_agents.agents.wiki.planner.Runner.run_sync", run_sync)
    checkpoints: list[list[WikiCandidatePlan]] = []
    agent = create_wiki_candidate_planner_agent(model="test-model")

    with pytest.raises(RuntimeError, match="invalid candidate batch"):
        plan_candidates_concurrently(
            agent,
            contexts,
            batch_size=2,
            workers=1,
            checkpoint=lambda plans: checkpoints.append(plans.copy()),
        )

    assert [[plan.candidate_id.int for plan in plans] for plans in checkpoints] == [
        [1, 2],
    ]
    fail_last_batch = False
    calls.clear()

    plans = plan_candidates_concurrently(
        agent,
        contexts,
        batch_size=2,
        workers=1,
        completed_plans=checkpoints[-1],
        checkpoint=lambda items: checkpoints.append(items.copy()),
    )

    assert [[candidate_id.int for candidate_id in batch] for batch in calls] == [[3, 4]]
    assert [plan.candidate_id.int for plan in plans] == [1, 2, 3, 4]


def test_candidate_planning_retries_duplicate_results_as_singletons(
    monkeypatch,
) -> None:
    contexts = [
        ConceptContext(candidate=_candidate(number), similar_concepts=[])
        for number in range(1, 3)
    ]
    calls: list[list[UUID]] = []

    def run_sync(_agent, payload):
        candidate_ids = [
            UUID(candidate["candidate_id"])
            for candidate in json.loads(payload)["candidates"]
        ]
        calls.append(candidate_ids)
        if len(candidate_ids) > 1:
            items = [_plan(1, "First version"), _plan(1, "Duplicate version")]
        else:
            candidate_id = candidate_ids[0]
            items = [_plan(candidate_id.int, f"Subject {candidate_id.int}")]
        output = SimpleNamespace(items=items)
        return SimpleNamespace(final_output_as=lambda *_args, **_kwargs: output)

    monkeypatch.setattr("wiki_agents.agents.wiki.planner.Runner.run_sync", run_sync)
    agent = create_wiki_candidate_planner_agent(model="test-model")

    plans = plan_candidates_concurrently(
        agent,
        contexts,
        batch_size=2,
        workers=1,
    )

    assert calls == [[UUID(int=1), UUID(int=2)], [UUID(int=1)], [UUID(int=2)]]
    assert [plan.candidate_id for plan in plans] == [UUID(int=1), UUID(int=2)]


def test_sparse_pairs_come_only_from_retrieved_neighbors() -> None:
    plans = [_plan(number, f"Subject {number}") for number in range(1, 1901)]

    class SparseStore:
        calls = 0

        def top_k_similar_work_packages(self, plan, *, top_k):
            self.calls += 1
            if plan.candidate_id.int == 1900:
                return []
            return [WorkPackageMatch(plans[plan.candidate_id.int], 0.9)][:top_k]

    store = SparseStore()
    pairs = retrieve_relation_candidates(plans, store, top_k=20)

    assert store.calls == 1900
    assert len(pairs) == 1899
    assert len(pairs) < 1900 * 20


def test_relation_resolution_serializes_candidate_ids(monkeypatch) -> None:
    plans = {
        UUID(int=number): _plan(number, f"Subject {number}") for number in range(1, 3)
    }
    candidates = {UUID(int=number): _candidate(number) for number in range(1, 3)}
    pair = WikiRelationCandidate(
        candidate_ids=(UUID(int=1), UUID(int=2)),
        similarity_score=0.9,
    )

    def run_sync(_agent, payload):
        pair_payload = json.loads(payload)["pairs"][0]
        assert pair_payload["candidate_ids"] == [str(UUID(int=1)), str(UUID(int=2))]
        assert "ranking_similarity" not in pair_payload
        assert "reciprocal" not in pair_payload
        assert "pair_priority" not in pair_payload
        output = WikiRelationBatch(
            items=[
                WikiRelationResolution(
                    candidate_ids=(UUID(int=1), UUID(int=2)),
                    decision=WikiRelationDecision.NONE,
                    rationale="No direct relation is required.",
                ),
            ],
        )
        return SimpleNamespace(final_output_as=lambda *_args, **_kwargs: output)

    monkeypatch.setattr("wiki_agents.agents.wiki.resolver.Runner.run_sync", run_sync)
    agent = create_wiki_relation_resolver_agent(model="test-model")

    relations = resolve_relations_concurrently(
        agent,
        [pair],
        plans,
        candidates,
        batch_size=1,
        workers=1,
    )

    assert relations[0].candidate_ids == pair.candidate_ids


def test_partition_boundary_rejects_merge_result(monkeypatch) -> None:
    plans = {UUID(int=number): _plan(number, f"Subject {number}") for number in (1, 2)}
    candidates = {UUID(int=number): _candidate(number) for number in (1, 2)}
    pair = WikiRelationCandidate(
        candidate_ids=(UUID(int=1), UUID(int=2)),
        similarity_score=0.9,
    )

    def run_sync(_agent, _payload):
        output = WikiRelationBatch(
            items=[
                WikiRelationResolution(
                    candidate_ids=pair.candidate_ids,
                    decision=WikiRelationDecision.MERGE,
                    rationale="The subjects overlap.",
                ),
            ],
        )
        return SimpleNamespace(final_output_as=lambda *_args, **_kwargs: output)

    monkeypatch.setattr("wiki_agents.agents.wiki.resolver.Runner.run_sync", run_sync)

    with pytest.raises(RuntimeError, match="boundary cannot retain MERGE"):
        WikiPartitionBoundaryWorkflow(model="test-model").run(
            [pair],
            plans,
            candidates,
            batch_size=1,
            workers=1,
        )


def test_page_definition_compilation_serializes_candidate_ids(monkeypatch) -> None:
    candidate_ids = (UUID(int=1), UUID(int=2))
    group = WikiPageGroup(
        page_id="page-a",
        member_candidate_ids=candidate_ids,
    )
    plans = {
        candidate_id: _plan(candidate_id.int, f"Subject {candidate_id.int}")
        for candidate_id in candidate_ids
    }
    candidates = {
        candidate_id: _candidate(candidate_id.int) for candidate_id in candidate_ids
    }

    def run_sync(_agent, payload):
        assert json.loads(payload)["member_candidate_ids"] == [
            str(candidate_id) for candidate_id in candidate_ids
        ]
        output = WikiPageDefinitionBatch(
            items=[
                WikiPageDefinitionDraft(
                    title="Combined subject",
                    scope="Combined scope.",
                    work_summary=["Explain the combined subject."],
                    member_candidate_ids=list(candidate_ids),
                ),
            ],
        )
        return SimpleNamespace(final_output_as=lambda *_args, **_kwargs: output)

    monkeypatch.setattr("wiki_agents.agents.wiki.aggregator.Runner.run_sync", run_sync)

    definitions = compile_page_definitions_concurrently(
        create_wiki_page_definition_agent(model="test-model"),
        [group],
        plans,
        candidates,
        workers=1,
    )

    assert definitions == [
        WikiPageDefinition(
            page_id=deterministic_page_id(candidate_ids),
            title="Combined subject",
            scope="Combined scope.",
            work_summary=["Explain the combined subject."],
            member_candidate_ids=list(candidate_ids),
        ),
    ]


def test_page_definition_compiler_can_partition_oversized_group(monkeypatch) -> None:
    candidate_ids = tuple(UUID(int=number) for number in range(1, 4))
    group = WikiPageGroup(
        page_id="provisional-page",
        member_candidate_ids=candidate_ids,
    )
    plans = {
        candidate_id: _plan(candidate_id.int, f"Subject {candidate_id.int}")
        for candidate_id in candidate_ids
    }
    candidates = {
        candidate_id: _candidate(candidate_id.int) for candidate_id in candidate_ids
    }

    def run_sync(_agent, _payload):
        output = WikiPageDefinitionBatch(
            items=[
                WikiPageDefinitionDraft(
                    title="First combined subject",
                    scope="First scope.",
                    work_summary=["Explain the first subject."],
                    member_candidate_ids=list(candidate_ids[:2]),
                ),
                WikiPageDefinitionDraft(
                    title="Third subject",
                    scope="Third scope.",
                    work_summary=["Explain the third subject."],
                    member_candidate_ids=[candidate_ids[2]],
                ),
            ],
        )
        return SimpleNamespace(final_output_as=lambda *_args, **_kwargs: output)

    monkeypatch.setattr("wiki_agents.agents.wiki.aggregator.Runner.run_sync", run_sync)

    definitions = compile_page_definitions_concurrently(
        create_wiki_page_definition_agent(model="test-model"),
        [group],
        plans,
        candidates,
        max_page_candidates=2,
        workers=1,
    )

    assert {tuple(item.member_candidate_ids) for item in definitions} == {
        candidate_ids[:2],
        candidate_ids[2:],
    }


def test_page_definition_compiler_retries_changed_membership(monkeypatch) -> None:
    candidate_ids = (UUID(int=1), UUID(int=2))
    unexpected_id = UUID(int=3)
    group = WikiPageGroup(
        page_id="provisional-page",
        member_candidate_ids=candidate_ids,
    )
    plans = {
        candidate_id: _plan(candidate_id.int, f"Subject {candidate_id.int}")
        for candidate_id in candidate_ids
    }
    candidates = {
        candidate_id: _candidate(candidate_id.int) for candidate_id in candidate_ids
    }
    payloads = []

    def run_sync(_agent, payload):
        payloads.append(json.loads(payload))
        members = [unexpected_id] if len(payloads) == 1 else list(candidate_ids)
        output = WikiPageDefinitionBatch(
            items=[
                WikiPageDefinitionDraft(
                    title="Combined subject",
                    scope="Combined scope.",
                    work_summary=["Explain the combined subject."],
                    member_candidate_ids=members,
                ),
            ],
        )
        return SimpleNamespace(final_output_as=lambda *_args, **_kwargs: output)

    monkeypatch.setattr("wiki_agents.agents.wiki.aggregator.Runner.run_sync", run_sync)

    definitions = compile_page_definitions_concurrently(
        create_wiki_page_definition_agent(model="test-model"),
        [group],
        plans,
        candidates,
        workers=1,
    )

    assert definitions[0].member_candidate_ids == list(candidate_ids)
    assert payloads[1]["membership_correction"] == {
        "instruction": (
            "Return a new complete batch. Preserve every supplied member candidate ID "
            "exactly once and do not add candidate IDs."
        ),
        "missing_candidate_ids": [str(candidate_id) for candidate_id in candidate_ids],
        "unexpected_candidate_ids": [str(unexpected_id)],
    }


def test_page_definition_compiler_stops_after_membership_retries(monkeypatch) -> None:
    candidate_ids = (UUID(int=1), UUID(int=2))
    group = WikiPageGroup(
        page_id="provisional-page",
        member_candidate_ids=candidate_ids,
    )
    plans = {
        candidate_id: _plan(candidate_id.int, f"Subject {candidate_id.int}")
        for candidate_id in candidate_ids
    }
    candidates = {
        candidate_id: _candidate(candidate_id.int) for candidate_id in candidate_ids
    }
    call_count = 0

    def run_sync(_agent, _payload):
        nonlocal call_count
        call_count += 1
        output = WikiPageDefinitionBatch(
            items=[
                WikiPageDefinitionDraft(
                    title="Incomplete subject",
                    scope="Incomplete scope.",
                    work_summary=["Explain the incomplete subject."],
                    member_candidate_ids=[candidate_ids[0]],
                ),
            ],
        )
        return SimpleNamespace(final_output_as=lambda *_args, **_kwargs: output)

    monkeypatch.setattr("wiki_agents.agents.wiki.aggregator.Runner.run_sync", run_sync)

    with pytest.raises(RuntimeError, match="changed group membership"):
        compile_page_definitions_concurrently(
            create_wiki_page_definition_agent(model="test-model"),
            [group],
            plans,
            candidates,
            max_membership_retries=2,
            workers=1,
        )

    assert call_count == 3


def test_page_definition_compiler_checkpoints_and_resumes(monkeypatch) -> None:
    groups = [
        WikiPageGroup(
            page_id=f"provisional-{number}",
            member_candidate_ids=(UUID(int=number), UUID(int=number + 1)),
        )
        for number in (1, 3)
    ]
    candidate_ids = tuple(
        candidate_id for group in groups for candidate_id in group.member_candidate_ids
    )
    plans = {
        candidate_id: _plan(candidate_id.int, f"Subject {candidate_id.int}")
        for candidate_id in candidate_ids
    }
    candidates = {
        candidate_id: _candidate(candidate_id.int) for candidate_id in candidate_ids
    }
    failed_group = set(groups[1].member_candidate_ids)
    fail_second_group = True
    calls: list[set[UUID]] = []

    def run_sync(_agent, payload):
        nonlocal fail_second_group
        members = {
            UUID(candidate_id)
            for candidate_id in json.loads(payload)["member_candidate_ids"]
        }
        calls.append(members)
        if fail_second_group and members == failed_group:
            raise RuntimeError("page-definition service unavailable")
        output = WikiPageDefinitionBatch(
            items=[
                WikiPageDefinitionDraft(
                    title="Combined subject",
                    scope="Combined scope.",
                    work_summary=["Explain the combined subject."],
                    member_candidate_ids=sorted(members, key=str),
                ),
            ],
        )
        return SimpleNamespace(final_output_as=lambda *_args, **_kwargs: output)

    monkeypatch.setattr("wiki_agents.agents.wiki.aggregator.Runner.run_sync", run_sync)
    checkpoints: list[list[WikiPageDefinition]] = []
    agent = create_wiki_page_definition_agent(model="test-model")

    with pytest.raises(RuntimeError, match="page-definition service unavailable"):
        compile_page_definitions_concurrently(
            agent,
            groups,
            plans,
            candidates,
            workers=1,
            checkpoint=lambda items: checkpoints.append(items.copy()),
        )

    assert len(checkpoints[-1]) == 1
    fail_second_group = False
    calls.clear()

    definitions = compile_page_definitions_concurrently(
        agent,
        groups,
        plans,
        candidates,
        workers=1,
        completed_definitions=checkpoints[-1],
        checkpoint=lambda items: checkpoints.append(items.copy()),
    )

    assert calls == [failed_group]
    assert len(definitions) == 2


def test_relation_resolution_checkpoints_responses_and_resumes(monkeypatch) -> None:
    plans = {
        UUID(int=number): _plan(number, f"Subject {number}") for number in range(1, 5)
    }
    candidates = {UUID(int=number): _candidate(number) for number in range(1, 5)}
    pairs = [
        WikiRelationCandidate(
            candidate_ids=(UUID(int=1), UUID(int=2)),
            similarity_score=0.9,
        ),
        WikiRelationCandidate(
            candidate_ids=(UUID(int=3), UUID(int=4)),
            similarity_score=0.8,
        ),
    ]
    calls: list[tuple[UUID, UUID]] = []
    fail_second_pair = True

    def run_sync(_agent, payload):
        nonlocal fail_second_pair
        pair_payload = json.loads(payload)["pairs"][0]
        serialized_ids = pair_payload["candidate_ids"]
        candidate_ids = (UUID(serialized_ids[0]), UUID(serialized_ids[1]))
        calls.append(candidate_ids)
        if fail_second_pair and candidate_ids == pairs[1].candidate_ids:
            raise RuntimeError("relation service unavailable")
        output = WikiRelationBatch(
            items=[
                WikiRelationResolution(
                    candidate_ids=candidate_ids,
                    decision=WikiRelationDecision.NONE,
                    rationale="No direct relation is required.",
                ),
            ],
        )
        return SimpleNamespace(final_output_as=lambda *_args, **_kwargs: output)

    monkeypatch.setattr("wiki_agents.agents.wiki.resolver.Runner.run_sync", run_sync)
    checkpoints: list[list[WikiRelationResolution]] = []
    agent = create_wiki_relation_resolver_agent(model="test-model")

    with pytest.raises(RuntimeError, match="relation service unavailable"):
        resolve_relations_concurrently(
            agent,
            pairs,
            plans,
            candidates,
            batch_size=1,
            workers=1,
            checkpoint=lambda items: checkpoints.append(items.copy()),
        )

    assert [item.candidate_ids for item in checkpoints[-1]] == [
        pairs[0].candidate_ids,
    ]
    fail_second_pair = False
    calls.clear()

    relations = resolve_relations_concurrently(
        agent,
        list(reversed(pairs)),
        plans,
        candidates,
        batch_size=1,
        workers=1,
        completed_relations=checkpoints[-1],
        checkpoint=lambda items: checkpoints.append(items.copy()),
    )

    assert calls == [pairs[1].candidate_ids]
    assert [relation.candidate_ids for relation in relations] == [
        pair.candidate_ids for pair in reversed(pairs)
    ]


def test_relation_resolution_splits_invalid_model_batch(monkeypatch) -> None:
    plans = {
        UUID(int=number): _plan(number, f"Subject {number}") for number in range(1, 5)
    }
    candidates = {UUID(int=number): _candidate(number) for number in range(1, 5)}
    pairs = [
        WikiRelationCandidate(
            candidate_ids=(UUID(int=1), UUID(int=2)),
            similarity_score=0.9,
        ),
        WikiRelationCandidate(
            candidate_ids=(UUID(int=3), UUID(int=4)),
            similarity_score=0.8,
        ),
    ]
    calls = []

    def run_sync(_agent, payload):
        pair_payloads = json.loads(payload)["pairs"]
        requested_pairs = [
            tuple(UUID(candidate_id) for candidate_id in item["candidate_ids"])
            for item in pair_payloads
        ]
        calls.append(requested_pairs)
        if len(requested_pairs) > 1:
            raise ModelBehaviorError("invalid final output")
        output = WikiRelationBatch(
            items=[
                WikiRelationResolution(
                    candidate_ids=requested_pairs[0],
                    decision=WikiRelationDecision.NONE,
                    rationale="No direct relation is required.",
                ),
            ],
        )
        return SimpleNamespace(final_output_as=lambda *_args, **_kwargs: output)

    monkeypatch.setattr("wiki_agents.agents.wiki.resolver.Runner.run_sync", run_sync)
    agent = create_wiki_relation_resolver_agent(model="test-model")

    relations = resolve_relations_concurrently(
        agent,
        pairs,
        plans,
        candidates,
        batch_size=2,
        workers=1,
    )

    assert calls == [
        [pairs[0].candidate_ids, pairs[1].candidate_ids],
        [pairs[0].candidate_ids],
        [pairs[1].candidate_ids],
    ]
    assert [relation.candidate_ids for relation in relations] == [
        pair.candidate_ids for pair in pairs
    ]


def test_relation_resolution_schema_defines_array_items() -> None:
    schema = WikiRelationBatch.model_json_schema()
    relation_schema = schema["$defs"]["WikiRelationResolution"]
    candidate_ids_schema = relation_schema["properties"]["candidate_ids"]

    assert candidate_ids_schema["items"] == {"format": "uuid", "type": "string"}
    assert candidate_ids_schema["minItems"] == 2
    assert candidate_ids_schema["maxItems"] == 2
    assert "prefixItems" not in candidate_ids_schema


def test_merge_collapse_lifts_links_and_drops_duplicates() -> None:
    plans = [_plan(number, f"Subject {number}") for number in range(1, 5)]

    def relation(first, second, decision, links=()):
        return WikiRelationResolution(
            candidate_ids=(UUID(int=first), UUID(int=second)),
            decision=decision,
            rationale="Fixture relation.",
            links=list(links),
        )

    relations = [
        relation(1, 2, WikiRelationDecision.MERGE),
        relation(2, 3, WikiRelationDecision.MERGE),
        relation(
            1,
            4,
            WikiRelationDecision.LINK,
            [
                WikiDirectedRelation(
                    source_candidate_id=UUID(int=1),
                    target_candidate_id=UUID(int=4),
                    relationship="Related.",
                ),
            ],
        ),
        relation(
            3,
            4,
            WikiRelationDecision.LINK,
            [
                WikiDirectedRelation(
                    source_candidate_id=UUID(int=3),
                    target_candidate_id=UUID(int=4),
                    relationship="Related.",
                ),
            ],
        ),
    ]

    pair_metadata = {
        relation.candidate_ids: WikiRelationCandidate(
            candidate_ids=(relation.candidate_ids[0], relation.candidate_ids[1]),
            similarity_score=0.8,
            ranking_similarity=0.8,
        )
        for relation in relations
    }
    groups, links = collapse_merge_relations(plans, relations, pair_metadata)

    assert sorted(len(group.member_candidate_ids) for group in groups) == [1, 3]
    assert len(links) == 1
    assert links[0].source_page_id != links[0].target_page_id
    assert links[0].relationships == ("Related.",)
    assert len(links[0].contributing_candidate_pairs) == 2
    assert links[0].semantic_score == pytest.approx(0.8)


def test_paths_are_order_independent_and_collisions_have_stable_suffixes() -> None:
    definitions = [
        WikiPageDefinition(
            page_id="aaaaaaaa-0000-0000-0000-000000000000",
            title="Life Cycle Assessment",
            scope="First.",
            work_summary=["First."],
            member_candidate_ids=[UUID(int=1)],
        ),
        WikiPageDefinition(
            page_id="bbbbbbbb-0000-0000-0000-000000000000",
            title="Life-cycle assessment",
            scope="Second.",
            work_summary=["Second."],
            member_candidate_ids=[UUID(int=2)],
        ),
    ]

    forward = assign_target_paths(definitions)
    reverse = assign_target_paths(list(reversed(definitions)))

    assert forward == reverse
    assert forward[definitions[0].page_id].endswith("-aaaaaaaa.md")
    assert forward[definitions[1].page_id].endswith("-bbbbbbbb.md")


def test_merged_page_context_keeps_original_candidates_and_claims() -> None:
    plans = [_plan(1, "Shared"), _plan(2, "Shared")]
    groups, _ = collapse_merge_relations(
        plans,
        [
            WikiRelationResolution(
                candidate_ids=(UUID(int=1), UUID(int=2)),
                decision=WikiRelationDecision.MERGE,
                rationale="Same central concept.",
            ),
        ],
        {},
    )
    definition = WikiPageDefinition(
        page_id=groups[0].page_id,
        title="Shared",
        scope="Shared scope.",
        work_summary=["Preserve both claims."],
        member_candidate_ids=list(groups[0].member_candidate_ids),
    )
    manifest = build_page_manifest([definition], {}, {definition.page_id: 1.0})
    candidates = [_candidate(1), _candidate(2)]

    context = build_page_contexts(manifest, candidates)[0]

    assert context.candidates == candidates
    assert [item.concept.claims[0].statement for item in context.candidates] == [
        "Claim 1.",
        "Claim 2.",
    ]
    assert {item.source.source_id for item in context.candidates} == {
        "paper_1",
        "paper_2",
    }


def test_finalizer_scope_allows_only_existing_planned_concepts() -> None:
    plan = _plan(1, "Subject")
    groups, links = collapse_merge_relations([plan], [], {})
    definition = singleton_page_definition(groups[0], plan)
    assert links == []
    manifest = build_page_manifest([definition], {}, {definition.page_id: 1.0})
    path = manifest.pages[0].target_path.removeprefix("wiki/")
    original = {
        path: b"old",
        "references/paper.md": b"reference",
        "concepts/index.md": b"index",
    }

    validate_finalizer_scope(original, {**original, path: b"new"}, {path})
    with pytest.raises(ValueError, match="outside work package"):
        validate_finalizer_scope(
            original,
            {**original, "references/paper.md": b"changed"},
            {path},
        )
    with pytest.raises(ValueError, match="changed the file set"):
        validate_finalizer_scope(
            original,
            {**original, "concepts/new.md": b"new"},
            {path},
        )


def test_finalizer_packages_bound_pages_and_keep_shared_context() -> None:
    plans = [_plan(number, f"Subject {number}") for number in range(1, 4)]
    groups, _ = collapse_merge_relations(plans, [], {})
    plans_by_id = {plan.candidate_id: plan for plan in plans}
    definitions = [
        singleton_page_definition(group, plans_by_id[group.member_candidate_ids[0]])
        for group in groups
    ]
    manifest = build_page_manifest(
        definitions,
        {},
        {definition.page_id: 1 / 3 for definition in definitions},
    )
    findings = tuple(
        wiki_workflow_module.WikiValidationIssue(
            finding_type=WikiReviewFindingType.LINK,
            target=page.target_path,
            problem="Fix the link.",
            required_change="Integrate the link into body text.",
            basis="Deterministic Wiki bundle validation",
        )
        for page in manifest.pages
    )

    packages = WikiPublicationMixin._finalizer_packages(
        findings,
        manifest,
        batch_size=2,
    )

    assert [len(paths) for paths, _ in packages] == [2, 1]
    first_paths, first_findings = packages[0]
    assert len(first_findings) == 2
    wiki_files = {
        page.target_path.removeprefix("wiki/"): page.page_id.encode()
        for page in manifest.pages
    } | {
        "concepts/index.md": b"index",
        "references/paper.md": b"reference",
    }
    package_wiki = WikiPublicationMixin._finalizer_package_wiki(
        wiki_files,
        first_paths,
    )
    assert set(package_wiki) == {
        *(path.removeprefix("wiki/") for path in first_paths),
        "concepts/index.md",
        "references/paper.md",
    }


def test_finalizer_package_has_a_wall_clock_timeout(monkeypatch) -> None:
    async def stalled_finalizer(
        *_args: object,
        **_kwargs: object,
    ) -> dict[str, bytes]:
        await asyncio.sleep(60)
        return {}

    monkeypatch.setattr(wiki_finalizer_module, "_run_finalizer", stalled_finalizer)
    agent = create_wiki_finalizer_agent(model="test-model")

    with pytest.raises(TimeoutError, match=r"exceeded 0\.01 seconds"):
        run_wiki_finalizer_agent(
            agent,
            {},
            WikiPageManifest(pages=[]),
            {},
            (),
            editable_paths=(),
            timeout_seconds=0.01,
        )


def test_finalizer_session_manifest_does_not_disclose_workspace_files() -> None:
    manifest = wiki_finalizer_module._finalizer_session_manifest()

    assert manifest.entries == {}
    assert manifest.describe(depth=3) == "/workspace\n"


def test_finalizer_workspace_materialization_keeps_session_manifest(
    monkeypatch,
) -> None:
    workspace_manifest = wiki_finalizer_module._finalizer_workspace(
        {"editable_paths": ["wiki/concepts/subject.md"]},
        {"concepts/subject.md": b"content"},
    )
    session = cast("BaseSandboxSession", object())
    calls: list[tuple[object, object, bool]] = []

    async def apply_manifest(
        received_session: object,
        *,
        manifest: object,
        provision_accounts: bool,
    ) -> None:
        calls.append((received_session, manifest, provision_accounts))

    monkeypatch.setattr(
        wiki_finalizer_module.manifest_ops,
        "apply_manifest",
        apply_manifest,
    )

    asyncio.run(
        wiki_finalizer_module._materialize_finalizer_workspace(
            session,
            workspace_manifest,
        ),
    )

    assert calls == [(session, workspace_manifest, False)]
