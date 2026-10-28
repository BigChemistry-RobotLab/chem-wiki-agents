"""Test graph checkpoint persistence and stage-compatible resume behavior."""

import json
from uuid import UUID

import pytest
from pydantic import ValidationError

from wiki_agents.agents.wiki.aggregator import WikiPageGroup, deterministic_page_id
from wiki_agents.agents.wiki.models import (
    WikiCandidateDecision,
    WikiCandidatePlan,
    WikiManifestPage,
    WikiPageContext,
    WikiPageDefinition,
    WikiPageIntent,
    WikiPageJob,
    WikiPageManifest,
    WikiPageRetryState,
    WikiPlan,
    WikiPlanDecision,
    WikiRelationCandidate,
    WikiRelationDecision,
    WikiRelationResolution,
    WikiReviewFinding,
    WikiReviewFindingType,
    WorkPackageGraphEdge,
)
from wiki_agents.concept.models import (
    Claim,
    Concept,
    ConceptCandidate,
    ConceptContext,
    ConceptMatch,
)
from wiki_agents.document.models import EvidenceSpan, SourceRef
from wiki_agents.orchestrator.mapping_checkpoints import (
    WikiCandidatePageRank,
    WikiCorpusCheckpoint,
    WikiFinalGraphCheckpoint,
    WikiMappingCheckpointStore,
    WikiPageAttempt,
    WikiPagePageRank,
    WikiPreliminaryGraphCheckpoint,
    wiki_stage_fingerprint,
)
from wiki_agents.orchestrator.wiki import (
    _relation_input_fingerprint,
    _reusable_page_definitions,
    _reusable_page_outcomes,
    _reusable_relation_resolutions,
)
from wiki_agents.orchestrator.wiki_checkpoint_policy import (
    _candidate_planning_input_fingerprint,
    _same_candidate_plans,
)


def _plan(number: int) -> WikiCandidatePlan:
    """Return one CREATE plan for a stable candidate ID."""
    return WikiCandidatePlan(
        candidate_id=UUID(int=number),
        decision=WikiCandidateDecision.CREATE,
        rationale="Represent this concept.",
        page_intent=WikiPageIntent(
            canonical_subject=f"Concept {number}",
            scope=f"Scope {number}.",
            work_summary=[f"Explain concept {number}."],
        ),
    )


def _candidate(number: int) -> ConceptCandidate:
    """Return one original candidate with a supported claim."""
    return ConceptCandidate(
        candidate_id=UUID(int=number),
        source=SourceRef(
            source_id=f"paper-{number}",
            sha256=f"{number:064x}",
            resource=f"/references/papers/paper-{number}.md",
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


def _relation(first: int, second: int) -> WikiRelationResolution:
    """Return one NONE decision for a sorted candidate pair."""
    return WikiRelationResolution(
        candidate_ids=(UUID(int=first), UUID(int=second)),
        decision=WikiRelationDecision.NONE,
        rationale="No direct relation is required.",
    )


def test_candidate_plan_comparison_ignores_checkpoint_order() -> None:
    first = _plan(1)
    second = _plan(2)

    assert _same_candidate_plans([first, second], [second, first])
    assert not _same_candidate_plans([first], [first, second])


def test_candidate_fingerprint_ignores_volatile_retrieval_matches() -> None:
    candidate = _candidate(1)
    without_match = ConceptContext(candidate=candidate, similar_concepts=[])
    with_match = ConceptContext(
        candidate=candidate,
        similar_concepts=[ConceptMatch(candidate=_candidate(2), score=0.75)],
    )

    assert _candidate_planning_input_fingerprint(
        [without_match],
        {"model": "planner"},
    ) == _candidate_planning_input_fingerprint(
        [with_match],
        {"model": "planner"},
    )


def _definition(page_id: str, *members: int) -> WikiPageDefinition:
    """Return one final page definition."""
    return WikiPageDefinition(
        page_id=page_id,
        title=f"Page {page_id}",
        scope="Supported scope.",
        work_summary=["Explain the concept."],
        member_candidate_ids=[UUID(int=member) for member in members],
    )


def test_corpus_checkpoint_round_trip_keeps_nested_graph_state(tmp_path) -> None:
    first = UUID(int=1)
    second = UUID(int=2)
    edge = WorkPackageGraphEdge(
        candidate_ids=(first, second),
        raw_similarity=0.8,
        similarity=0.8,
        reciprocal=True,
    )
    preliminary = WikiPreliminaryGraphCheckpoint(
        upstream_input_fingerprint="upstream",
        nodes=[first, second],
        edges=[edge],
        content_fingerprint="graph-content",
        pagerank_config_fingerprint="pagerank-config",
        centrality_scores=[
            WikiCandidatePageRank(candidate_id=first, pagerank=0.5),
            WikiCandidatePageRank(candidate_id=second, pagerank=0.5),
        ],
    )
    manifest = WikiPageManifest(
        pages=[
            WikiManifestPage(
                page_id="page-a",
                target_path="wiki/concepts/page-a.md",
                bundle_path="/concepts/page-a.md",
                title="Page A",
                scope="Supported scope.",
                work_summary=["Explain A."],
                member_candidate_ids=[first],
                global_pagerank=1.0,
            ),
        ],
    )
    final = WikiFinalGraphCheckpoint(
        input_fingerprint="final-input",
        content_fingerprint="final-content",
        ranking_fingerprint="final-ranking",
        global_pagerank=[WikiPagePageRank(page_id="page-a", pagerank=1.0)],
        page_manifest=manifest,
    )
    checkpoint = WikiCorpusCheckpoint(
        candidate_planning_fingerprint="candidate-input",
        candidate_plans=[_plan(1), _plan(2)],
        preliminary_graph=preliminary,
        relation_input_fingerprint="relation-input",
        relation_resolutions=[_relation(1, 2)],
        page_definition_input_fingerprint="definition-input",
        page_definitions=[_definition("page-a", 1)],
        final_graph=final,
    )
    store = WikiMappingCheckpointStore(
        tmp_path,
        "stable-key",
    )
    store.page_plans_path().mkdir(parents=True)
    stale_plan = store.page_plans_path() / "stale.json"
    stale_plan.write_text("{}")

    store.save_corpus(checkpoint)

    assert store.load_corpus() == checkpoint.model_copy(
        update={"page_definition_ids": ["page-a"]},
    )
    assert store.corpus_path() == (
        tmp_path / "wiki" / "corpus" / "stable-key" / "state.json"
    )
    state_payload = json.loads(store.corpus_path().read_text())
    assert state_payload["page_definition_ids"] == ["page-a"]
    assert "page_definitions" not in state_payload
    assert store.load_page_plan("page-a") == checkpoint.page_definitions[0]
    assert not stale_plan.exists()


def test_old_checkpoint_versions_are_rejected(tmp_path) -> None:
    store = WikiMappingCheckpointStore(
        tmp_path,
        "stable-key",
    )
    store.corpus_path().write_text(
        json.dumps({"version": 3}),
    )

    with pytest.raises(ValueError, match="Unsupported Wiki corpus checkpoint version"):
        store.load_corpus()


def test_page_outcome_checkpoints_stay_with_their_generation(tmp_path) -> None:
    store = WikiMappingCheckpointStore(
        tmp_path,
        "stable-key",
    )
    accepted = WikiPageAttempt(page_id="page-a", attempts=2, accepted=True)
    failed = WikiPageAttempt(
        page_id="page-b",
        attempts=3,
        accepted=False,
        last_error="Review failed.",
        retry_state=WikiPageRetryState(
            content=b"latest reviewed Markdown\xff",
            raw_output="latest creator response",
            findings=[
                WikiReviewFinding(
                    finding_type=WikiReviewFindingType.FACTUAL,
                    target="wiki/concepts/page-b.md",
                    problem="Review failed.",
                    required_change="Correct the claim.",
                    basis="Source text.",
                ),
            ],
            correction_runs=2,
        ),
    )
    pending_review = WikiPageAttempt(
        page_id="page-c",
        attempts=1,
        accepted=False,
        last_error="Proposal awaits review.",
        retry_state=WikiPageRetryState(
            content=b"valid proposal",
            raw_output="latest creator response",
            findings=[],
            correction_runs=0,
        ),
    )

    path = store.save_page_proposal(
        "writing-input",
        "page-a",
        b"accepted Markdown",
    )
    store.save_page_attempt("writing-input", accepted)
    store.save_page_attempt("writing-input", failed)
    store.save_page_attempt("writing-input", pending_review)

    assert path == (
        tmp_path
        / "wiki"
        / "corpus"
        / "stable-key"
        / "page-proposals"
        / "writing-input"
        / "page-a.md"
    )
    assert store.load_page_proposal("writing-input", "page-a") == (b"accepted Markdown")
    assert store.load_page_attempts("writing-input") == [
        accepted,
        failed,
        pending_review,
    ]
    assert store.load_page_proposal("changed-input", "page-a") is None
    assert store.load_page_attempts("changed-input") == []


def test_preliminary_checkpoint_requires_stable_complete_snapshot() -> None:
    first = UUID(int=1)
    second = UUID(int=2)
    edge = WorkPackageGraphEdge(
        candidate_ids=(first, second),
        raw_similarity=0.7,
        similarity=0.7,
        reciprocal=False,
    )

    with pytest.raises(ValidationError, match="nodes must be sorted"):
        WikiPreliminaryGraphCheckpoint(
            upstream_input_fingerprint="upstream",
            nodes=[second, first],
            edges=[edge],
            content_fingerprint="content",
            pagerank_config_fingerprint="config",
            centrality_scores=[
                WikiCandidatePageRank(candidate_id=second, pagerank=0.5),
                WikiCandidatePageRank(candidate_id=first, pagerank=0.5),
            ],
        )

    with pytest.raises(ValidationError, match="scores must match"):
        WikiPreliminaryGraphCheckpoint(
            upstream_input_fingerprint="upstream",
            nodes=[first, second],
            edges=[edge],
            content_fingerprint="content",
            pagerank_config_fingerprint="config",
            centrality_scores=[
                WikiCandidatePageRank(candidate_id=first, pagerank=1.0),
            ],
        )


def test_relation_resume_uses_pair_membership_after_priority_change() -> None:
    first = _relation(1, 2)
    second = _relation(2, 3)
    checkpoint = WikiCorpusCheckpoint(
        relation_input_fingerprint="semantic-input",
        relation_resolutions=[first, second],
    )

    reused = _reusable_relation_resolutions(
        checkpoint,
        [
            (UUID(int=2), UUID(int=3)),
            (UUID(int=1), UUID(int=2)),
        ],
        "semantic-input",
    )

    assert reused == [second, first]


def test_relation_fingerprint_excludes_priority_order_and_ranking_values() -> None:
    plans = {UUID(int=item): _plan(item) for item in (1, 2, 3)}
    candidates = {UUID(int=item): _candidate(item) for item in (1, 2, 3)}
    first = WikiRelationCandidate(
        candidate_ids=(UUID(int=1), UUID(int=2)),
        similarity_score=0.8,
        ranking_similarity=0.8,
        reciprocal=False,
    )
    second = WikiRelationCandidate(
        candidate_ids=(UUID(int=2), UUID(int=3)),
        similarity_score=0.6,
        ranking_similarity=0.6,
        reciprocal=False,
    )
    reordered_first = first.model_copy(
        update={"ranking_similarity": 0.1, "reciprocal": True},
    )
    reordered_second = second.model_copy(
        update={"ranking_similarity": 0.9, "reciprocal": True},
    )

    original = _relation_input_fingerprint(
        [first, second],
        plans,
        candidates,
        retrieval_fingerprint="retrieval",
        model="relation-model",
        batch_size=24,
    )
    reprioritized = _relation_input_fingerprint(
        [reordered_second, reordered_first],
        plans,
        candidates,
        retrieval_fingerprint="retrieval",
        model="relation-model",
        batch_size=24,
    )

    assert reprioritized == original


def test_page_definition_resume_requires_exact_merge_membership() -> None:
    members = (UUID(int=1), UUID(int=2))
    page_id = deterministic_page_id(members)
    definition = _definition(page_id, 1, 2)
    checkpoint = WikiCorpusCheckpoint(
        page_definition_input_fingerprint="definition-input",
        page_definitions=[definition],
    )
    compatible_group = WikiPageGroup(
        page_id=page_id,
        member_candidate_ids=members,
    )
    changed_group = WikiPageGroup(
        page_id="page-a",
        member_candidate_ids=(UUID(int=1), UUID(int=3)),
    )

    assert _reusable_page_definitions(
        checkpoint,
        [compatible_group],
        "definition-input",
    ) == [definition]
    assert (
        _reusable_page_definitions(
            checkpoint,
            [changed_group],
            "definition-input",
        )
        == []
    )


def test_page_definition_resume_keeps_only_complete_groups() -> None:
    first_members = (UUID(int=1), UUID(int=2))
    second_members = (UUID(int=3), UUID(int=4), UUID(int=5))
    first_definition = _definition(
        deterministic_page_id(first_members),
        1,
        2,
    )
    partial_second_definition = _definition(
        deterministic_page_id(second_members[:2]),
        3,
        4,
    )
    checkpoint = WikiCorpusCheckpoint(
        page_definition_input_fingerprint="definition-input",
        page_definitions=[first_definition, partial_second_definition],
    )
    groups = [
        WikiPageGroup(
            page_id=deterministic_page_id(first_members),
            member_candidate_ids=first_members,
        ),
        WikiPageGroup(
            page_id=deterministic_page_id(second_members),
            member_candidate_ids=second_members,
        ),
    ]

    assert _reusable_page_definitions(
        checkpoint,
        groups,
        "definition-input",
    ) == [first_definition]


def test_stage_fingerprint_is_stable_for_mapping_order() -> None:
    assert wiki_stage_fingerprint({"a": 1, "b": 2}) == wiki_stage_fingerprint(
        {"b": 2, "a": 1},
    )


def test_accepted_page_proposal_reuse_requires_writing_fingerprint() -> None:
    plan = WikiPlan(
        decision=WikiPlanDecision.CREATE,
        rationale="Create the final page.",
        target_path="wiki/concepts/page-a.md",
        title="Page A",
        work_summary=["Explain A."],
    )
    context = WikiPageContext(
        page_job=WikiPageJob(
            page_id="page-a",
            plan=plan,
            member_candidate_ids=[UUID(int=1)],
        ),
        candidates=[_candidate(1)],
    )
    checkpoint = WikiCorpusCheckpoint(
        page_generation_fingerprint="writing-input",
        page_attempts=[
            WikiPageAttempt(page_id="page-a", attempts=2, accepted=True),
        ],
        accepted_page_ids=["page-a"],
    )

    reused = _reusable_page_outcomes(
        checkpoint,
        [context],
        "writing-input",
        lambda _page_id: b"accepted Markdown",
    )

    assert reused["page-a"].correction_runs == 1
    assert reused["page-a"].creation is not None
    assert reused["page-a"].creation.changes[0].content == b"accepted Markdown"
    assert (
        _reusable_page_outcomes(
            checkpoint,
            [context],
            "changed-writing-input",
            lambda _page_id: b"stale Markdown",
        )
        == {}
    )
