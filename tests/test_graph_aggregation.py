"""Test graph metadata while candidate relations collapse to Wiki pages."""

from __future__ import annotations

from uuid import UUID

import pytest

from wiki_agents.agents.wiki.aggregator import collapse_merge_relations
from wiki_agents.agents.wiki.models import (
    WikiCandidateDecision,
    WikiCandidatePlan,
    WikiDirectedRelation,
    WikiPageIntent,
    WikiRelationCandidate,
    WikiRelationDecision,
    WikiRelationResolution,
)


def _candidate_id(value: int) -> UUID:
    return UUID(int=value)


def _plan(value: int) -> WikiCandidatePlan:
    return WikiCandidatePlan(
        candidate_id=_candidate_id(value),
        decision=WikiCandidateDecision.CREATE,
        rationale="The candidate has supported work.",
        page_intent=WikiPageIntent(
            canonical_subject=f"Subject {value}",
            scope=f"Scope {value}",
            work_summary=[f"Work {value}"],
        ),
    )


def _merge_resolution(first: int, second: int) -> WikiRelationResolution:
    return WikiRelationResolution(
        candidate_ids=(_candidate_id(first), _candidate_id(second)),
        decision=WikiRelationDecision.MERGE,
        rationale="The work packages describe one page.",
    )


def _link_resolution(
    source: int,
    target: int,
    relationship: str,
) -> WikiRelationResolution:
    return WikiRelationResolution(
        candidate_ids=(_candidate_id(source), _candidate_id(target)),
        decision=WikiRelationDecision.LINK,
        rationale="The pages need a directed concept link.",
        links=[
            WikiDirectedRelation(
                source_candidate_id=_candidate_id(source),
                target_candidate_id=_candidate_id(target),
                relationship=relationship,
            ),
        ],
    )


def _relation_candidate(
    first: int,
    second: int,
    similarity: float,
) -> WikiRelationCandidate:
    return WikiRelationCandidate(
        candidate_ids=(_candidate_id(first), _candidate_id(second)),
        similarity_score=similarity,
        ranking_similarity=similarity,
    )


def test_merge_collapse_combines_link_provenance_and_uses_maximum_score() -> None:
    plans = [_plan(1), _plan(2), _plan(3)]
    merge = _merge_resolution(1, 2)
    first_link = _link_resolution(1, 3, "Zeta relationship.")
    second_link = _link_resolution(2, 3, "Alpha relationship.")
    first_candidate = _relation_candidate(1, 3, 0.6)
    second_candidate = _relation_candidate(2, 3, 0.8)
    metadata = {
        first_candidate.candidate_ids: first_candidate,
        second_candidate.candidate_ids: second_candidate,
    }

    groups, page_relations = collapse_merge_relations(
        plans,
        [second_link, merge, first_link],
        metadata,
    )

    merged_group = next(
        group for group in groups if len(group.member_candidate_ids) == 2
    )
    singleton_group = next(
        group for group in groups if len(group.member_candidate_ids) == 1
    )
    assert merged_group.member_candidate_ids == (_candidate_id(1), _candidate_id(2))
    assert singleton_group.member_candidate_ids == (_candidate_id(3),)
    assert len(page_relations) == 1
    relation = page_relations[0]
    assert relation.source_page_id == merged_group.page_id
    assert relation.target_page_id == singleton_group.page_id
    assert relation.relationships == ("Alpha relationship.", "Zeta relationship.")
    assert relation.semantic_score == pytest.approx(0.8)
    assert [
        contribution.source_candidate_id
        for contribution in relation.contributing_candidate_pairs
    ] == [_candidate_id(1), _candidate_id(2)]
    assert [
        contribution.target_candidate_id
        for contribution in relation.contributing_candidate_pairs
    ] == [_candidate_id(3), _candidate_id(3)]
    assert [
        contribution.similarity
        for contribution in relation.contributing_candidate_pairs
    ] == pytest.approx([0.6, 0.8])


def test_link_internal_to_transitive_merge_component_is_removed() -> None:
    plans = [_plan(1), _plan(2), _plan(3)]
    internal_link = _link_resolution(1, 3, "Internal relationship.")
    internal_candidate = _relation_candidate(1, 3, 0.9)

    groups, page_relations = collapse_merge_relations(
        plans,
        [_merge_resolution(1, 2), _merge_resolution(2, 3), internal_link],
        {internal_candidate.candidate_ids: internal_candidate},
    )

    assert len(groups) == 1
    assert groups[0].member_candidate_ids == (
        _candidate_id(1),
        _candidate_id(2),
        _candidate_id(3),
    )
    assert page_relations == []


def test_link_without_retrieved_pair_similarity_is_an_invariant_error() -> None:
    link = _link_resolution(1, 2, "Missing metadata.")

    with pytest.raises(ValueError, match="no retrieved-pair similarity"):
        collapse_merge_relations([_plan(1), _plan(2)], [link], {})
