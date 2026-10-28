"""Test deterministic ranking for sparse work-package and Wiki graphs."""

from __future__ import annotations

import math
from uuid import UUID

import networkx as nx
import pytest
from pydantic import ValidationError

from wiki_agents.agents.wiki.graph_ranking import (
    WikiGraphRankingError,
    _run_pagerank,
    build_wiki_page_graph,
    build_work_package_similarity_graph,
    prioritize_relation_candidates,
    rank_global_pages,
    rank_links_for_page,
    rank_work_package_centrality,
    restore_work_package_similarity_graph,
    work_package_graph_snapshot,
)
from wiki_agents.agents.wiki.models import (
    WikiCandidateDecision,
    WikiCandidatePairContribution,
    WikiCandidatePlan,
    WikiGraphRankingConfig,
    WikiPageDefinition,
    WikiPageIntent,
    WikiPageRelation,
    WorkPackageNeighbor,
)


def _candidate_id(value: int) -> UUID:
    return UUID(int=value)


def _create_plan(value: int) -> WikiCandidatePlan:
    candidate_id = _candidate_id(value)
    return WikiCandidatePlan(
        candidate_id=candidate_id,
        decision=WikiCandidateDecision.CREATE,
        rationale="The candidate has supported work.",
        page_intent=WikiPageIntent(
            canonical_subject=f"Subject {value}",
            scope=f"Scope {value}",
            work_summary=[f"Work {value}"],
        ),
    )


def _definition(page_id: str, candidate_value: int) -> WikiPageDefinition:
    return WikiPageDefinition(
        page_id=page_id,
        title=f"Page {page_id}",
        scope=f"Scope {page_id}",
        work_summary=[f"Work {page_id}"],
        member_candidate_ids=[_candidate_id(candidate_value)],
    )


def _relation(  # ruff: ignore[too-many-arguments] - The helper names both directed relation endpoints.
    source_page_id: str,
    target_page_id: str,
    source_candidate_value: int,
    target_candidate_value: int,
    *,
    semantic_score: float = 0.5,
    relationship: str = "supports",
) -> WikiPageRelation:
    return WikiPageRelation(
        source_page_id=source_page_id,
        target_page_id=target_page_id,
        relationships=(relationship,),
        semantic_score=semantic_score,
        contributing_candidate_pairs=(
            WikiCandidatePairContribution(
                source_candidate_id=_candidate_id(source_candidate_value),
                target_candidate_id=_candidate_id(target_candidate_value),
                similarity=semantic_score,
            ),
        ),
    )


def test_work_package_centrality_is_deterministic_and_favors_center() -> None:
    plans = [_create_plan(value) for value in range(1, 5)]
    a, b, c, d = (plan.candidate_id for plan in plans)
    neighbors = [
        WorkPackageNeighbor(
            source_candidate_id=a,
            target_candidate_id=b,
            similarity_score=0.8,
        ),
        WorkPackageNeighbor(
            source_candidate_id=b,
            target_candidate_id=a,
            similarity_score=0.7,
        ),
        WorkPackageNeighbor(
            source_candidate_id=b,
            target_candidate_id=c,
            similarity_score=0.8,
        ),
        WorkPackageNeighbor(
            source_candidate_id=b,
            target_candidate_id=d,
            similarity_score=0.8,
        ),
    ]

    graph = build_work_package_similarity_graph(plans, neighbors)
    reversed_graph = build_work_package_similarity_graph(
        list(reversed(plans)),
        list(reversed(neighbors)),
    )
    scores = rank_work_package_centrality(graph, alpha=0.85)
    reversed_scores = rank_work_package_centrality(reversed_graph, alpha=0.85)

    assert scores[b] > scores[a]
    assert scores[b] > scores[c]
    assert scores[b] > scores[d]
    assert scores == pytest.approx(reversed_scores, abs=1e-15)
    assert [
        item.candidate_ids for item in prioritize_relation_candidates(graph, scores)
    ] == [
        item.candidate_ids
        for item in prioritize_relation_candidates(reversed_graph, reversed_scores)
    ]
    assert graph.edges[a, b]["reciprocal"] is True
    assert graph.edges[b, c]["reciprocal"] is False


def test_work_package_graph_deduplicates_and_clamps_observations() -> None:
    plans = [_create_plan(value) for value in range(1, 5)]
    a, b, c, isolated = (plan.candidate_id for plan in plans)
    neighbors = [
        WorkPackageNeighbor(
            source_candidate_id=a,
            target_candidate_id=b,
            similarity_score=0.3,
        ),
        WorkPackageNeighbor(
            source_candidate_id=a,
            target_candidate_id=b,
            similarity_score=0.9,
        ),
        WorkPackageNeighbor(
            source_candidate_id=b,
            target_candidate_id=a,
            similarity_score=0.4,
        ),
        WorkPackageNeighbor(
            source_candidate_id=b,
            target_candidate_id=c,
            similarity_score=-0.25,
        ),
    ]

    graph = build_work_package_similarity_graph(plans, neighbors)
    pairs = prioritize_relation_candidates(
        graph,
        dict.fromkeys(graph.nodes, 1.0),
    )

    assert set(graph.nodes) == {a, b, c, isolated}
    assert graph.degree[isolated] == 0
    assert graph.edges[a, b]["raw_similarity"] == pytest.approx(0.9)
    assert graph.edges[a, b]["similarity"] == pytest.approx(0.9)
    assert graph.edges[a, b]["reciprocal"] is True
    assert graph.edges[b, c]["raw_similarity"] == pytest.approx(-0.25)
    assert graph.edges[b, c]["similarity"] == pytest.approx(0.0)
    negative_pair = next(item for item in pairs if item.candidate_ids == (b, c))
    assert negative_pair.similarity_score == pytest.approx(-0.25)
    assert negative_pair.ranking_similarity == pytest.approx(0.0)


def test_work_package_graph_snapshot_round_trip_is_canonical() -> None:
    plans = [_create_plan(value) for value in range(1, 4)]
    graph = build_work_package_similarity_graph(
        list(reversed(plans)),
        [
            WorkPackageNeighbor(
                source_candidate_id=_candidate_id(2),
                target_candidate_id=_candidate_id(1),
                similarity_score=0.7,
            ),
        ],
    )

    nodes, edges = work_package_graph_snapshot(graph)
    restored = restore_work_package_similarity_graph(
        tuple(reversed(nodes)),
        tuple(reversed(edges)),
    )

    assert nodes == tuple(sorted(nodes, key=str))
    assert work_package_graph_snapshot(restored) == (nodes, edges)


def test_work_package_observation_rejects_nonfinite_score() -> None:
    with pytest.raises(ValidationError, match="finite"):
        WorkPackageNeighbor(
            source_candidate_id=_candidate_id(1),
            target_candidate_id=_candidate_id(2),
            similarity_score=math.inf,
        )


def test_work_package_graph_rejects_unknown_candidate() -> None:
    with pytest.raises(ValueError, match="unknown CREATE candidate"):
        build_work_package_similarity_graph(
            [_create_plan(1)],
            [
                WorkPackageNeighbor(
                    source_candidate_id=_candidate_id(1),
                    target_candidate_id=_candidate_id(2),
                    similarity_score=0.8,
                ),
            ],
        )


def test_relation_pair_priority_uses_exact_formula_and_stable_ties() -> None:
    plans = [_create_plan(value) for value in range(1, 5)]
    a, b, c, d = (plan.candidate_id for plan in plans)
    graph = build_work_package_similarity_graph(
        plans,
        [
            WorkPackageNeighbor(
                source_candidate_id=a,
                target_candidate_id=b,
                similarity_score=0.2,
            ),
            WorkPackageNeighbor(
                source_candidate_id=a,
                target_candidate_id=c,
                similarity_score=0.1,
            ),
            WorkPackageNeighbor(
                source_candidate_id=b,
                target_candidate_id=d,
                similarity_score=0.9,
            ),
        ],
    )

    prioritized = prioritize_relation_candidates(
        graph,
        {a: 0.9, b: 0.18, c: 0.72, d: 0.36},
    )

    assert [item.candidate_ids for item in prioritized] == [
        (a, c),  # 1.0 + 0.5 * 0.8 + 0.1 = 1.5
        (b, d),  # 0.4 + 0.5 * 0.2 + 0.9 = 1.4
        (a, b),  # 1.0 + 0.5 * 0.2 + 0.2 = 1.3
    ]
    assert len({item.candidate_ids for item in prioritized}) == 3
    assert prioritized[-1].model_dump() == {
        "candidate_ids": (a, b),
        "similarity_score": 0.2,
        "ranking_similarity": 0.2,
        "reciprocal": False,
    }

    tied = prioritize_relation_candidates(
        graph,
        dict.fromkeys(graph.nodes, 1.0),
    )
    assert [item.candidate_ids for item in tied] == [(b, d), (a, b), (a, c)]


def test_relation_pair_priority_rejects_incomplete_centrality() -> None:
    plans = [_create_plan(1), _create_plan(2)]
    graph = build_work_package_similarity_graph(
        plans,
        [
            WorkPackageNeighbor(
                source_candidate_id=_candidate_id(1),
                target_candidate_id=_candidate_id(2),
                similarity_score=0.5,
            ),
        ],
    )
    with pytest.raises(ValueError, match="must match"):
        prioritize_relation_candidates(graph, {_candidate_id(1): 1.0})


def test_global_pagerank_favors_the_shared_target_over_source_pages() -> None:
    definitions = [
        _definition("A", 1),
        _definition("B", 2),
        _definition("C", 3),
        _definition("D", 4),
        _definition("E", 5),
    ]
    graph = build_wiki_page_graph(
        definitions,
        [
            _relation("A", "B", 1, 2),
            _relation("C", "B", 3, 2),
            _relation("D", "B", 4, 2),
            _relation("B", "E", 2, 5),
        ],
    )

    scores = rank_global_pages(graph, alpha=0.85)

    assert scores["B"] > scores["A"]
    assert scores["B"] > scores["C"]
    assert scores["B"] > scores["D"]


def test_personalized_pagerank_stays_in_the_seed_component() -> None:
    graph = nx.DiGraph([("A", "B"), ("B", "C"), ("D", "E"), ("E", "F")])
    personalization = {
        node: 1.0 if node == "A" else 0.0 for node in sorted(graph.nodes)
    }

    scores = _run_pagerank(
        graph,
        alpha=0.85,
        weight=None,
        personalization=personalization,
        dangling=personalization,
    )

    assert scores["B"] > scores["E"]
    assert scores["B"] > scores["F"]
    assert scores["C"] > scores["E"]
    assert scores["C"] > scores["F"]


def test_page_without_outgoing_links_skips_personalized_pagerank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = build_wiki_page_graph([_definition("A", 1)], [])
    calls = 0

    def fail_if_called(*args: object, **kwargs: object) -> dict[str, float]:
        nonlocal calls
        calls += 1
        return {}

    monkeypatch.setattr(nx, "pagerank", fail_if_called)

    assert (
        rank_links_for_page(
            graph,
            "A",
            global_scores={"A": 1.0},
            config=WikiGraphRankingConfig(),
        )
        == []
    )
    assert calls == 0


def test_link_ranking_never_authorizes_a_non_edge() -> None:
    definitions = [_definition("A", 1), _definition("B", 2), _definition("C", 3)]
    graph = build_wiki_page_graph(
        definitions,
        [_relation("A", "B", 1, 2, semantic_score=0.3)],
    )

    ranked = rank_links_for_page(
        graph,
        "A",
        global_scores={"A": 0.01, "B": 0.01, "C": 0.98},
        config=WikiGraphRankingConfig(),
    )

    assert [item.target_page_id for item in ranked] == ["B"]
    assert ranked[0].selected_for_page is True


def test_embedding_only_nodes_do_not_create_final_page_edges() -> None:
    graph = build_wiki_page_graph(
        [_definition("A", 1), _definition("B", 2)],
        [],
    )
    assert sorted(graph.nodes) == ["A", "B"]
    assert list(graph.edges) == []


def test_strong_semantic_link_can_rank_above_a_global_hub() -> None:
    definitions = [
        _definition("A", 1),
        _definition("B", 2),
        _definition("HUB", 3),
        _definition("X", 4),
    ]
    graph = build_wiki_page_graph(
        definitions,
        [
            _relation("A", "B", 1, 2, semantic_score=0.95),
            _relation("A", "HUB", 1, 3, semantic_score=0.20),
            _relation("X", "HUB", 4, 3, semantic_score=0.80),
        ],
    )

    ranked = rank_links_for_page(
        graph,
        "A",
        global_scores={"A": 0.05, "B": 0.05, "HUB": 0.85, "X": 0.05},
        config=WikiGraphRankingConfig(),
    )

    assert [item.target_page_id for item in ranked] == ["B", "HUB"]
    assert ranked[0].ranking_score > ranked[1].ranking_score


def _star_page_graph(scores: list[float]) -> nx.DiGraph:
    definitions = [_definition("A", 1)]
    relations = []
    for index, score in enumerate(scores, start=2):
        target = f"T{index:02d}"
        definitions.append(_definition(target, index))
        relations.append(_relation("A", target, 1, index, semantic_score=score))
    return build_wiki_page_graph(definitions, relations)


def _uniform_global_scores(graph: nx.DiGraph) -> dict[str, float]:
    score = 1.0 / len(graph)
    return dict.fromkeys(graph.nodes, score)


def test_link_cap_selects_the_eight_highest_valid_links() -> None:
    graph = _star_page_graph([value / 100 for value in range(10, 22)])

    ranked = rank_links_for_page(
        graph,
        "A",
        global_scores=_uniform_global_scores(graph),
        config=WikiGraphRankingConfig(),
    )

    selected = [item for item in ranked if item.selected_for_page]
    assert len(selected) == 8
    assert [item.semantic_score for item in selected] == sorted(
        (item.semantic_score for item in selected),
        reverse=True,
    )
    assert {item.target_page_id for item in selected} == {
        "T06",
        "T07",
        "T08",
        "T09",
        "T10",
        "T11",
        "T12",
        "T13",
    }


@pytest.mark.parametrize("count", [0, 3])
def test_link_selection_keeps_every_valid_link_below_cap(count: int) -> None:
    graph = _star_page_graph([0.1] * count)

    ranked = rank_links_for_page(
        graph,
        "A",
        global_scores=_uniform_global_scores(graph),
        config=WikiGraphRankingConfig(),
    )

    assert len(ranked) == count
    assert all(item.selected_for_page for item in ranked)


def test_threshold_links_survive_cap_including_exact_threshold() -> None:
    graph = _star_page_graph([0.99, 0.98, 0.92, 0.91])

    ranked = rank_links_for_page(
        graph,
        "A",
        global_scores=_uniform_global_scores(graph),
        config=WikiGraphRankingConfig(max_outgoing_links=2),
    )

    selected = [item for item in ranked if item.selected_for_page]
    assert [item.semantic_score for item in selected] == [0.99, 0.98, 0.92]
    assert ranked[-1].semantic_score == pytest.approx(0.91)
    assert ranked[-1].selected_for_page is False


def test_more_than_eight_threshold_links_all_survive() -> None:
    graph = _star_page_graph([0.92] * 9)

    ranked = rank_links_for_page(
        graph,
        "A",
        global_scores=_uniform_global_scores(graph),
        config=WikiGraphRankingConfig(),
    )

    assert len(ranked) == 9
    assert all(item.selected_for_page for item in ranked)
    assert [item.target_page_id for item in ranked] == sorted(
        item.target_page_id for item in ranked
    )


def test_equal_final_link_scores_use_target_page_id() -> None:
    graph = _star_page_graph([0.2, 0.2, 0.2])

    ranked = rank_links_for_page(
        graph,
        "A",
        global_scores=_uniform_global_scores(graph),
        config=WikiGraphRankingConfig(max_outgoing_links=2),
    )

    assert [item.target_page_id for item in ranked] == ["T02", "T03", "T04"]
    assert [item.selected_for_page for item in ranked] == [True, True, False]


def test_page_graph_rejects_duplicate_or_unknown_relations() -> None:
    definitions = [_definition("A", 1), _definition("B", 2)]
    relation = _relation("A", "B", 1, 2)
    with pytest.raises(ValueError, match="repeat"):
        build_wiki_page_graph(definitions, [relation, relation])
    with pytest.raises(ValueError, match="unknown"):
        build_wiki_page_graph(
            definitions,
            [_relation("A", "C", 1, 3)],
        )


def test_pagerank_nonconvergence_has_clear_graph_stage_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: object, **kwargs: object) -> dict[str, float]:
        raise nx.PowerIterationFailedConvergence(1000)

    monkeypatch.setattr(nx, "pagerank", fail)

    with pytest.raises(WikiGraphRankingError, match="did not converge"):
        rank_global_pages(nx.DiGraph([("A", "B")]), alpha=0.85)
