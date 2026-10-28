"""Rank sparse work-package graphs and accepted Wiki page links."""

from __future__ import annotations

import math
from collections.abc import Hashable
from itertools import starmap
from typing import TYPE_CHECKING, cast
from uuid import UUID

import networkx as nx

from wiki_agents.agents.wiki.models import (
    RankedWikiLink,
    WikiCandidateDecision,
    WikiGraphRankingConfig,
    WikiPageRelation,
    WikiRelationCandidate,
    WorkPackageGraphEdge,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from wiki_agents.agents.wiki.models import (
        WikiCandidatePlan,
        WikiPageDefinition,
        WorkPackageNeighbor,
    )

_PAGERANK_MAX_ITERATIONS = 1000
_PAGERANK_TOLERANCE = 1e-12
GRAPH_RANKING_ALGORITHM_VERSION = "1"


class WikiGraphRankingError(RuntimeError):
    """Report a failure in a deterministic graph-ranking stage."""


def _stable_id(value: object) -> str:
    """Return the stable text key for one graph node."""
    return str(value)


def _stable_pair[NodeT](first: NodeT, second: NodeT) -> tuple[NodeT, NodeT]:
    """Return two graph nodes in stable text order."""
    if _stable_id(first) <= _stable_id(second):
        return first, second
    return second, first


def _ordered_graph(graph: nx.Graph) -> nx.Graph:
    """Copy a graph with stable node and edge insertion order."""
    ordered: nx.Graph
    if graph.is_directed():
        ordered = nx.DiGraph()
        edges = sorted(
            graph.edges(data=True),
            key=lambda edge: (_stable_id(edge[0]), _stable_id(edge[1])),
        )
    else:
        ordered = nx.Graph()
        edges = sorted(
            graph.edges(data=True),
            key=lambda edge: tuple(
                _stable_id(item) for item in _stable_pair(edge[0], edge[1])
            ),
        )
    ordered.add_nodes_from(sorted(graph.nodes, key=_stable_id))
    ordered.add_edges_from(
        (source, target, dict(data)) for source, target, data in edges
    )
    return ordered


def _run_pagerank[NodeT: Hashable](
    graph: nx.Graph,
    *,
    alpha: float,
    weight: str | None,
    personalization: Mapping[NodeT, float] | None = None,
    dangling: Mapping[NodeT, float] | None = None,
) -> dict[NodeT, float]:
    """Run PageRank with stable insertion order and fixed convergence settings."""
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("PageRank alpha must be finite and between zero and one")
    ordered = _ordered_graph(graph)
    try:
        scores = cast(
            "dict[NodeT, float]",
            nx.pagerank(
                ordered,
                alpha=alpha,
                personalization=personalization,
                dangling=dangling,
                weight=weight,
                max_iter=_PAGERANK_MAX_ITERATIONS,
                tol=_PAGERANK_TOLERANCE,
            ),
        )
    except nx.PowerIterationFailedConvergence as error:
        raise WikiGraphRankingError(
            "the graph PageRank calculation did not converge after 1000 iterations",
        ) from error
    return {node: float(scores[node]) for node in sorted(scores, key=_stable_id)}


def build_work_package_similarity_graph(
    plans: Sequence[WikiCandidatePlan],
    neighbors: Sequence[WorkPackageNeighbor],
) -> nx.Graph:
    """Build a stable sparse graph from directed top-K observations.

    Args:
        plans: Candidate plans for the current corpus build.
        neighbors: Directed top-K similarity observations.

    Returns:
        An undirected graph with all CREATE candidates and retrieved pairs.

    Raises:
        ValueError: If a plan ID repeats or an observation uses an unknown ID.

    """
    create_ids = [
        plan.candidate_id
        for plan in plans
        if plan.decision is WikiCandidateDecision.CREATE
    ]
    if len(create_ids) != len(set(create_ids)):
        raise ValueError("CREATE plans cannot repeat a candidate ID")
    known_ids = set(create_ids)

    directional_scores: dict[tuple[UUID, UUID], float] = {}
    for neighbor in neighbors:
        direction = (neighbor.source_candidate_id, neighbor.target_candidate_id)
        if direction[0] not in known_ids or direction[1] not in known_ids:
            raise ValueError(
                "a work-package neighbor references an unknown CREATE candidate",
            )
        directional_scores[direction] = max(
            directional_scores.get(direction, float("-inf")),
            float(neighbor.similarity_score),
        )

    pair_directions: dict[tuple[UUID, UUID], dict[tuple[UUID, UUID], float]] = {}
    for direction, raw_similarity in directional_scores.items():
        pair = _stable_pair(*direction)
        pair_directions.setdefault(pair, {})[direction] = raw_similarity

    graph = nx.Graph()
    graph.add_nodes_from(sorted(create_ids, key=_stable_id))
    for pair in sorted(
        pair_directions,
        key=lambda item: (_stable_id(item[0]), _stable_id(item[1])),
    ):
        observations = pair_directions[pair]
        raw_similarity = max(observations.values())
        edge = WorkPackageGraphEdge(
            candidate_ids=pair,
            raw_similarity=raw_similarity,
            similarity=min(1.0, max(0.0, raw_similarity)),
            reciprocal=(pair in observations and (pair[1], pair[0]) in observations),
        )
        graph.add_edge(
            pair[0],
            pair[1],
            candidate_ids=edge.candidate_ids,
            raw_similarity=float(edge.raw_similarity),
            similarity=edge.similarity,
            reciprocal=edge.reciprocal,
        )
    return graph


def rank_work_package_centrality(
    graph: nx.Graph,
    *,
    alpha: float,
) -> dict[UUID, float]:
    """Return weighted structural centrality for each work-package node."""
    return cast(
        "dict[UUID, float]",
        _run_pagerank(graph, alpha=alpha, weight="similarity"),
    )


def _work_package_edge(
    first: UUID,
    second: UUID,
    data: Mapping[str, object],
) -> WorkPackageGraphEdge:
    """Validate and return one work-package edge snapshot."""
    pair = _stable_pair(first, second)
    payload = dict(data)
    payload["candidate_ids"] = pair
    return WorkPackageGraphEdge.model_validate(payload)


def work_package_graph_snapshot(
    graph: nx.Graph,
) -> tuple[tuple[UUID, ...], tuple[WorkPackageGraphEdge, ...]]:
    """Return a canonical checkpoint snapshot of a work-package graph."""
    if graph.is_directed():
        raise ValueError("a work-package similarity graph must be undirected")
    if any(not isinstance(node, UUID) for node in graph.nodes):
        raise ValueError("work-package graph nodes must be candidate UUIDs")
    nodes = tuple(sorted(cast("list[UUID]", list(graph.nodes)), key=_stable_id))
    edges = tuple(
        sorted(
            (starmap(_work_package_edge, graph.edges(data=True))),
            key=lambda edge: (
                _stable_id(edge.candidate_ids[0]),
                _stable_id(edge.candidate_ids[1]),
            ),
        ),
    )
    return nodes, edges


def restore_work_package_similarity_graph(
    nodes: Sequence[UUID],
    edges: Sequence[WorkPackageGraphEdge],
) -> nx.Graph:
    """Restore a canonical work-package graph from checkpoint metadata."""
    if len(nodes) != len(set(nodes)):
        raise ValueError("a work-package graph snapshot cannot repeat a node")
    known_nodes = set(nodes)
    edge_pairs = [edge.candidate_ids for edge in edges]
    if len(edge_pairs) != len(set(edge_pairs)):
        raise ValueError("a work-package graph snapshot cannot repeat an edge")
    if any(set(pair) - known_nodes for pair in edge_pairs):
        raise ValueError("a work-package graph edge references an unknown node")

    graph = nx.Graph()
    graph.add_nodes_from(sorted(nodes, key=_stable_id))
    for edge in sorted(
        edges,
        key=lambda item: (
            _stable_id(item.candidate_ids[0]),
            _stable_id(item.candidate_ids[1]),
        ),
    ):
        graph.add_edge(
            edge.candidate_ids[0],
            edge.candidate_ids[1],
            candidate_ids=edge.candidate_ids,
            raw_similarity=float(edge.raw_similarity),
            similarity=edge.similarity,
            reciprocal=edge.reciprocal,
        )
    return graph


def prioritize_relation_candidates(
    graph: nx.Graph,
    centrality: Mapping[UUID, float],
) -> list[WikiRelationCandidate]:
    """Return every sparse relation pair in deterministic priority order."""
    graph_nodes = set(graph.nodes)
    if set(centrality) != graph_nodes:
        raise ValueError("centrality scores must match the work-package graph nodes")
    if any(not math.isfinite(score) or score < 0.0 for score in centrality.values()):
        raise ValueError("centrality scores must be finite and nonnegative")

    maximum = max(centrality.values(), default=0.0)
    normalized = {
        candidate_id: score / maximum if maximum > 0.0 else 0.0
        for candidate_id, score in centrality.items()
    }
    prioritized: list[tuple[float, WikiRelationCandidate]] = []
    for first, second, data in graph.edges(data=True):
        edge = _work_package_edge(first, second, data)
        first_score = normalized[first]
        second_score = normalized[second]
        priority = (
            max(first_score, second_score)
            + (0.5 * min(first_score, second_score))
            + edge.similarity
        )
        prioritized.append(
            (
                priority,
                WikiRelationCandidate(
                    candidate_ids=edge.candidate_ids,
                    similarity_score=edge.raw_similarity,
                    ranking_similarity=edge.similarity,
                    reciprocal=edge.reciprocal,
                ),
            ),
        )
    prioritized.sort(
        key=lambda item: (
            -item[0],
            _stable_id(item[1].candidate_ids[0]),
            _stable_id(item[1].candidate_ids[1]),
        ),
    )
    return [candidate for _, candidate in prioritized]


def build_wiki_page_graph(
    definitions: Sequence[WikiPageDefinition],
    relations: Sequence[WikiPageRelation],
) -> nx.DiGraph:
    """Build the accepted directed page graph before manifest ranking."""
    page_ids = [definition.page_id for definition in definitions]
    if len(page_ids) != len(set(page_ids)):
        raise ValueError("page definitions cannot repeat a page ID")
    known_ids = set(page_ids)

    graph = nx.DiGraph()
    graph.add_nodes_from(sorted(page_ids))
    relation_pairs: set[tuple[str, str]] = set()
    for relation in sorted(
        relations,
        key=lambda item: (item.source_page_id, item.target_page_id),
    ):
        pair = (relation.source_page_id, relation.target_page_id)
        if pair[0] not in known_ids or pair[1] not in known_ids:
            raise ValueError("a page relation references an unknown page")
        if pair in relation_pairs:
            raise ValueError("page relations cannot repeat a directed page pair")
        relation_pairs.add(pair)
        graph.add_edge(
            pair[0],
            pair[1],
            relationships=relation.relationships,
            semantic_score=relation.semantic_score,
            contributing_candidate_pairs=relation.contributing_candidate_pairs,
        )
    return graph


def rank_global_pages(
    graph: nx.DiGraph,
    *,
    alpha: float,
) -> dict[str, float]:
    """Return unweighted structural centrality for every final Wiki page."""
    return cast("dict[str, float]", _run_pagerank(graph, alpha=alpha, weight=None))


def _validate_global_scores(
    graph: nx.DiGraph,
    global_scores: Mapping[str, float],
) -> None:
    """Require one finite normalized global score for every graph page."""
    if set(global_scores) != set(graph.nodes):
        raise ValueError("global PageRank scores must match the page graph nodes")
    if any(
        not math.isfinite(score) or not 0.0 <= score <= 1.0
        for score in global_scores.values()
    ):
        raise ValueError("global PageRank scores must be finite and normalized")


def rank_links_for_page(  # ruff: ignore[too-many-locals] - Link scoring keeps the three normalized components visible.
    graph: nx.DiGraph,
    source_page_id: str,
    *,
    global_scores: Mapping[str, float],
    config: WikiGraphRankingConfig,
) -> list[RankedWikiLink]:
    """Rank and select only accepted outgoing links for one page."""
    if source_page_id not in graph:
        raise ValueError("the source page is absent from the page graph")
    _validate_global_scores(graph, global_scores)
    eligible_targets = sorted(graph.successors(source_page_id))
    if not eligible_targets:
        return []

    ordered = _ordered_graph(graph)
    personalization = {
        page_id: 1.0 if page_id == source_page_id else 0.0
        for page_id in sorted(ordered.nodes)
    }
    personalized_scores = cast(
        "dict[str, float]",
        _run_pagerank(
            ordered,
            alpha=config.alpha,
            personalization=personalization,
            dangling=personalization,
            weight=None,
        ),
    )
    global_maximum = max(global_scores.values(), default=0.0)
    personalized_maximum = max(
        (personalized_scores[target] for target in eligible_targets),
        default=0.0,
    )

    unselected: list[RankedWikiLink] = []
    for target_page_id in eligible_targets:
        data = graph.edges[source_page_id, target_page_id]
        relation = WikiPageRelation.model_validate(
            {
                **data,
                "source_page_id": source_page_id,
                "target_page_id": target_page_id,
            },
        )
        semantic_score = float(relation.semantic_score)
        global_component = (
            global_scores[target_page_id] / global_maximum
            if global_maximum > 0.0
            else 0.0
        )
        personalized_component = (
            personalized_scores[target_page_id] / personalized_maximum
            if personalized_maximum > 0.0
            else 0.0
        )
        ranking_score = (
            semantic_score * config.semantic_weight
            + personalized_component * config.personalized_pagerank_weight
            + global_component * config.global_pagerank_weight
        )
        unselected.append(
            RankedWikiLink(
                target_page_id=target_page_id,
                relationships=relation.relationships,
                semantic_score=semantic_score,
                global_pagerank=global_scores[target_page_id],
                personalized_pagerank=personalized_scores[target_page_id],
                ranking_score=ranking_score,
                selected_for_page=False,
                contributing_candidate_pairs=relation.contributing_candidate_pairs,
            ),
        )

    unselected.sort(key=lambda item: (-item.ranking_score, item.target_page_id))
    threshold_targets = {
        item.target_page_id
        for item in unselected
        if item.semantic_score >= config.always_include_similarity
    }
    selected_targets = set(threshold_targets)
    slots = max(0, config.max_outgoing_links - len(selected_targets))
    for item in unselected:
        if slots == 0:
            break
        if item.target_page_id not in selected_targets:
            selected_targets.add(item.target_page_id)
            slots -= 1

    ranked = [
        item.model_copy(
            update={"selected_for_page": item.target_page_id in selected_targets},
        )
        for item in unselected
    ]
    ranked.sort(
        key=lambda item: (
            not item.selected_for_page,
            -item.ranking_score,
            item.target_page_id,
        ),
    )
    return ranked
