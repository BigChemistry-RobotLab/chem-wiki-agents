"""Retrieve and resolve sparse work-package relations."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from itertools import batched
from threading import Lock
from typing import TYPE_CHECKING, Protocol

from agents import Agent, Runner
from agents.exceptions import ModelBehaviorError

from wiki_agents.agents.wiki.models import (
    WikiCandidatePlan,
    WikiRelationBatch,
    WikiRelationCandidate,
    WikiRelationResolution,
    WorkPackageNeighbor,
)
from wiki_agents.tracing import pdf_source_metadata, traced_workflow

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from uuid import UUID

    from wiki_agents.concept.models import ConceptCandidate
    from wiki_agents.embedding.work_package_store import WorkPackageMatch


class WorkPackageSimilarityStore(Protocol):
    """Define the bounded retrieval operation used for sparse pair creation."""

    def top_k_similar_work_packages(
        self,
        plan: WikiCandidatePlan,
        *,
        top_k: int,
    ) -> Sequence[WorkPackageMatch]:
        """Return at most ``top_k`` work-package neighbors."""
        ...


class _RelationBatchOutputError(RuntimeError):
    """Report relation output that does not match the requested pairs."""


def retrieve_work_package_neighbors(
    plans: Sequence[WikiCandidatePlan],
    store: WorkPackageSimilarityStore,
    *,
    top_k: int = 20,
    progress: Callable[[int], None] | None = None,
) -> list[WorkPackageNeighbor]:
    """Retrieve and preserve directed top-K work-package observations."""
    if top_k < 1:
        return []
    create_plans = [plan for plan in plans if plan.decision.value == "CREATE"]
    valid_ids = {plan.candidate_id for plan in create_plans}
    observations: dict[tuple[UUID, UUID], float] = {}
    for plan in sorted(create_plans, key=lambda item: str(item.candidate_id)):
        for match in store.top_k_similar_work_packages(plan, top_k=top_k):
            target_id = match.plan.candidate_id
            if target_id not in valid_ids:
                raise ValueError(
                    "a retrieved work-package target has no current CREATE plan",
                )
            observation = WorkPackageNeighbor(
                source_candidate_id=plan.candidate_id,
                target_candidate_id=target_id,
                similarity_score=match.score,
            )
            direction = (
                observation.source_candidate_id,
                observation.target_candidate_id,
            )
            score = observation.similarity_score
            observations[direction] = max(
                observations.get(direction, float("-inf")),
                score,
            )
        if progress is not None:
            progress(1)
    return [
        WorkPackageNeighbor(
            source_candidate_id=source_id,
            target_candidate_id=target_id,
            similarity_score=score,
        )
        for (source_id, target_id), score in sorted(
            observations.items(),
            key=lambda item: (str(item[0][0]), str(item[0][1])),
        )
    ]


def retrieve_relation_candidates(
    plans: Sequence[WikiCandidatePlan],
    store: WorkPackageSimilarityStore,
    *,
    top_k: int = 20,
) -> list[WikiRelationCandidate]:
    """Return canonical pairs for callers that do not schedule with PageRank."""
    observations = retrieve_work_package_neighbors(plans, store, top_k=top_k)
    pairs: dict[tuple[UUID, UUID], tuple[float, bool]] = {}
    directions = {
        (item.source_candidate_id, item.target_candidate_id) for item in observations
    }
    for item in observations:
        sorted_ids = sorted(
            (item.source_candidate_id, item.target_candidate_id),
            key=str,
        )
        pair = (sorted_ids[0], sorted_ids[1])
        raw, _ = pairs.get(pair, (float("-inf"), False))
        reciprocal = (item.target_candidate_id, item.source_candidate_id) in directions
        pairs[pair] = (max(raw, item.similarity_score), reciprocal)
    return [
        WikiRelationCandidate(
            candidate_ids=pair,
            similarity_score=raw,
            ranking_similarity=min(1.0, max(0.0, raw)),
            reciprocal=reciprocal,
        )
        for pair, (raw, reciprocal) in sorted(
            pairs.items(),
            key=lambda item: (str(item[0][0]), str(item[0][1])),
        )
    ]


def _pair_payload(
    pair: WikiRelationCandidate,
    plans: Mapping[UUID, WikiCandidatePlan],
    candidates: Mapping[UUID, ConceptCandidate],
) -> dict[str, object]:
    """Format both sides of one pair with original disambiguation data."""
    sides = []
    for candidate_id in pair.candidate_ids:
        plan = plans[candidate_id]
        candidate = candidates[candidate_id]
        sides.append(
            {
                "candidate_id": str(candidate_id),
                "page_intent": plan.page_intent.model_dump(mode="json")
                if plan.page_intent is not None
                else None,
                "original_title": candidate.concept.title,
                "original_description": candidate.concept.description,
                "claim_statements": [
                    claim.statement for claim in candidate.concept.claims
                ],
            },
        )
    return {
        "candidate_ids": [str(candidate_id) for candidate_id in pair.candidate_ids],
        "similarity_score": pair.similarity_score,
        "sides": sides,
    }


def resolve_relations_concurrently(  # ruff: ignore[complex-structure,too-many-arguments] - Resolution needs fixed corpus inputs, resume state, and bounded execution settings.
    agent: Agent[None],
    pairs: Sequence[WikiRelationCandidate],
    plans: Mapping[UUID, WikiCandidatePlan],
    candidates: Mapping[UUID, ConceptCandidate],
    *,
    batch_size: int = 24,
    workers: int = 8,
    completed_relations: Sequence[WikiRelationResolution] = (),
    checkpoint: Callable[[list[WikiRelationResolution]], None] | None = None,
) -> list[WikiRelationResolution]:
    """Resolve pending pair batches and checkpoint each valid response."""
    if batch_size < 1 or workers < 1:
        raise ValueError("relation batch size and workers must be positive")
    pair_ids = [pair.candidate_ids for pair in pairs]
    pair_id_set = set(pair_ids)
    completed_ids = [relation.candidate_ids for relation in completed_relations]
    if len(completed_ids) != len(set(completed_ids)):
        raise ValueError("completed relations cannot repeat a candidate pair")
    if not set(completed_ids).issubset(pair_id_set):
        raise ValueError("completed relations must match requested candidate pairs")

    resolved = {relation.candidate_ids: relation for relation in completed_relations}
    pending_pairs = [pair for pair in pairs if pair.candidate_ids not in resolved]
    pair_batches = list(batched(pending_pairs, batch_size))
    checkpoint_lock = Lock()

    def record_response(items: Sequence[WikiRelationResolution]) -> None:
        with checkpoint_lock:
            resolved.update((item.candidate_ids, item) for item in items)
            if checkpoint is not None:
                checkpoint(
                    [resolved[pair_id] for pair_id in pair_ids if pair_id in resolved],
                )

    def request_batch(
        pair_batch: tuple[WikiRelationCandidate, ...],
    ) -> list[WikiRelationResolution]:
        candidate_ids = sorted(
            {
                candidate_id
                for pair in pair_batch
                for candidate_id in pair.candidate_ids
            },
            key=str,
        )
        with traced_workflow(
            "Resolve wiki relation batch",
            metadata={
                "stage": "relation_resolution",
                "message": "Resolve merge and link relationships between candidates.",
                "candidate_ids": [str(candidate_id) for candidate_id in candidate_ids],
                "pair_count": len(pair_batch),
                **pdf_source_metadata(
                    [candidates[candidate_id].source for candidate_id in candidate_ids],
                ),
            },
        ) as trace_metadata:
            payload = {
                "pairs": [
                    _pair_payload(pair, plans, candidates) for pair in pair_batch
                ],
            }
            result = Runner.run_sync(agent, json.dumps(payload, sort_keys=True))
            output = result.final_output_as(
                WikiRelationBatch,
                raise_if_incorrect_type=True,
            )
            expected = [pair.candidate_ids for pair in pair_batch]
            if [item.candidate_ids for item in output.items] != expected:
                raise _RelationBatchOutputError(
                    "relation resolver changed pair order or membership",
                )
            trace_metadata["decisions"] = [item.decision.value for item in output.items]
            record_response(output.items)
            return output.items

    def run_batch(
        pair_batch: tuple[WikiRelationCandidate, ...],
    ) -> list[WikiRelationResolution]:
        try:
            return request_batch(pair_batch)
        except (ModelBehaviorError, _RelationBatchOutputError):
            if len(pair_batch) == 1:
                return request_batch(pair_batch)
            midpoint = len(pair_batch) // 2
            return run_batch(pair_batch[:midpoint]) + run_batch(pair_batch[midpoint:])

    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(run_batch, pair_batches))
    return [resolved[pair_id] for pair_id in pair_ids]
