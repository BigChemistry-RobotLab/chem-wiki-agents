"""Plan one grounded change to an OKF wiki."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import batched
from typing import TYPE_CHECKING

from agents import Agent, Model, Runner
from pydantic import BaseModel, ConfigDict, Field

from wiki_agents.agents.wiki.models import (
    WikiCandidateBatchPlan,
    WikiCandidatePlan,
)
from wiki_agents.tracing import pdf_source_metadata, traced_workflow

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from wiki_agents.concept.models import ConceptContext


class _WikiCandidateBatchOutput(BaseModel):
    """Parse candidate planner output before strict membership validation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    items: list[WikiCandidatePlan] = Field(min_length=1)


def create_wiki_candidate_planner_agent(
    *,
    model: str | Model | None = None,
) -> Agent[None]:
    """Create the full-corpus candidate-intent planning agent."""
    return Agent(
        name="Wiki candidate planner",
        model=model,
        instructions="""Plan each supplied candidate independently.

Return exactly one candidate plan per input candidate, in input order, and copy its
candidate ID. Use CREATE when the candidate contains useful supported knowledge
that must be represented somewhere in the generated Wiki. Use SKIP only when the
candidate is not useful Wiki knowledge. A similar candidate is never a reason to
SKIP useful knowledge; later aggregation handles duplication.

For CREATE, write a concise normalized page intent. The canonical subject, scope,
and work items must describe the semantic Wiki subject, not the source paper. Use
all supplied claims and evidence. Do not select a path, existing page, UPDATE
action, or final Wiki link graph. Treat each result as independent even though the
request contains a batch. Do not add scientific knowledge.
""",
        output_type=_WikiCandidateBatchOutput,
    )


def _candidate_planning_payload(context: ConceptContext) -> dict[str, object]:
    """Format one candidate and its concept-level semantic matches."""
    candidate = context.candidate
    return {
        "candidate_id": str(candidate.candidate_id),
        "title": candidate.concept.title,
        "description": candidate.concept.description,
        "claims": [claim.model_dump(mode="json") for claim in candidate.concept.claims],
        "similar_concepts": [
            {
                "candidate_id": str(match.candidate.candidate_id),
                "title": match.candidate.concept.title,
                "description": match.candidate.concept.description,
                "score": match.score,
            }
            for match in context.similar_concepts
        ],
    }


def plan_candidates_concurrently(  # ruff: ignore[complex-structure,too-many-arguments] - Checkpoint options keep batch planning resumable.
    agent: Agent[None],
    contexts: Sequence[ConceptContext],
    *,
    batch_size: int = 5,
    workers: int = 8,
    completed_plans: Sequence[WikiCandidatePlan] = (),
    checkpoint: Callable[[list[WikiCandidatePlan]], None] | None = None,
) -> list[WikiCandidatePlan]:
    """Plan pending batches and checkpoint successful results in input order."""
    if batch_size < 1 or workers < 1:
        raise ValueError("candidate planning batch size and workers must be positive")
    candidate_ids = [context.candidate.candidate_id for context in contexts]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate planning contexts must have unique candidate IDs")
    candidate_id_set = set(candidate_ids)
    plans_by_id = {plan.candidate_id: plan for plan in completed_plans}
    if len(plans_by_id) != len(completed_plans):
        raise ValueError("completed candidate plans must have unique candidate IDs")
    if unknown_ids := set(plans_by_id) - candidate_id_set:
        unknown = ", ".join(sorted(str(candidate_id) for candidate_id in unknown_ids))
        raise ValueError(f"completed candidate plans contain unknown IDs: {unknown}")
    pending_contexts = [
        context
        for context in contexts
        if context.candidate.candidate_id not in plans_by_id
    ]
    context_batches = list(batched(pending_contexts, batch_size))

    def ordered_plans() -> list[WikiCandidatePlan]:
        return [
            plans_by_id[candidate_id]
            for candidate_id in candidate_ids
            if candidate_id in plans_by_id
        ]

    def run_batch(batch: tuple[ConceptContext, ...]) -> list[WikiCandidatePlan]:
        candidate_ids = [str(item.candidate.candidate_id) for item in batch]
        with traced_workflow(
            "Plan wiki candidate batch",
            metadata={
                "stage": "candidate_planning",
                "message": "Choose the wiki representation for source concepts.",
                "candidate_ids": candidate_ids,
                **pdf_source_metadata([item.candidate.source for item in batch]),
            },
        ) as trace_metadata:
            payload = {
                "candidates": [_candidate_planning_payload(item) for item in batch],
            }
            result = Runner.run_sync(agent, json.dumps(payload, sort_keys=True))
            output = result.final_output_as(
                _WikiCandidateBatchOutput,
                raise_if_incorrect_type=True,
            )
            expected = [item.candidate.candidate_id for item in batch]
            plans_for_id: dict[object, list[WikiCandidatePlan]] = {}
            for plan in output.items:
                plans_for_id.setdefault(plan.candidate_id, []).append(plan)
            resolved = {
                candidate_id: plans[0]
                for candidate_id, plans in plans_for_id.items()
                if candidate_id in expected and len(plans) == 1
            }
            unresolved = [
                context
                for context in batch
                if context.candidate.candidate_id not in resolved
            ]
            if unresolved and len(batch) == 1:
                raise RuntimeError(
                    "candidate planner changed candidate membership for a singleton request",
                )
            for context in unresolved:
                [plan] = run_batch((context,))
                resolved[plan.candidate_id] = plan
            strict_output = WikiCandidateBatchPlan(
                items=[resolved[candidate_id] for candidate_id in expected],
            )
            trace_metadata["decisions"] = [
                plan.decision.value for plan in strict_output.items
            ]
            return strict_output.items

    first_error: BaseException | None = None
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_batch, batch) for batch in context_batches]
        for future in as_completed(futures):
            error = future.exception()
            if error is not None:
                if first_error is None:
                    first_error = error
                continue
            batch_plans = future.result()
            plans_by_id.update((plan.candidate_id, plan) for plan in batch_plans)
            if checkpoint is not None:
                checkpoint(ordered_plans())
    if first_error is not None:
        raise first_error
    return ordered_plans()
