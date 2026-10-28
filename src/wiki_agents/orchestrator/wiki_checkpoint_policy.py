"""Define checkpoint compatibility policy for the Wiki pipeline."""

from collections.abc import Callable, Mapping, Sequence
from uuid import UUID

from wiki_agents.agents.wiki.aggregator import WikiPageGroup, deterministic_page_id
from wiki_agents.agents.wiki.models import (
    WikiCandidatePlan,
    WikiChangeStatus,
    WikiCreationResult,
    WikiFileChange,
    WikiPageContext,
    WikiPageDefinition,
    WikiPageRelation,
    WikiRelationCandidate,
    WikiRelationResolution,
)
from wiki_agents.agents.wiki.workflow import (
    WikiConceptStatus,
    WikiPageOutcome,
)
from wiki_agents.concept.models import ConceptCandidate, ConceptContext
from wiki_agents.orchestrator.mapping_checkpoints import (
    WikiCorpusCheckpoint,
    wiki_stage_fingerprint,
)

WIKI_MAPPING_PROMPT_VERSION = "1"
WIKI_MAPPING_SCHEMA_VERSION = "1"
WIKI_RELATION_PROMPT_VERSION = "1"
WIKI_RELATION_SCHEMA_VERSION = "1"
WIKI_PAGE_DEFINITION_PROMPT_VERSION = "2"
WIKI_PAGE_DEFINITION_SCHEMA_VERSION = "2"
WIKI_CORPUS_CHECKPOINT_KEY_VERSION = "4"


def _candidate_planning_input_fingerprint(
    contexts: Sequence[ConceptContext],
    config: Mapping[str, object],
) -> str:
    """Fingerprint stable candidate inputs without retrieval matches."""
    return wiki_stage_fingerprint(
        {
            "config": dict(config),
            "candidates": [
                context.candidate.model_dump(mode="json")
                for context in sorted(
                    contexts,
                    key=lambda item: str(item.candidate.candidate_id),
                )
            ],
        },
    )


def _same_candidate_plans(
    first: Sequence[WikiCandidatePlan],
    second: Sequence[WikiCandidatePlan],
) -> bool:
    """Return whether two plan sequences contain the same plans by ID."""
    first_by_id = {plan.candidate_id: plan for plan in first}
    second_by_id = {plan.candidate_id: plan for plan in second}
    return (
        len(first_by_id) == len(first)
        and len(second_by_id) == len(second)
        and first_by_id == second_by_id
    )


def _reusable_candidate_plans(
    checkpoint: WikiCorpusCheckpoint,
    candidate_ids: Sequence[UUID],
    input_fingerprint: str,
) -> list[WikiCandidatePlan]:
    """Return compatible candidate plans in current candidate order."""
    if checkpoint.candidate_planning_fingerprint not in {None, input_fingerprint}:
        return []
    plans_by_id = {plan.candidate_id: plan for plan in checkpoint.candidate_plans}
    if len(plans_by_id) != len(checkpoint.candidate_plans):
        return []
    if not set(plans_by_id).issubset(candidate_ids):
        return []
    return [plans_by_id[item] for item in candidate_ids if item in plans_by_id]


def _reusable_relation_resolutions(
    checkpoint: WikiCorpusCheckpoint,
    pair_ids: Sequence[tuple[UUID, UUID]],
    input_fingerprint: str,
) -> list[WikiRelationResolution]:
    """Return compatible semantic resolutions by pair membership."""
    if checkpoint.relation_input_fingerprint not in {None, input_fingerprint}:
        return []
    relations_by_pair = {
        relation.candidate_ids: relation for relation in checkpoint.relation_resolutions
    }
    if len(relations_by_pair) != len(checkpoint.relation_resolutions):
        return []
    if not set(relations_by_pair).issubset(pair_ids):
        return []
    return [relations_by_pair[item] for item in pair_ids if item in relations_by_pair]


def _reusable_page_definitions(
    checkpoint: WikiCorpusCheckpoint,
    groups: Sequence[WikiPageGroup],
    input_fingerprint: str,
) -> list[WikiPageDefinition]:
    """Return definitions for compatible completed page groups."""
    if checkpoint.page_definition_input_fingerprint not in {
        None,
        input_fingerprint,
    }:
        return []
    definitions_by_id = {
        definition.page_id: definition for definition in checkpoint.page_definitions
    }
    if len(definitions_by_id) != len(checkpoint.page_definitions):
        return []
    candidate_to_group = {
        candidate_id: group.page_id
        for group in groups
        for candidate_id in group.member_candidate_ids
    }
    actual_candidates = [
        candidate_id
        for definition in checkpoint.page_definitions
        for candidate_id in definition.member_candidate_ids
    ]
    if len(actual_candidates) != len(set(actual_candidates)) or not set(
        actual_candidates,
    ).issubset(candidate_to_group):
        return []
    definitions_by_group: dict[str, list[WikiPageDefinition]] = {}
    for definition in checkpoint.page_definitions:
        members = definition.member_candidate_ids
        if definition.page_id != deterministic_page_id(members):
            return []
        group_ids = {candidate_to_group[item] for item in members}
        if len(group_ids) != 1:
            return []
        definitions_by_group.setdefault(group_ids.pop(), []).append(definition)
    completed_group_ids = {
        group.page_id
        for group in groups
        if {
            candidate_id
            for definition in definitions_by_group.get(group.page_id, [])
            for candidate_id in definition.member_candidate_ids
        }
        == set(group.member_candidate_ids)
    }
    return sorted(
        (
            definition
            for group_id in completed_group_ids
            for definition in definitions_by_group[group_id]
        ),
        key=lambda item: item.page_id,
    )


def _relation_input_fingerprint(  # ruff: ignore[too-many-arguments] - The fingerprint has explicit semantic inputs and configuration.
    pairs: Sequence[WikiRelationCandidate],
    plans: dict[UUID, WikiCandidatePlan],
    candidates: dict[UUID, ConceptCandidate],
    *,
    retrieval_fingerprint: str,
    model: str,
    batch_size: int,
) -> str:
    """Fingerprint semantic pair inputs without scheduling metadata."""
    pair_inputs = [
        {
            "candidate_ids": [str(item) for item in pair.candidate_ids],
            "raw_similarity": pair.similarity_score,
            "plans": [
                plans[item].model_dump(mode="json") for item in pair.candidate_ids
            ],
            "candidates": [
                candidates[item].model_dump(mode="json") for item in pair.candidate_ids
            ],
        }
        for pair in sorted(
            pairs,
            key=lambda item: tuple(map(str, item.candidate_ids)),
        )
    ]
    return wiki_stage_fingerprint(
        {
            "retrieval_fingerprint": retrieval_fingerprint,
            "pairs": pair_inputs,
            "model": model,
            "prompt_version": WIKI_RELATION_PROMPT_VERSION,
            "schema_version": WIKI_RELATION_SCHEMA_VERSION,
            "batch_size": batch_size,
        },
    )


def _page_definition_input_fingerprint(  # ruff: ignore[too-many-arguments] - The fingerprint includes page gate settings.
    groups: Sequence[WikiPageGroup],
    plans: dict[UUID, WikiCandidatePlan],
    candidates: dict[UUID, ConceptCandidate],
    *,
    relation_fingerprint: str,
    model: str,
    max_page_candidates: int,
    max_page_work_items: int,
) -> str:
    """Fingerprint deterministic MERGE groups and definition-agent inputs."""
    group_inputs = [
        {
            "page_id": group.page_id,
            "member_candidate_ids": [str(item) for item in group.member_candidate_ids],
            "plans": [
                plans[item].model_dump(mode="json")
                for item in group.member_candidate_ids
            ],
            "claims": [
                {
                    "candidate_id": str(item),
                    "claims": [
                        claim.model_dump(mode="json")
                        for claim in candidates[item].concept.claims
                    ],
                }
                for item in group.member_candidate_ids
            ],
        }
        for group in groups
    ]
    return wiki_stage_fingerprint(
        {
            "groups": group_inputs,
            "relation_fingerprint": relation_fingerprint,
            "model": model,
            "max_page_candidates": max_page_candidates,
            "max_page_work_items": max_page_work_items,
            "prompt_version": WIKI_PAGE_DEFINITION_PROMPT_VERSION,
            "schema_version": WIKI_PAGE_DEFINITION_SCHEMA_VERSION,
        },
    )


def _page_graph_content_metadata(
    definitions: Sequence[WikiPageDefinition],
    relations: Sequence[WikiPageRelation],
) -> dict[str, object]:
    """Return stable accepted-graph data for fingerprints."""
    return {
        "nodes": sorted(definition.page_id for definition in definitions),
        "edges": [
            relation.model_dump(mode="json")
            for relation in sorted(
                relations,
                key=lambda item: (item.source_page_id, item.target_page_id),
            )
        ],
    }


def _reusable_page_outcomes(
    checkpoint: WikiCorpusCheckpoint,
    contexts: Sequence[WikiPageContext],
    page_generation_fingerprint: str,
    load_page_proposal: Callable[[str], bytes | None],
) -> dict[str, WikiPageOutcome]:
    """Restore compatible accepted page proposals without running page workers."""
    if checkpoint.page_generation_fingerprint != page_generation_fingerprint:
        return {}
    attempts_by_id = {attempt.page_id: attempt for attempt in checkpoint.page_attempts}
    accepted_page_ids = set(checkpoint.accepted_page_ids)
    outcomes: dict[str, WikiPageOutcome] = {}
    for context in contexts:
        page_id = context.page_job.page_id
        if page_id not in accepted_page_ids:
            continue
        content = load_page_proposal(page_id)
        attempt = attempts_by_id.get(page_id)
        target_path = context.page_job.plan.target_path
        if content is None or attempt is None or target_path is None:
            continue
        outcomes[page_id] = WikiPageOutcome(
            page_id=page_id,
            status=WikiConceptStatus.ACCEPTED,
            creation=WikiCreationResult(
                plan=context.page_job.plan,
                changes=(
                    WikiFileChange(
                        path=target_path,
                        status=WikiChangeStatus.ADDED,
                        content=content,
                    ),
                ),
                raw_output="Reused a compatible accepted page proposal.",
            ),
            correction_runs=attempt.attempts - 1,
        )
    return outcomes
