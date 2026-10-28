"""Compile candidate relations into deterministic Wiki page definitions."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import NAMESPACE_URL, UUID, uuid5

from agents import Agent, Model, Runner

from wiki_agents.agents.wiki.models import (
    RankedWikiLink,
    WikiCandidateDecision,
    WikiCandidatePairContribution,
    WikiCandidatePlan,
    WikiLinkPlan,
    WikiManifestLink,
    WikiManifestPage,
    WikiPageContext,
    WikiPageDefinition,
    WikiPageDefinitionBatch,
    WikiPageJob,
    WikiPageManifest,
    WikiPageRelation,
    WikiPlan,
    WikiPlanDecision,
    WikiRelationCandidate,
    WikiRelationDecision,
    WikiRelationResolution,
    WikiSelectedLinkContext,
    candidate_contribution_sort_key,
)
from wiki_agents.tracing import pdf_source_metadata, traced_workflow

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    from wiki_agents.concept.models import ConceptCandidate

_PAGE_ID_NAMESPACE = uuid5(NAMESPACE_URL, "wiki-agents/page-group/v1")
_NON_SLUG = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True, slots=True)
class WikiPageGroup:
    """Keep the stable identity and members of one merged page group."""

    page_id: str
    member_candidate_ids: tuple[UUID, ...]


class _DisjointSet:
    """Maintain deterministic candidate merge components."""

    def __init__(self, members: Iterable[UUID]) -> None:
        self._parent = {member: member for member in members}

    def find(self, member: UUID) -> UUID:
        """Return the representative and compress the path."""
        parent = self._parent[member]
        if parent != member:
            self._parent[member] = self.find(parent)
        return self._parent[member]

    def union(self, first: UUID, second: UUID) -> None:
        """Join two components using a stable representative."""
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return
        low, high = sorted((first_root, second_root), key=str)
        self._parent[high] = low


def deterministic_page_id(candidate_ids: Iterable[UUID]) -> str:
    """Return a stable UUIDv5 for a non-empty sorted candidate set."""
    values = sorted({str(candidate_id) for candidate_id in candidate_ids})
    if not values:
        raise ValueError("a page group must contain at least one candidate")
    return str(uuid5(_PAGE_ID_NAMESPACE, "\n".join(values)))


def collapse_merge_relations(
    plans: Sequence[WikiCandidatePlan],
    relations: Sequence[WikiRelationResolution],
    relation_candidates: Mapping[tuple[UUID, UUID], WikiRelationCandidate],
) -> tuple[list[WikiPageGroup], list[WikiPageRelation]]:
    """Collapse MERGE components and lift approved candidate LINK decisions."""
    candidate_ids = {
        plan.candidate_id
        for plan in plans
        if plan.decision is WikiCandidateDecision.CREATE
    }
    disjoint = _DisjointSet(candidate_ids)
    for relation in relations:
        first, second = relation.candidate_ids
        if first not in candidate_ids or second not in candidate_ids:
            raise ValueError("a relation references a candidate without a CREATE plan")
        if relation.decision is WikiRelationDecision.MERGE:
            disjoint.union(first, second)

    components: dict[UUID, list[UUID]] = defaultdict(list)
    for candidate_id in candidate_ids:
        components[disjoint.find(candidate_id)].append(candidate_id)
    groups = [
        WikiPageGroup(
            page_id=deterministic_page_id(members),
            member_candidate_ids=tuple(sorted(members, key=str)),
        )
        for members in components.values()
    ]
    groups.sort(key=lambda group: group.page_id)
    return groups, lift_candidate_links(groups, relations, relation_candidates)


def lift_candidate_links(
    groups: Sequence[WikiPageGroup],
    relations: Sequence[WikiRelationResolution],
    relation_candidates: Mapping[tuple[UUID, UUID], WikiRelationCandidate],
) -> list[WikiPageRelation]:
    """Lift candidate LINK decisions to the supplied final page groups."""
    candidate_to_page = {
        candidate_id: group.page_id
        for group in groups
        for candidate_id in group.member_candidate_ids
    }

    lifted_relationships: dict[tuple[str, str], set[str]] = defaultdict(set)
    lifted_contributions: dict[
        tuple[str, str],
        dict[tuple[UUID, UUID], WikiCandidatePairContribution],
    ] = defaultdict(dict)
    for relation in relations:
        if relation.decision is not WikiRelationDecision.LINK:
            continue
        first_candidate_id, second_candidate_id = relation.candidate_ids
        candidate_pair = (first_candidate_id, second_candidate_id)
        try:
            candidate_metadata = relation_candidates[candidate_pair]
        except KeyError as error:
            raise ValueError(
                "a LINK resolution has no retrieved-pair similarity",
            ) from error
        for link in relation.links:
            source_page = candidate_to_page[link.source_candidate_id]
            target_page = candidate_to_page[link.target_candidate_id]
            if source_page != target_page:
                page_pair = (source_page, target_page)
                lifted_relationships[page_pair].add(link.relationship)
                contribution = WikiCandidatePairContribution(
                    source_candidate_id=link.source_candidate_id,
                    target_candidate_id=link.target_candidate_id,
                    similarity=candidate_metadata.ranking_similarity,
                )
                contribution_key = (
                    contribution.source_candidate_id,
                    contribution.target_candidate_id,
                )
                lifted_contributions[page_pair][contribution_key] = contribution
    page_relations = []
    for (source, target), reasons in sorted(lifted_relationships.items()):
        contributions = tuple(
            sorted(
                lifted_contributions[source, target].values(),
                key=candidate_contribution_sort_key,
            ),
        )
        page_relations.append(
            WikiPageRelation(
                source_page_id=source,
                target_page_id=target,
                relationships=tuple(sorted(reasons)),
                semantic_score=max(item.similarity for item in contributions),
                contributing_candidate_pairs=contributions,
            ),
        )
    return page_relations


def singleton_page_definition(
    group: WikiPageGroup,
    plan: WikiCandidatePlan,
) -> WikiPageDefinition:
    """Convert one singleton intent without another model request."""
    if group.member_candidate_ids != (plan.candidate_id,):
        raise ValueError("the candidate plan does not match the singleton group")
    if plan.page_intent is None:
        raise ValueError("a singleton page requires a CREATE page intent")
    return WikiPageDefinition(
        page_id=group.page_id,
        title=plan.page_intent.canonical_subject,
        scope=plan.page_intent.scope,
        work_summary=plan.page_intent.work_summary,
        member_candidate_ids=[plan.candidate_id],
    )


def create_wiki_page_definition_agent(
    *,
    model: str | Model | None = None,
) -> Agent[None]:
    """Create the agent that partitions and compiles one provisional page group."""
    return Agent(
        name="Wiki page-definition compiler",
        model=model,
        instructions="""Compile one provisional MERGE group into one or more page drafts.

Use only the supplied candidate plans and original claims. Preserve every member
candidate ID exactly once. Keep one page when all candidates form one coherent Wiki
subject and the supplied page limits permit it. Otherwise, partition the candidates
into the smallest coherent set of pages that satisfies the supplied limits. A page
with one indivisible candidate may exceed the work-item limit.

For each page, preserve every distinct supported work item and combine duplicate work
items. Create one canonical title and coherent scope. Do not add scientific knowledge.
Do not choose a page ID, target path, or Wiki links.
""",
        output_type=WikiPageDefinitionBatch,
    )


def compile_page_definitions_concurrently(  # ruff: ignore[complex-structure,too-many-arguments,too-many-branches,too-many-statements] - Compilation validates model partitions and supports resumable progress reports.
    agent: Agent[None],
    groups: Sequence[WikiPageGroup],
    plans: Mapping[UUID, WikiCandidatePlan],
    candidates: Mapping[UUID, ConceptCandidate],
    *,
    max_page_candidates: int = 12,
    max_page_work_items: int = 20,
    max_membership_retries: int = 2,
    workers: int = 8,
    completed_definitions: Sequence[WikiPageDefinition] = (),
    checkpoint: Callable[[list[WikiPageDefinition]], None] | None = None,
    progress: Callable[[int], None] | None = None,
) -> list[WikiPageDefinition]:
    """Compile groups and partition groups that exceed the page gates."""
    if min(workers, max_page_candidates, max_page_work_items) < 1:
        raise ValueError("page-plan workers and page limits must be positive")
    if max_membership_retries < 0:
        raise ValueError("page-definition membership retries must be non-negative")
    definitions = {
        definition.page_id: definition for definition in completed_definitions
    }
    if len(definitions) != len(completed_definitions):
        raise ValueError("completed page definitions must have unique page IDs")
    completed_candidate_ids = {
        candidate_id
        for definition in completed_definitions
        for candidate_id in definition.member_candidate_ids
    }
    if sum(
        len(definition.member_candidate_ids) for definition in completed_definitions
    ) != len(completed_candidate_ids):
        raise ValueError("completed page definitions must have unique candidate IDs")
    multi_groups = []
    completed_group_count = 0
    added_singletons = False
    for group in groups:
        completed_members = set(group.member_candidate_ids) & completed_candidate_ids
        if completed_members:
            if completed_members != set(group.member_candidate_ids):
                raise ValueError("completed page definitions contain a partial group")
            completed_group_count += 1
            continue
        if len(group.member_candidate_ids) == 1:
            candidate_id = group.member_candidate_ids[0]
            definitions[group.page_id] = singleton_page_definition(
                group,
                plans[candidate_id],
            )
            completed_group_count += 1
            added_singletons = True
        else:
            multi_groups.append(group)
    if progress is not None:
        progress(completed_group_count)
    if checkpoint is not None and added_singletons:
        checkpoint(sorted(definitions.values(), key=lambda item: item.page_id))

    def compile_group(group: WikiPageGroup) -> list[WikiPageDefinition]:
        with traced_workflow(
            "Compile wiki page definition",
            metadata={
                "stage": "page_definition",
                "message": "Compile merged source concepts into one page definition.",
                "page_id": group.page_id,
                "candidate_ids": [
                    str(candidate_id) for candidate_id in group.member_candidate_ids
                ],
                **pdf_source_metadata(
                    [
                        candidates[candidate_id].source
                        for candidate_id in group.member_candidate_ids
                    ],
                ),
            },
        ) as trace_metadata:
            payload = {
                "provisional_page_id": group.page_id,
                "limits": {
                    "max_page_candidates": max_page_candidates,
                    "max_page_work_items": max_page_work_items,
                },
                "member_candidate_ids": [
                    str(candidate_id) for candidate_id in group.member_candidate_ids
                ],
                "candidate_plans": [
                    plans[candidate_id].model_dump(mode="json")
                    for candidate_id in group.member_candidate_ids
                ],
                "original_claims": [
                    {
                        "candidate_id": str(candidate_id),
                        "claims": [
                            claim.model_dump(mode="json")
                            for claim in candidates[candidate_id].concept.claims
                        ],
                    }
                    for candidate_id in group.member_candidate_ids
                ],
            }
            expected_members = set(group.member_candidate_ids)
            run_payload = payload
            batch: WikiPageDefinitionBatch | None = None
            for retry_number in range(max_membership_retries + 1):
                result = Runner.run_sync(
                    agent,
                    json.dumps(run_payload, sort_keys=True),
                )
                batch = result.final_output_as(
                    WikiPageDefinitionBatch,
                    raise_if_incorrect_type=True,
                )
                actual_members = {
                    candidate_id
                    for item in batch.items
                    for candidate_id in item.member_candidate_ids
                }
                if actual_members == expected_members:
                    break
                if retry_number == max_membership_retries:
                    raise RuntimeError(
                        "page-definition compiler changed group membership",
                    )
                run_payload = {
                    **payload,
                    "membership_correction": {
                        "instruction": (
                            "Return a new complete batch. Preserve every supplied member "
                            "candidate ID exactly once and do not add candidate IDs."
                        ),
                        "missing_candidate_ids": sorted(
                            str(candidate_id)
                            for candidate_id in expected_members - actual_members
                        ),
                        "unexpected_candidate_ids": sorted(
                            str(candidate_id)
                            for candidate_id in actual_members - expected_members
                        ),
                    },
                }
            if batch is None:
                raise RuntimeError("page-definition compiler did not run")
            for item in batch.items:
                if len(item.member_candidate_ids) > max_page_candidates:
                    raise RuntimeError(
                        "page-definition compiler exceeded the candidate limit",
                    )
                if (
                    len(item.member_candidate_ids) > 1
                    and len(item.work_summary) > max_page_work_items
                ):
                    raise RuntimeError(
                        "page-definition compiler exceeded the work-item limit",
                    )
            compiled = [
                WikiPageDefinition(
                    page_id=deterministic_page_id(item.member_candidate_ids),
                    title=item.title,
                    scope=item.scope,
                    work_summary=item.work_summary,
                    member_candidate_ids=sorted(item.member_candidate_ids, key=str),
                )
                for item in batch.items
            ]
            compiled.sort(key=lambda item: item.page_id)
            trace_metadata.update(
                page_ids=[item.page_id for item in compiled],
                page_count=len(compiled),
                status="completed",
            )
            if progress is not None:
                progress(1)
            return compiled

    first_error: BaseException | None = None
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(compile_group, group) for group in multi_groups]
        for future in as_completed(futures):
            error = future.exception()
            if error is not None:
                if first_error is None:
                    first_error = error
                continue
            compiled = future.result()
            for definition in compiled:
                if definition.page_id in definitions:
                    raise RuntimeError(
                        "page-definition compiler repeated a page identity",
                    )
                definitions[definition.page_id] = definition
            if checkpoint is not None:
                checkpoint(sorted(definitions.values(), key=lambda item: item.page_id))
    if first_error is not None:
        raise first_error
    return sorted(definitions.values(), key=lambda item: item.page_id)


def normalize_wiki_slug(title: str) -> str:
    """Return a repository-style lowercase ASCII filename stem."""
    ascii_title = (
        unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    )
    slug = _NON_SLUG.sub("-", ascii_title.casefold()).strip("-")
    return slug or "concept"


def assign_target_paths(
    definitions: Sequence[WikiPageDefinition],
) -> dict[str, str]:
    """Assign order-independent paths with stable suffixes for collisions."""
    by_slug: dict[str, list[WikiPageDefinition]] = defaultdict(list)
    for definition in definitions:
        by_slug[normalize_wiki_slug(definition.title)].append(definition)
    paths: dict[str, str] = {}
    for slug, values in by_slug.items():
        for definition in values:
            suffix = (
                ""
                if len(values) == 1
                else f"-{definition.page_id.replace('-', '')[:8]}"
            )
            paths[definition.page_id] = f"wiki/concepts/{slug}{suffix}.md"
    return paths


def build_page_manifest(
    definitions: Sequence[WikiPageDefinition],
    ranked_links: Mapping[str, Sequence[RankedWikiLink]],
    global_scores: Mapping[str, float],
) -> WikiPageManifest:
    """Build the complete immutable accepted graph and selection manifest."""
    definitions_by_id = {definition.page_id: definition for definition in definitions}
    if len(definitions_by_id) != len(definitions):
        raise ValueError("page definitions contain a duplicate page ID")
    if set(global_scores) != definitions_by_id.keys():
        raise ValueError("global PageRank scores do not match page definitions")
    if not set(ranked_links).issubset(definitions_by_id):
        raise ValueError("ranked links contain an unknown source page")
    paths = assign_target_paths(definitions)
    outbound: dict[str, list[WikiManifestLink]] = defaultdict(list)
    for source_page_id, links in ranked_links.items():
        seen_targets: set[str] = set()
        for link in links:
            if link.target_page_id not in definitions_by_id:
                raise ValueError("a ranked link has an unknown target page")
            if link.target_page_id in seen_targets:
                raise ValueError("ranked links repeat an accepted target")
            seen_targets.add(link.target_page_id)
            target_path = paths[link.target_page_id]
            outbound[source_page_id].append(
                WikiManifestLink(
                    target_page_id=link.target_page_id,
                    target_path="/" + target_path.removeprefix("wiki/"),
                    relationships=list(link.relationships),
                    semantic_score=link.semantic_score,
                    global_pagerank=link.global_pagerank,
                    personalized_pagerank=link.personalized_pagerank,
                    ranking_score=link.ranking_score,
                    selected_for_page=link.selected_for_page,
                    contributing_candidate_pairs=list(
                        link.contributing_candidate_pairs,
                    ),
                ),
            )
    pages = []
    for definition in sorted(
        definitions,
        key=lambda item: (-global_scores[item.page_id], item.page_id),
    ):
        target_path = paths[definition.page_id]
        page_links: list[WikiManifestLink] = list(outbound[definition.page_id])
        page_links.sort(key=_manifest_link_sort_key)
        pages.append(
            WikiManifestPage(
                page_id=definition.page_id,
                target_path=target_path,
                bundle_path="/" + target_path.removeprefix("wiki/"),
                title=definition.title,
                scope=definition.scope,
                work_summary=definition.work_summary,
                member_candidate_ids=definition.member_candidate_ids,
                global_pagerank=global_scores[definition.page_id],
                links=page_links,
            ),
        )
    return WikiPageManifest(pages=pages)


def _manifest_link_sort_key(link: WikiManifestLink) -> tuple[bool, float, str]:
    """Return the stable sort key for one final manifest link."""
    return not link.selected_for_page, -link.ranking_score, link.target_page_id


def build_page_contexts(
    manifest: WikiPageManifest,
    candidates: Sequence[ConceptCandidate],
) -> list[WikiPageContext]:
    """Build page jobs from original candidate objects without rewriting claims."""
    candidates_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    pages_by_id = {page.page_id: page for page in manifest.pages}
    contexts = []
    for page in manifest.pages:
        page_candidates = [candidates_by_id[item] for item in page.member_candidate_ids]
        selected_links = [link for link in page.links if link.selected_for_page]
        plan = WikiPlan(
            decision=WikiPlanDecision.CREATE,
            rationale="The corpus planner assigned this final page definition.",
            target_path=page.target_path,
            title=page.title,
            work_summary=page.work_summary,
            links=[
                WikiLinkPlan(
                    target_path=link.target_path,
                    relationship="; ".join(link.relationships),
                )
                for link in selected_links
            ],
        )
        selected_link_contexts = [
            WikiSelectedLinkContext(
                target_page_id=linked.page_id,
                target_path=link.target_path,
                target_title=linked.title,
                relationship="; ".join(link.relationships),
            )
            for link in selected_links
            for linked in [pages_by_id[link.target_page_id]]
        ]
        contexts.append(
            WikiPageContext(
                page_job=WikiPageJob(
                    page_id=page.page_id,
                    plan=plan,
                    member_candidate_ids=page.member_candidate_ids,
                ),
                candidates=page_candidates,
                selected_links=selected_link_contexts,
            ),
        )
    return contexts
