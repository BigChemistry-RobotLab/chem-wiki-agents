"""Plan the corpus-wide Wiki graph."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from wiki_agents.agents.wiki.aggregator import (
    WikiPageGroup,
    build_page_manifest,
    collapse_merge_relations,
    compile_page_definitions_concurrently,
    lift_candidate_links,
)
from wiki_agents.agents.wiki.graph_ranking import (
    GRAPH_RANKING_ALGORITHM_VERSION,
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
    WikiCandidatePlan,
    WikiGraphRankingConfig,
    WikiPageDefinition,
    WikiPageManifest,
    WikiPageRelation,
    WikiRelationDecision,
    WikiRelationResolution,
)
from wiki_agents.agents.wiki.resolver import retrieve_work_package_neighbors
from wiki_agents.orchestrator.mapping_checkpoints import (
    WikiCandidatePageMapping,
    WikiCandidatePageRank,
    WikiCorpusCheckpoint,
    WikiFinalGraphCheckpoint,
    WikiMappingCheckpointStore,
    WikiPagePageRank,
    WikiPreliminaryGraphCheckpoint,
    wiki_stage_fingerprint,
)
from wiki_agents.orchestrator.wiki_checkpoint_policy import (
    _candidate_planning_input_fingerprint,
    _page_definition_input_fingerprint,
    _page_graph_content_metadata,
    _relation_input_fingerprint,
    _reusable_candidate_plans,
    _reusable_page_definitions,
    _reusable_relation_resolutions,
    _same_candidate_plans,
)
from wiki_agents.tracing import (
    init_weave_tracing,
    pdf_source_metadata,
    record_trace_event,
    traced_workflow,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from uuid import UUID

    import networkx as nx

    from wiki_agents.agents.wiki.models import WikiRelationCandidate
    from wiki_agents.agents.wiki.workflow import (
        WikiCandidatePlanningWorkflow,
        WikiPartitionBoundaryWorkflow,
        WikiRelationResolutionWorkflow,
    )
    from wiki_agents.concept.models import (
        ConceptCandidate,
        ConceptContext,
        ExtractedSourceConcepts,
    )
    from wiki_agents.orchestrator.concept_store import ConceptStoreProvider
    from wiki_agents.orchestrator.config import WikiOrchestratorConfig
    from wiki_agents.progress import ProgressReporter

logger = logging.getLogger("wiki_agents.orchestrator")


@dataclass(frozen=True, slots=True)
class _PreliminaryPlanningState:
    """Keep the outputs from the preliminary graph stage."""

    checkpoint: WikiCorpusCheckpoint
    similarity_graph: nx.Graph
    centrality: dict[UUID, float]
    input_fingerprint: str


@dataclass(frozen=True, slots=True)
class _RelationPlanningState:
    """Keep the outputs from the relation resolution stage."""

    checkpoint: WikiCorpusCheckpoint
    groups: list[WikiPageGroup]
    relations: list[WikiRelationResolution]
    relation_candidates: dict[tuple[UUID, UUID], WikiRelationCandidate]
    plans_by_id: dict[UUID, WikiCandidatePlan]


@dataclass(frozen=True, slots=True)
class _FinalGraphFingerprints:
    """Keep the fingerprints for one final graph ranking."""

    input: str
    content: str
    ranking: str


class WikiPlanningMixin:
    """Provide corpus graph planning to the Wiki orchestrator."""

    config: WikiOrchestratorConfig
    candidate_planning_config_metadata: dict[str, object]
    work_package_retrieval_config_metadata: dict[str, object]
    preliminary_pagerank_config_fingerprint: str
    graph_ranking_config: WikiGraphRankingConfig
    concept_store_provider: ConceptStoreProvider
    candidate_planning_workflow: WikiCandidatePlanningWorkflow
    relation_resolution_workflow: WikiRelationResolutionWorkflow
    partition_boundary_workflow: WikiPartitionBoundaryWorkflow
    page_definition_agent: Any
    progress: ProgressReporter
    wiki_mapping_checkpoints: WikiMappingCheckpointStore
    load_extracted_sources: Callable[[], list[ExtractedSourceConcepts]]
    prepare_wiki_input: Callable[
        [Sequence[ExtractedSourceConcepts]],
        list[ConceptContext],
    ]

    def create_plans(
        self,
        extracted_sources: Sequence[ExtractedSourceConcepts] | None = None,
    ) -> WikiPageManifest | None:
        """Create the candidate, relation, page, and graph plans."""
        init_weave_tracing()
        if extracted_sources is None:
            extracted_sources = self.load_extracted_sources()
        if not extracted_sources:
            logger.info("No extracted sources are available for planning")
            return None
        candidates = [
            candidate
            for extracted in extracted_sources
            for candidate in extracted.candidates
        ]
        if not candidates:
            logger.info("No extracted concept candidates are available for planning")
            return None
        with traced_workflow(
            "Plan complete wiki graph",
            metadata={
                "stage": "wiki_planning",
                "message": "Plan candidates, relations, page groups, and wiki links.",
                "candidate_count": len(candidates),
                **pdf_source_metadata(
                    [extracted.source for extracted in extracted_sources],
                ),
            },
        ) as trace_metadata:
            manifest = self._create_plans(candidates, extracted_sources)
            trace_metadata.update(
                status="completed" if manifest is not None else "no_pages",
                page_count=len(manifest.pages) if manifest is not None else 0,
            )
            return manifest

    def _create_plans(
        self,
        candidates: list[ConceptCandidate],
        extracted_sources: Sequence[ExtractedSourceConcepts],
    ) -> WikiPageManifest | None:
        """Execute graph planning inside the correlated planning trace."""
        logger.info(
            "Stage 2/10: preparing %d corpus candidate contexts",
            len(candidates),
        )
        contexts = self.prepare_wiki_input(extracted_sources)
        candidates_by_id = {
            candidate.candidate_id: candidate for candidate in candidates
        }
        create_plans, checkpoint = self._plan_candidates(contexts)
        if not create_plans:
            logger.info("No candidates require Wiki representation")
            return None

        preliminary_state = self._prepare_work_package_graph(
            create_plans,
            checkpoint,
        )
        relation_state = self._resolve_candidate_relations(
            create_plans,
            candidates_by_id,
            preliminary_state,
        )
        definitions, checkpoint = self._compile_page_definitions(
            candidates_by_id,
            relation_state,
        )
        effective_relations, checkpoint = self._resolve_partition_boundaries(
            definitions,
            candidates_by_id,
            relation_state,
            checkpoint,
        )
        final_groups = [
            WikiPageGroup(
                page_id=definition.page_id,
                member_candidate_ids=tuple(definition.member_candidate_ids),
            )
            for definition in definitions
        ]
        page_relations = lift_candidate_links(
            final_groups,
            effective_relations,
            relation_state.relation_candidates,
        )
        return self._build_final_manifest(
            definitions,
            page_relations,
            checkpoint,
        )

    def _plan_candidates(
        self,
        contexts: list[ConceptContext],
    ) -> tuple[list[WikiCandidatePlan], WikiCorpusCheckpoint]:
        """Plan candidate work packages and save resumable progress."""
        candidate_ids = [context.candidate.candidate_id for context in contexts]
        candidate_planning_fingerprint = _candidate_planning_input_fingerprint(
            contexts,
            self.candidate_planning_config_metadata,
        )
        saved_checkpoint = self.wiki_mapping_checkpoints.load_corpus()
        reusable_plans = _reusable_candidate_plans(
            saved_checkpoint,
            candidate_ids,
            candidate_planning_fingerprint,
        )
        if saved_checkpoint.candidate_plans and not reusable_plans:
            logger.warning("Ignoring stale candidate planning checkpoint")
        if reusable_plans:
            reusable_ids = {plan.candidate_id for plan in reusable_plans}
            reusable_contexts = [
                context
                for context in contexts
                if context.candidate.candidate_id in reusable_ids
            ]
            record_trace_event(
                "Reuse candidate planning checkpoint",
                {
                    "stage": "candidate_planning",
                    "message": "Reuse compatible candidate plans.",
                    "status": "reused",
                    "candidate_ids": [
                        str(context.candidate.candidate_id)
                        for context in reusable_contexts
                    ],
                    **pdf_source_metadata(
                        [context.candidate.source for context in reusable_contexts],
                    ),
                },
            )
            logger.info(
                "Resuming candidate planning with %d checkpointed plans",
                len(reusable_plans),
            )

        def save_candidate_plans(plans: list[WikiCandidatePlan]) -> None:
            self.wiki_mapping_checkpoints.save_corpus(
                WikiCorpusCheckpoint(
                    candidate_planning_fingerprint=candidate_planning_fingerprint,
                    candidate_plans=plans,
                ),
            )

        logger.info("Stage 3/10: planning all candidate work packages")
        with self.progress.task(
            "planning candidate work packages",
            len(contexts),
            completed=len(reusable_plans),
        ) as progress:

            def save_and_report_candidate_plans(
                plans: list[WikiCandidatePlan],
            ) -> None:
                save_candidate_plans(plans)
                progress.update(len(plans))

            candidate_plans = self.candidate_planning_workflow.run(
                contexts,
                batch_size=self.config.wiki_planning_batch_size,
                workers=self.config.wiki_planning_workers,
                completed_plans=reusable_plans,
                checkpoint=save_and_report_candidate_plans,
            )
        create_plans = [
            plan
            for plan in candidate_plans
            if plan.decision is WikiCandidateDecision.CREATE
        ]
        if (
            _same_candidate_plans(
                candidate_plans,
                saved_checkpoint.candidate_plans,
            )
            and reusable_plans
        ):
            checkpoint = saved_checkpoint.model_copy(
                update={
                    "candidate_planning_fingerprint": candidate_planning_fingerprint,
                },
            )
        else:
            checkpoint = WikiCorpusCheckpoint(
                candidate_planning_fingerprint=candidate_planning_fingerprint,
                candidate_plans=candidate_plans,
            )
        self.wiki_mapping_checkpoints.save_corpus(checkpoint)
        return create_plans, checkpoint

    def _prepare_work_package_graph(
        self,
        create_plans: list[WikiCandidatePlan],
        checkpoint: WikiCorpusCheckpoint,
    ) -> _PreliminaryPlanningState:
        """Build or restore the preliminary work-package graph."""
        logger.info("Stage 4/10: embedding and retrieving sparse work-package pairs")
        work_package_input_fingerprint = wiki_stage_fingerprint(
            {
                "create_plans": [
                    plan.model_dump(mode="json")
                    for plan in sorted(
                        create_plans,
                        key=lambda item: str(item.candidate_id),
                    )
                ],
                "retrieval_config": self.work_package_retrieval_config_metadata,
            },
        )
        preliminary_graph = checkpoint.preliminary_graph
        similarity_graph = None
        if (
            preliminary_graph is not None
            and preliminary_graph.upstream_input_fingerprint
            == work_package_input_fingerprint
        ):
            snapshot_fingerprint = wiki_stage_fingerprint(
                {
                    "nodes": preliminary_graph.nodes,
                    "edges": preliminary_graph.edges,
                },
            )
            if snapshot_fingerprint == preliminary_graph.content_fingerprint:
                similarity_graph = restore_work_package_similarity_graph(
                    preliminary_graph.nodes,
                    preliminary_graph.edges,
                )
            else:
                logger.warning("Ignoring corrupt preliminary graph checkpoint")
        if similarity_graph is None:
            work_package_store = self.concept_store_provider.get_work_package_store()
            with self.progress.task(
                "embedding candidate work packages",
                len(create_plans),
            ) as progress:
                work_package_store.replace_work_packages(
                    create_plans,
                    mapping_fingerprint=work_package_input_fingerprint,
                )
                if create_plans:
                    progress.advance(len(create_plans))
            with self.progress.task(
                "retrieving work-package neighbors",
                len(create_plans),
            ) as progress:
                neighbors = retrieve_work_package_neighbors(
                    create_plans,
                    work_package_store,
                    top_k=self.config.wiki_work_package_top_k,
                    progress=progress.advance,
                )
            similarity_graph = build_work_package_similarity_graph(
                create_plans,
                neighbors,
            )
            graph_nodes, graph_edges = work_package_graph_snapshot(similarity_graph)
            graph_content_fingerprint = wiki_stage_fingerprint(
                {"nodes": graph_nodes, "edges": graph_edges},
            )
            centrality = rank_work_package_centrality(
                similarity_graph,
                alpha=self.config.wiki_graph_pagerank_alpha,
            )
            preliminary_graph = WikiPreliminaryGraphCheckpoint(
                upstream_input_fingerprint=work_package_input_fingerprint,
                nodes=list(graph_nodes),
                edges=list(graph_edges),
                content_fingerprint=graph_content_fingerprint,
                pagerank_config_fingerprint=(
                    self.preliminary_pagerank_config_fingerprint
                ),
                centrality_scores=[
                    WikiCandidatePageRank(
                        candidate_id=candidate_id,
                        pagerank=centrality[candidate_id],
                    )
                    for candidate_id in graph_nodes
                ],
            )
        else:
            if preliminary_graph is None:
                raise RuntimeError("a restored similarity graph has no checkpoint")
            if (
                preliminary_graph.pagerank_config_fingerprint
                == self.preliminary_pagerank_config_fingerprint
            ):
                centrality = {
                    item.candidate_id: item.pagerank
                    for item in preliminary_graph.centrality_scores
                }
            else:
                centrality = rank_work_package_centrality(
                    similarity_graph,
                    alpha=self.config.wiki_graph_pagerank_alpha,
                )
                preliminary_graph = preliminary_graph.model_copy(
                    update={
                        "pagerank_config_fingerprint": (
                            self.preliminary_pagerank_config_fingerprint
                        ),
                        "centrality_scores": [
                            WikiCandidatePageRank(
                                candidate_id=candidate_id,
                                pagerank=centrality[candidate_id],
                            )
                            for candidate_id in preliminary_graph.nodes
                        ],
                    },
                )
        if preliminary_graph is None:
            raise RuntimeError("the preliminary graph stage produced no checkpoint")
        checkpoint = checkpoint.model_copy(
            update={"preliminary_graph": preliminary_graph},
        )
        self.wiki_mapping_checkpoints.save_corpus(checkpoint)
        graph_edges = list(preliminary_graph.edges)
        pagerank_values = list(centrality.values())
        logger.info(
            "work_package_nodes=%d knn_edges=%d reciprocal_knn_edges=%d "
            "preliminary_pagerank_min=%.12g preliminary_pagerank_max=%.12g",
            len(preliminary_graph.nodes),
            len(graph_edges),
            sum(edge.reciprocal for edge in graph_edges),
            min(pagerank_values, default=0.0),
            max(pagerank_values, default=0.0),
        )
        return _PreliminaryPlanningState(
            checkpoint=checkpoint,
            similarity_graph=similarity_graph,
            centrality=centrality,
            input_fingerprint=work_package_input_fingerprint,
        )

    def _resolve_candidate_relations(
        self,
        create_plans: list[WikiCandidatePlan],
        candidates_by_id: dict[UUID, ConceptCandidate],
        preliminary_state: _PreliminaryPlanningState,
    ) -> _RelationPlanningState:
        """Resolve candidate relations and collapse merge groups."""
        checkpoint = preliminary_state.checkpoint
        pairs = prioritize_relation_candidates(
            preliminary_state.similarity_graph,
            preliminary_state.centrality,
        )
        plans_by_id = {plan.candidate_id: plan for plan in create_plans}
        pair_ids = [pair.candidate_ids for pair in pairs]
        relation_input_fingerprint = _relation_input_fingerprint(
            pairs,
            plans_by_id,
            candidates_by_id,
            retrieval_fingerprint=preliminary_state.input_fingerprint,
            model=self.config.wiki_relation_model,
            batch_size=self.config.wiki_relation_batch_size,
        )
        reusable_relations = _reusable_relation_resolutions(
            checkpoint,
            pair_ids,
            relation_input_fingerprint,
        )
        if checkpoint.relation_resolutions and not reusable_relations:
            logger.warning("Ignoring stale relation resolution checkpoint")
        if reusable_relations:
            reusable_candidate_ids = sorted(
                {
                    candidate_id
                    for relation in reusable_relations
                    for candidate_id in relation.candidate_ids
                },
                key=str,
            )
            record_trace_event(
                "Reuse relation resolution checkpoint",
                {
                    "stage": "relation_resolution",
                    "message": "Reuse compatible candidate relation decisions.",
                    "status": "reused",
                    "candidate_ids": [
                        str(candidate_id) for candidate_id in reusable_candidate_ids
                    ],
                    **pdf_source_metadata(
                        [
                            candidates_by_id[candidate_id].source
                            for candidate_id in reusable_candidate_ids
                        ],
                    ),
                },
            )
            logger.info(
                "Resuming relation resolution with %d checkpointed pairs",
                len(reusable_relations),
            )

        relation_state_changed = checkpoint.relation_input_fingerprint not in {
            None,
            relation_input_fingerprint,
        } or bool(checkpoint.relation_resolutions and not reusable_relations)
        relation_checkpoint = checkpoint.model_copy(
            update={
                "relation_input_fingerprint": relation_input_fingerprint,
                "relation_resolutions": reusable_relations,
                "page_definition_input_fingerprint": (
                    None
                    if relation_state_changed
                    else checkpoint.page_definition_input_fingerprint
                ),
                "page_definitions": (
                    [] if relation_state_changed else checkpoint.page_definitions
                ),
                "partition_relation_input_fingerprint": (
                    None
                    if relation_state_changed
                    else checkpoint.partition_relation_input_fingerprint
                ),
                "partition_relation_resolutions": (
                    []
                    if relation_state_changed
                    else checkpoint.partition_relation_resolutions
                ),
                "final_graph": (
                    None if relation_state_changed else checkpoint.final_graph
                ),
                "page_generation_fingerprint": (
                    None
                    if relation_state_changed
                    else checkpoint.page_generation_fingerprint
                ),
                "candidate_pages": (
                    [] if relation_state_changed else checkpoint.candidate_pages
                ),
                "page_attempts": (
                    [] if relation_state_changed else checkpoint.page_attempts
                ),
                "accepted_page_ids": (
                    [] if relation_state_changed else checkpoint.accepted_page_ids
                ),
                "finalizer_state": (
                    {} if relation_state_changed else checkpoint.finalizer_state
                ),
                "final_commit_sha": (
                    None if relation_state_changed else checkpoint.final_commit_sha
                ),
            },
        )
        self.wiki_mapping_checkpoints.save_corpus(relation_checkpoint)

        def save_relation_resolutions(
            resolved: list[WikiRelationResolution],
        ) -> None:
            self.wiki_mapping_checkpoints.save_corpus(
                relation_checkpoint.model_copy(
                    update={"relation_resolutions": resolved},
                ),
            )

        logger.info("Stage 5/10: resolving %d sparse candidate pairs", len(pairs))
        with self.progress.task(
            "resolving sparse candidate pairs",
            len(pairs),
            completed=len(reusable_relations),
        ) as progress:

            def save_and_report_relation_resolutions(
                resolved: list[WikiRelationResolution],
            ) -> None:
                save_relation_resolutions(resolved)
                progress.update(len(resolved))

            relations = (
                self.relation_resolution_workflow.run(
                    pairs,
                    plans_by_id,
                    candidates_by_id,
                    batch_size=self.config.wiki_relation_batch_size,
                    workers=self.config.wiki_relation_workers,
                    completed_relations=reusable_relations,
                    checkpoint=save_and_report_relation_resolutions,
                )
                if pairs
                else []
            )
        checkpoint = relation_checkpoint.model_copy(
            update={"relation_resolutions": relations},
        )
        logger.info(
            "resolved_merge_edges=%d resolved_link_edges=%d resolved_none_pairs=%d",
            sum(
                relation.decision is WikiRelationDecision.MERGE
                for relation in relations
            ),
            sum(
                len(relation.links)
                for relation in relations
                if relation.decision is WikiRelationDecision.LINK
            ),
            sum(
                relation.decision is WikiRelationDecision.NONE for relation in relations
            ),
        )
        relation_candidates = {pair.candidate_ids: pair for pair in pairs}
        groups, _ = collapse_merge_relations(
            create_plans,
            relations,
            relation_candidates,
        )
        return _RelationPlanningState(
            checkpoint=checkpoint,
            groups=groups,
            relations=relations,
            relation_candidates=relation_candidates,
            plans_by_id=plans_by_id,
        )

    def _compile_page_definitions(
        self,
        candidates_by_id: dict[UUID, ConceptCandidate],
        relation_state: _RelationPlanningState,
    ) -> tuple[list[WikiPageDefinition], WikiCorpusCheckpoint]:
        """Compile or restore final page definitions."""
        groups = relation_state.groups
        plans_by_id = relation_state.plans_by_id
        checkpoint = relation_state.checkpoint
        logger.info("Stage 6/10: compiling %d final page definitions", len(groups))
        relation_input_fingerprint = checkpoint.relation_input_fingerprint
        if relation_input_fingerprint is None:
            raise RuntimeError("the relation stage produced no input fingerprint")
        page_definition_input_fingerprint = _page_definition_input_fingerprint(
            groups,
            plans_by_id,
            candidates_by_id,
            relation_fingerprint=relation_input_fingerprint,
            model=self.config.wiki_page_plan_model,
            max_page_candidates=self.config.wiki_max_page_candidates,
            max_page_work_items=self.config.wiki_max_page_work_items,
        )
        reusable_definitions = _reusable_page_definitions(
            checkpoint,
            groups,
            page_definition_input_fingerprint,
        )
        reusable_candidate_ids = {
            candidate_id
            for definition in reusable_definitions
            for candidate_id in definition.member_candidate_ids
        }
        all_candidate_ids = {
            candidate_id
            for group in groups
            for candidate_id in group.member_candidate_ids
        }
        if reusable_definitions:
            definition_candidate_ids = sorted(
                {
                    candidate_id
                    for definition in reusable_definitions
                    for candidate_id in definition.member_candidate_ids
                },
                key=str,
            )
            record_trace_event(
                "Reuse page definition checkpoint",
                {
                    "stage": "page_definition",
                    "message": "Reuse compatible compiled page definitions.",
                    "status": "reused",
                    "page_ids": [
                        definition.page_id for definition in reusable_definitions
                    ],
                    "candidate_ids": [
                        str(candidate_id) for candidate_id in definition_candidate_ids
                    ],
                    **pdf_source_metadata(
                        [
                            candidates_by_id[candidate_id].source
                            for candidate_id in definition_candidate_ids
                        ],
                    ),
                },
            )
            logger.info(
                "Reusing %d compatible page definitions",
                len(reusable_definitions),
            )
        if reusable_candidate_ids == all_candidate_ids:
            definitions = reusable_definitions
        else:
            checkpoint = checkpoint.model_copy(
                update={
                    "page_definition_input_fingerprint": (
                        page_definition_input_fingerprint
                    ),
                    "page_definitions": reusable_definitions,
                    "partition_relation_input_fingerprint": None,
                    "partition_relation_resolutions": [],
                    "final_graph": None,
                    "page_generation_fingerprint": None,
                    "candidate_pages": [],
                    "page_attempts": [],
                    "accepted_page_ids": [],
                    "finalizer_state": {},
                    "final_commit_sha": None,
                },
            )
            self.wiki_mapping_checkpoints.save_corpus(checkpoint)

            def save_page_definitions(
                compiled: list[WikiPageDefinition],
            ) -> None:
                self.wiki_mapping_checkpoints.save_corpus(
                    checkpoint.model_copy(update={"page_definitions": compiled}),
                )

            with self.progress.task(
                "compiling final page definitions",
                len(groups),
            ) as progress:
                definitions = compile_page_definitions_concurrently(
                    self.page_definition_agent,
                    groups,
                    plans_by_id,
                    candidates_by_id,
                    max_page_candidates=self.config.wiki_max_page_candidates,
                    max_page_work_items=self.config.wiki_max_page_work_items,
                    workers=self.config.wiki_page_plan_workers,
                    completed_definitions=reusable_definitions,
                    checkpoint=save_page_definitions,
                    progress=progress.advance,
                )
        checkpoint = checkpoint.model_copy(
            update={
                "page_definition_input_fingerprint": (
                    page_definition_input_fingerprint
                ),
                "page_definitions": definitions,
            },
        )
        self.wiki_mapping_checkpoints.save_corpus(checkpoint)
        return definitions, checkpoint

    def _resolve_partition_boundaries(
        self,
        definitions: list[WikiPageDefinition],
        candidates_by_id: dict[UUID, ConceptCandidate],
        relation_state: _RelationPlanningState,
        checkpoint: WikiCorpusCheckpoint,
    ) -> tuple[list[WikiRelationResolution], WikiCorpusCheckpoint]:
        """Resolve MERGE decisions that a refined page boundary cuts."""
        candidate_to_page = {
            candidate_id: definition.page_id
            for definition in definitions
            for candidate_id in definition.member_candidate_ids
        }
        cut_relations = [
            relation
            for relation in relation_state.relations
            if relation.decision is WikiRelationDecision.MERGE
            and candidate_to_page[relation.candidate_ids[0]]
            != candidate_to_page[relation.candidate_ids[1]]
        ]
        cut_pairs = [
            relation_state.relation_candidates[
                relation.candidate_ids[0],
                relation.candidate_ids[1],
            ]
            for relation in cut_relations
        ]
        page_definition_fingerprint = checkpoint.page_definition_input_fingerprint
        if page_definition_fingerprint is None:
            raise RuntimeError("the page-definition stage produced no fingerprint")
        input_fingerprint = wiki_stage_fingerprint(
            {
                "page_definition_fingerprint": page_definition_fingerprint,
                "definitions": [item.model_dump(mode="json") for item in definitions],
                "cut_pairs": [item.model_dump(mode="json") for item in cut_pairs],
                "model": self.config.wiki_relation_model,
                "prompt_version": "1",
            },
        )
        reusable = []
        if checkpoint.partition_relation_input_fingerprint == input_fingerprint:
            by_pair = {
                item.candidate_ids: item
                for item in checkpoint.partition_relation_resolutions
            }
            if len(by_pair) == len(checkpoint.partition_relation_resolutions) and set(
                by_pair,
            ) == {item.candidate_ids for item in cut_pairs}:
                reusable = [by_pair[item.candidate_ids] for item in cut_pairs]
        state_changed = (
            checkpoint.partition_relation_input_fingerprint != input_fingerprint
        )
        checkpoint = checkpoint.model_copy(
            update={
                "partition_relation_input_fingerprint": input_fingerprint,
                "partition_relation_resolutions": reusable,
                "final_graph": None if state_changed else checkpoint.final_graph,
                "page_generation_fingerprint": (
                    None if state_changed else checkpoint.page_generation_fingerprint
                ),
                "candidate_pages": [] if state_changed else checkpoint.candidate_pages,
                "page_attempts": [] if state_changed else checkpoint.page_attempts,
                "accepted_page_ids": (
                    [] if state_changed else checkpoint.accepted_page_ids
                ),
                "finalizer_state": {} if state_changed else checkpoint.finalizer_state,
                "final_commit_sha": None
                if state_changed
                else checkpoint.final_commit_sha,
            },
        )
        self.wiki_mapping_checkpoints.save_corpus(checkpoint)

        def save_boundary_relations(
            resolutions: list[WikiRelationResolution],
        ) -> None:
            self.wiki_mapping_checkpoints.save_corpus(
                checkpoint.model_copy(
                    update={"partition_relation_resolutions": resolutions},
                ),
            )

        resolved = (
            self.partition_boundary_workflow.run(
                cut_pairs,
                relation_state.plans_by_id,
                candidates_by_id,
                batch_size=self.config.wiki_relation_batch_size,
                workers=self.config.wiki_relation_workers,
                completed_relations=reusable,
                checkpoint=save_boundary_relations,
            )
            if cut_pairs
            else []
        )
        checkpoint = checkpoint.model_copy(
            update={"partition_relation_resolutions": resolved},
        )
        cut_pair_ids = {item.candidate_ids for item in cut_relations}
        effective = [
            item
            for item in relation_state.relations
            if item.candidate_ids not in cut_pair_ids
        ]
        effective.extend(resolved)
        return effective, checkpoint

    def _build_final_manifest(
        self,
        definitions: list[WikiPageDefinition],
        page_relations: list[WikiPageRelation],
        checkpoint: WikiCorpusCheckpoint,
    ) -> WikiPageManifest:
        """Build, rank, save, and report the final Wiki page graph."""
        page_graph = build_wiki_page_graph(definitions, page_relations)
        fingerprints = self._final_graph_fingerprints(
            definitions,
            page_relations,
        )
        final_graph_checkpoint, manifest, global_scores = self._rank_final_graph(
            definitions,
            page_graph,
            checkpoint,
            fingerprints,
        )
        checkpoint = self._update_final_graph_checkpoint(
            checkpoint,
            final_graph_checkpoint,
            manifest,
        )
        self.wiki_mapping_checkpoints.save_corpus(checkpoint)
        self._log_final_graph(page_graph, manifest, global_scores)
        return manifest

    def _final_graph_fingerprints(
        self,
        definitions: list[WikiPageDefinition],
        page_relations: list[WikiPageRelation],
    ) -> _FinalGraphFingerprints:
        """Calculate fingerprints for the final graph and ranking inputs."""
        page_graph_metadata = _page_graph_content_metadata(
            definitions,
            page_relations,
        )
        final_graph_content_fingerprint = wiki_stage_fingerprint(
            page_graph_metadata,
        )
        final_graph_input_fingerprint = wiki_stage_fingerprint(
            {
                "definitions": [
                    definition.model_dump(mode="json")
                    for definition in sorted(
                        definitions,
                        key=lambda item: item.page_id,
                    )
                ],
                "accepted_graph": page_graph_metadata,
            },
        )
        final_ranking_fingerprint = wiki_stage_fingerprint(
            {
                "input_fingerprint": final_graph_input_fingerprint,
                "content_fingerprint": final_graph_content_fingerprint,
                "algorithm_version": GRAPH_RANKING_ALGORITHM_VERSION,
                "ranking_config": self.graph_ranking_config,
            },
        )
        return _FinalGraphFingerprints(
            input=final_graph_input_fingerprint,
            content=final_graph_content_fingerprint,
            ranking=final_ranking_fingerprint,
        )

    def _rank_final_graph(
        self,
        definitions: list[WikiPageDefinition],
        page_graph: nx.DiGraph,
        checkpoint: WikiCorpusCheckpoint,
        fingerprints: _FinalGraphFingerprints,
    ) -> tuple[WikiFinalGraphCheckpoint, WikiPageManifest, dict[str, float]]:
        """Restore or calculate final graph ranking data."""
        saved_final_graph = checkpoint.final_graph
        if (
            saved_final_graph is not None
            and saved_final_graph.input_fingerprint == fingerprints.input
            and saved_final_graph.content_fingerprint == fingerprints.content
            and saved_final_graph.ranking_fingerprint == fingerprints.ranking
        ):
            final_graph_checkpoint = saved_final_graph
            manifest = saved_final_graph.page_manifest
            global_scores = {
                item.page_id: item.pagerank
                for item in saved_final_graph.global_pagerank
            }
        else:
            global_scores = rank_global_pages(
                page_graph,
                alpha=self.config.wiki_graph_pagerank_alpha,
            )
            ranked_links = {
                page_id: rank_links_for_page(
                    page_graph,
                    page_id,
                    global_scores=global_scores,
                    config=self.graph_ranking_config,
                )
                for page_id in sorted(page_graph.nodes)
            }
            manifest = build_page_manifest(
                definitions,
                ranked_links,
                global_scores,
            )
            final_graph_checkpoint = WikiFinalGraphCheckpoint(
                input_fingerprint=fingerprints.input,
                content_fingerprint=fingerprints.content,
                ranking_fingerprint=fingerprints.ranking,
                global_pagerank=[
                    WikiPagePageRank(
                        page_id=page_id,
                        pagerank=global_scores[page_id],
                    )
                    for page_id in sorted(global_scores)
                ],
                page_manifest=manifest,
            )
        return final_graph_checkpoint, manifest, global_scores

    @staticmethod
    def _update_final_graph_checkpoint(
        checkpoint: WikiCorpusCheckpoint,
        final_graph_checkpoint: WikiFinalGraphCheckpoint,
        manifest: WikiPageManifest,
    ) -> WikiCorpusCheckpoint:
        """Save final graph data and invalidate changed writing state."""
        final_graph_changed = checkpoint.final_graph != final_graph_checkpoint
        candidate_pages = [
            WikiCandidatePageMapping(
                candidate_id=candidate_id,
                page_id=page.page_id,
                target_path=page.target_path,
            )
            for page in manifest.pages
            for candidate_id in page.member_candidate_ids
        ]
        return checkpoint.model_copy(
            update={
                "final_graph": final_graph_checkpoint,
                "candidate_pages": candidate_pages,
                "page_generation_fingerprint": (
                    None
                    if final_graph_changed
                    else checkpoint.page_generation_fingerprint
                ),
                "page_attempts": []
                if final_graph_changed
                else checkpoint.page_attempts,
                "accepted_page_ids": (
                    [] if final_graph_changed else checkpoint.accepted_page_ids
                ),
                "finalizer_state": (
                    {} if final_graph_changed else checkpoint.finalizer_state
                ),
                "final_commit_sha": (
                    None if final_graph_changed else checkpoint.final_commit_sha
                ),
            },
        )

    def _log_final_graph(
        self,
        page_graph: nx.DiGraph,
        manifest: WikiPageManifest,
        global_scores: dict[str, float],
    ) -> None:
        """Log summary metrics and the highest-ranked final pages."""
        valid_link_count = sum(len(page.links) for page in manifest.pages)
        selected_link_count = sum(
            link.selected_for_page for page in manifest.pages for link in page.links
        )
        page_count = len(manifest.pages)
        logger.info(
            "final_page_nodes=%d final_page_link_edges=%d isolated_pages=%d "
            "mean_valid_links_per_page=%.6g mean_selected_links_per_page=%.6g "
            "pages_at_link_cap=%d always_included_links=%d",
            page_count,
            page_graph.number_of_edges(),
            sum(page_graph.degree(page_id) == 0 for page_id in page_graph.nodes),
            valid_link_count / page_count if page_count else 0.0,
            selected_link_count / page_count if page_count else 0.0,
            sum(
                sum(link.selected_for_page for link in page.links)
                >= self.config.wiki_max_outgoing_links
                for page in manifest.pages
            ),
            sum(
                link.selected_for_page
                and link.semantic_score
                >= self.config.wiki_link_always_include_similarity
                for page in manifest.pages
                for link in page.links
            ),
        )
        pages_by_id = {page.page_id: page for page in manifest.pages}
        for page_id, pagerank in sorted(
            global_scores.items(),
            key=lambda item: (-item[1], item[0]),
        )[:10]:
            logger.info(
                "page_id=%s title=%s pagerank=%.12g in_degree=%d out_degree=%d",
                page_id,
                pages_by_id[page_id].title,
                pagerank,
                page_graph.in_degree(page_id),
                page_graph.out_degree(page_id),
            )
