import logging
from pathlib import Path
from types import TracebackType
from typing import Self

import pymupdf

from wiki_agents.agents.wiki.aggregator import create_wiki_page_definition_agent
from wiki_agents.agents.wiki.graph_ranking import GRAPH_RANKING_ALGORITHM_VERSION
from wiki_agents.agents.wiki.models import WikiGraphRankingConfig
from wiki_agents.agents.wiki.workflow import (
    WikiCandidatePlanningWorkflow,
    WikiFinalizationWorkflow,
    WikiPageWorkflow,
    WikiPartitionBoundaryWorkflow,
    WikiRelationResolutionWorkflow,
)
from wiki_agents.concept.config import ConceptExtractionConfig
from wiki_agents.concept.models import ExtractedSourceConcepts
from wiki_agents.concept.pipeline import ConceptExtractionPipeline
from wiki_agents.concept.workflow import ConceptExtractionWorkflow
from wiki_agents.embedding.formats import QUERY_FORMAT, WORK_PACKAGE_FORMAT
from wiki_agents.orchestrator.concept_store import (
    ConceptStoreProvider,
    ConceptStoreSettings,
)
from wiki_agents.orchestrator.config import WikiOrchestratorConfig
from wiki_agents.orchestrator.ingestion import PaperIngestionService
from wiki_agents.orchestrator.mapping_checkpoints import (
    WikiMappingCheckpointStore,
    wiki_stage_fingerprint,
)
from wiki_agents.orchestrator.wiki_checkpoint_policy import (
    WIKI_CORPUS_CHECKPOINT_KEY_VERSION,
    WIKI_MAPPING_PROMPT_VERSION,
    WIKI_MAPPING_SCHEMA_VERSION,
    WIKI_PAGE_DEFINITION_PROMPT_VERSION,
    WIKI_PAGE_DEFINITION_SCHEMA_VERSION,
    WIKI_RELATION_PROMPT_VERSION,
    WIKI_RELATION_SCHEMA_VERSION,
    _page_definition_input_fingerprint,
    _page_graph_content_metadata,
    _relation_input_fingerprint,
    _reusable_candidate_plans,
    _reusable_page_definitions,
    _reusable_page_outcomes,
    _reusable_relation_resolutions,
)
from wiki_agents.orchestrator.wiki_extraction import WikiExtractionMixin
from wiki_agents.orchestrator.wiki_outcomes import WikiCorpusOutcome
from wiki_agents.orchestrator.wiki_planning import WikiPlanningMixin
from wiki_agents.orchestrator.wiki_publication import WikiPublicationMixin
from wiki_agents.progress import ProgressReporter
from wiki_agents.repository import BotIdentity, WikiRepository
from wiki_agents.tracing import init_weave_tracing, traced_workflow

__all__ = [
    "WIKI_CORPUS_CHECKPOINT_KEY_VERSION",
    "WIKI_MAPPING_PROMPT_VERSION",
    "WIKI_MAPPING_SCHEMA_VERSION",
    "WIKI_PAGE_DEFINITION_PROMPT_VERSION",
    "WIKI_PAGE_DEFINITION_SCHEMA_VERSION",
    "WIKI_RELATION_PROMPT_VERSION",
    "WIKI_RELATION_SCHEMA_VERSION",
    "ExtractedSourceConcepts",
    "WikiCorpusOutcome",
    "WikiOrchestrator",
    "WikiOrchestratorConfig",
    "_page_definition_input_fingerprint",
    "_page_graph_content_metadata",
    "_relation_input_fingerprint",
    "_reusable_candidate_plans",
    "_reusable_page_definitions",
    "_reusable_page_outcomes",
    "_reusable_relation_resolutions",
]

logger = logging.getLogger("wiki_agents.orchestrator")


class WikiOrchestrator(
    WikiPlanningMixin,
    WikiPublicationMixin,
    WikiExtractionMixin,
):
    def __init__(
        self,
        config: WikiOrchestratorConfig,
        *,
        progress: bool = False,
    ) -> None:
        """Create an orchestrator from one configuration object."""
        self.config = config
        self.progress = ProgressReporter(enabled=progress)
        configured_sources_path = (
            Path(config.sources_path).expanduser().resolve()
            if config.sources_path is not None
            else None
        )
        if configured_sources_path is not None and not configured_sources_path.is_dir():
            raise ValueError(
                f"Sources path {configured_sources_path} is not a directory",
            )

        self.concept_workflow = ConceptExtractionWorkflow(
            extractor_model=config.extractor_model,
            critic_model=config.critic_model,
            fixer_model=config.fixer_model,
            max_revision_turns=config.max_concept_turns,
        )
        self.candidate_planning_workflow = WikiCandidatePlanningWorkflow(
            model=config.wiki_planner_model,
        )
        self.relation_resolution_workflow = WikiRelationResolutionWorkflow(
            model=config.wiki_relation_model,
        )
        self.partition_boundary_workflow = WikiPartitionBoundaryWorkflow(
            model=config.wiki_relation_model,
        )
        self.page_definition_agent = create_wiki_page_definition_agent(
            model=config.wiki_page_plan_model,
        )
        self.page_workflow = WikiPageWorkflow(
            creator_model=config.wiki_creator_model,
            reviewer_model=config.wiki_reviewer_model,
            max_correction_runs=config.max_wiki_turns,
            transient_retries=config.wiki_page_transient_retries,
        )
        self.finalization_workflow = WikiFinalizationWorkflow(
            model=config.wiki_finalizer_model,
            timeout_seconds=config.wiki_finalizer_timeout_seconds,
        )

        self.wiki_repository = WikiRepository(
            config.repository_path,
            identity=BotIdentity(config.bot_name, config.bot_email),
        )
        self.repository_path = self.wiki_repository.path
        self.sources_path = configured_sources_path or self.repository_path / "sources"
        self.checkpoints_path = self.repository_path / ".checkpoints"
        self.checkpoints_path.mkdir(parents=True, exist_ok=True)
        self.staging_path = self.repository_path / ".staging"
        self.staging_path.mkdir(parents=True, exist_ok=True)

        self.concept_store_provider = ConceptStoreProvider(
            ConceptStoreSettings(
                repository_path=self.repository_path,
                qdrant_url=config.qdrant_url,
                qdrant_path=config.qdrant_path,
                qdrant_api_key=config.qdrant_api_key,
                qdrant_collection=config.qdrant_collection,
                qdrant_work_package_collection=config.qdrant_work_package_collection,
                embedding_model=config.embedding_model,
                embedding_base_url=config.embedding_base_url,
                embedding_api_key=config.embedding_api_key,
                embedding_batch_size=config.embedding_batch_size,
                embedding_max_input_chars=config.embedding_max_input_chars,
            ),
            store=config.concept_store,
        )

        self.concept_pipeline = ConceptExtractionPipeline(
            ConceptExtractionConfig(
                checkpoints_path=self.checkpoints_path,
                extractor_model=config.extractor_model,
                critic_model=config.critic_model,
                fixer_model=config.fixer_model,
                num_concept_pages=config.num_concept_pages,
                document_parser=self._parse_document,
                concept_extractor=self.concept_workflow.run,
                concept_store_provider=self.concept_store_provider.get,
                similar_concepts_top_k=config.similar_concepts_top_k,
                parser_version=f"pymupdf:{pymupdf.VersionBind}",
            ),
        )
        self.candidate_planning_config_metadata = {
            "model": config.wiki_planner_model,
            "batch_size": config.wiki_planning_batch_size,
            "prompt_version": WIKI_MAPPING_PROMPT_VERSION,
            "schema_version": WIKI_MAPPING_SCHEMA_VERSION,
        }
        self.work_package_retrieval_config_metadata = {
            "embedding_model": config.embedding_model,
            "work_package_format": WORK_PACKAGE_FORMAT,
            "query_format": QUERY_FORMAT,
            "embedding_max_input_chars": config.embedding_max_input_chars,
            "top_k": config.wiki_work_package_top_k,
        }
        self.preliminary_pagerank_config_fingerprint = wiki_stage_fingerprint(
            {
                "alpha": config.wiki_graph_pagerank_alpha,
                "algorithm_version": GRAPH_RANKING_ALGORITHM_VERSION,
            },
        )
        self.graph_ranking_config = WikiGraphRankingConfig(
            alpha=config.wiki_graph_pagerank_alpha,
            max_outgoing_links=config.wiki_max_outgoing_links,
            always_include_similarity=(config.wiki_link_always_include_similarity),
            semantic_weight=config.wiki_link_semantic_weight,
            personalized_pagerank_weight=(
                config.wiki_link_personalized_pagerank_weight
            ),
            global_pagerank_weight=config.wiki_link_global_pagerank_weight,
        )
        self.page_generation_config_fingerprint = wiki_stage_fingerprint(
            {
                "creator_model": config.wiki_creator_model,
                "reviewer_model": config.wiki_reviewer_model,
                "prompt_version": WIKI_MAPPING_PROMPT_VERSION,
                "schema_version": WIKI_MAPPING_SCHEMA_VERSION,
            },
        )
        self.wiki_mapping_checkpoints = WikiMappingCheckpointStore(
            self.checkpoints_path,
            wiki_stage_fingerprint(
                {
                    "checkpoint_key_version": WIKI_CORPUS_CHECKPOINT_KEY_VERSION,
                },
            ),
        )
        self.ingestion = PaperIngestionService(
            repository=self.wiki_repository,
            pipeline=self.concept_pipeline,
            staging_path=self.staging_path,
        )

    def __enter__(self) -> Self:
        """Return this orchestrator for use as a context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release resources owned by this orchestrator."""
        self.close()

    def close(self) -> None:
        """Close resources that this orchestrator owns."""
        self.concept_store_provider.close()

    def run(self) -> WikiCorpusOutcome | None:
        """Run all corpus stages in sequence."""
        init_weave_tracing()
        with traced_workflow(
            "Build complete wiki corpus",
            metadata={
                "stage": "full_run",
                "message": "Extract sources, plan the graph, and publish the wiki.",
                "repository_path": str(self.repository_path),
                "pipeline_fingerprint": self.concept_pipeline.fingerprint,
            },
        ) as trace_metadata:
            extracted_sources = self.extract_concepts()
            trace_metadata["pdf_source_ids"] = [
                extracted.source.source_id
                for extracted in extracted_sources
                if hasattr(extracted, "source")
            ]
            if not extracted_sources:
                trace_metadata["status"] = "no_sources"
                logger.info(
                    "Workflow complete: no extracted concepts require Wiki mapping",
                )
                return None
            if not any(extracted.candidates for extracted in extracted_sources):
                trace_metadata["status"] = "no_candidates"
                logger.info(
                    "Workflow complete: extraction produced no concept candidates",
                )
                return None
            manifest = self.create_plans(extracted_sources)
            if manifest is None:
                trace_metadata["status"] = "no_pages"
                logger.info(
                    "Workflow complete: no candidates require Wiki representation",
                )
                return WikiCorpusOutcome((), None)
            outcome = self.write_wiki(extracted_sources)
            trace_metadata.update(
                status="completed" if outcome.commit_sha is not None else "failed",
                page_count=len(outcome.pages),
                commit_sha=outcome.commit_sha,
            )
            return outcome
