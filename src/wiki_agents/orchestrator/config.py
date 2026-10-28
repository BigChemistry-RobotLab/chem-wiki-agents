import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

from dotenv import load_dotenv

from wiki_agents.concept.store import ConceptStore
from wiki_agents.orchestrator.utils import _nonnegative_int_env, _positive_int_env


def _finite_float_env(name: str, default: float) -> float:
    """Read a finite float from the environment.

    Args:
        name: Environment variable name.
        default: Value to use when the variable is not set.

    Returns:
        A finite float.

    Raises:
        ValueError: If the configured value is not finite.

    """
    value = float(os.getenv(name, str(default)))
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _validate_positive_finite(name: str, value: float) -> None:
    """Require a finite positive configuration value."""
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True, slots=True)
class WikiOrchestratorConfig:
    """Configure a wiki orchestration run."""

    repository_path: str | Path
    extractor_model: str
    critic_model: str
    fixer_model: str
    wiki_planner_model: str = "openai/gpt-5.6-luna"
    wiki_relation_model: str = "openai/gpt-5.6-luna"
    wiki_page_plan_model: str = "openai/gpt-5.6-luna"
    wiki_creator_model: str = "openai/gpt-5.6-luna"
    wiki_reviewer_model: str = "openai/gpt-5.6-luna"
    wiki_finalizer_model: str = "openai/gpt-5.6-luna"
    max_wiki_turns: int = 5
    max_concept_turns: int = 5
    num_extraction_workers: int = 4
    num_concept_pages: int = 3
    wiki_planning_batch_size: int = 5
    wiki_planning_workers: int = 8
    wiki_work_package_top_k: int = 20
    wiki_relation_batch_size: int = 24
    wiki_relation_workers: int = 8
    wiki_graph_pagerank_alpha: float = 0.85
    wiki_max_outgoing_links: int = 8
    wiki_link_always_include_similarity: float = 0.92
    wiki_link_semantic_weight: float = 0.50
    wiki_link_personalized_pagerank_weight: float = 0.35
    wiki_link_global_pagerank_weight: float = 0.15
    wiki_max_page_candidates: int = 12
    wiki_max_page_work_items: int = 20
    wiki_page_plan_workers: int = 8
    wiki_page_workers: int = 4
    wiki_page_transient_retries: int = 2
    wiki_finalizer_batch_size: int = 20
    wiki_finalizer_timeout_seconds: float = 900.0
    sources_path: str | Path | None = None
    concept_store: ConceptStore | None = None
    bot_name: str = "Wiki Agent"
    bot_email: str = "wiki-agent@localhost"
    similar_concepts_top_k: int = 5
    qdrant_url: str | None = None
    qdrant_path: str | Path | None = None
    qdrant_api_key: str | None = field(default=None, repr=False)
    qdrant_collection: str = "wiki-concepts"
    qdrant_work_package_collection: str = "wiki-work-packages"
    embedding_model: str = "embeddinggemma-300m"
    embedding_base_url: str | None = None
    embedding_api_key: str | None = field(default=None, repr=False)
    embedding_batch_size: int = 64
    embedding_max_input_chars: int = 1200

    def __post_init__(self) -> None:
        """Validate settings that must be positive or exclusive."""
        positive_settings = {
            "max_wiki_turns": self.max_wiki_turns,
            "max_concept_turns": self.max_concept_turns,
            "num_extraction_workers": self.num_extraction_workers,
            "num_concept_pages": self.num_concept_pages,
            "wiki_planning_batch_size": self.wiki_planning_batch_size,
            "wiki_planning_workers": self.wiki_planning_workers,
            "wiki_work_package_top_k": self.wiki_work_package_top_k,
            "wiki_relation_batch_size": self.wiki_relation_batch_size,
            "wiki_relation_workers": self.wiki_relation_workers,
            "wiki_max_outgoing_links": self.wiki_max_outgoing_links,
            "wiki_max_page_candidates": self.wiki_max_page_candidates,
            "wiki_max_page_work_items": self.wiki_max_page_work_items,
            "wiki_page_plan_workers": self.wiki_page_plan_workers,
            "wiki_page_workers": self.wiki_page_workers,
            "wiki_finalizer_batch_size": self.wiki_finalizer_batch_size,
            "similar_concepts_top_k": self.similar_concepts_top_k,
            "embedding_batch_size": self.embedding_batch_size,
            "embedding_max_input_chars": self.embedding_max_input_chars,
        }
        for name, value in positive_settings.items():
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if self.wiki_page_transient_retries < 0:
            raise ValueError("wiki_page_transient_retries must not be negative")
        _validate_positive_finite(
            "wiki_finalizer_timeout_seconds",
            self.wiki_finalizer_timeout_seconds,
        )
        if not math.isfinite(self.wiki_graph_pagerank_alpha) or not (
            0 < self.wiki_graph_pagerank_alpha < 1
        ):
            raise ValueError("wiki_graph_pagerank_alpha must be between 0 and 1")
        ranking_weights = {
            "wiki_link_semantic_weight": self.wiki_link_semantic_weight,
            "wiki_link_personalized_pagerank_weight": (
                self.wiki_link_personalized_pagerank_weight
            ),
            "wiki_link_global_pagerank_weight": (self.wiki_link_global_pagerank_weight),
        }
        for name, value in ranking_weights.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not math.isclose(
            sum(ranking_weights.values()),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("Wiki link ranking weights must sum to 1.0")
        if not math.isfinite(self.wiki_link_always_include_similarity) or not (
            0 <= self.wiki_link_always_include_similarity <= 1
        ):
            raise ValueError(
                "wiki_link_always_include_similarity must be between 0 and 1",
            )
        if self.qdrant_url and self.qdrant_path:
            raise ValueError("QDRANT_URL and QDRANT_PATH are mutually exclusive")

    @classmethod
    def from_env(
        cls,
        repository_path: str | Path,
        *,
        sources_path: str | Path | None = None,
    ) -> Self:
        """Create configuration from environment variables and `.env`.

        Args:
            repository_path: Path to the local wiki repository.
            sources_path: Optional path to the source PDF directory.

        Returns:
            Configuration with all environment settings resolved.

        """
        load_dotenv()
        return cls(
            repository_path=repository_path,
            sources_path=sources_path,
            extractor_model=os.getenv("EXTRACTOR_MODEL", "openai/gpt-5.6-luna"),
            critic_model=os.getenv("CRITIC_MODEL", "openai/gpt-5.6-luna"),
            fixer_model=os.getenv("FIXER_MODEL", "openai/gpt-5.6-luna"),
            wiki_planner_model=os.getenv(
                "WIKI_PLANNER_MODEL",
                "openai/gpt-5.6-luna",
            ),
            wiki_creator_model=os.getenv(
                "WIKI_CREATOR_MODEL",
                "openai/gpt-5.6-luna",
            ),
            wiki_reviewer_model=os.getenv(
                "WIKI_REVIEWER_MODEL",
                "openai/gpt-5.6-luna",
            ),
            wiki_relation_model=os.getenv(
                "WIKI_RELATION_MODEL",
                "openai/gpt-5.6-luna",
            ),
            wiki_page_plan_model=os.getenv(
                "WIKI_PAGE_PLAN_MODEL",
                "openai/gpt-5.6-luna",
            ),
            wiki_finalizer_model=os.getenv(
                "WIKI_FINALIZER_MODEL",
                "openai/gpt-5.6-luna",
            ),
            max_wiki_turns=_positive_int_env("WIKI_MAX_TURNS", 5),
            max_concept_turns=_positive_int_env("CONCEPT_MAX_TURNS", 5),
            num_extraction_workers=_positive_int_env("WIKI_EXTRACTION_WORKERS", 4),
            num_concept_pages=_positive_int_env("WIKI_NUM_CONCEPT_PAGES", 3),
            wiki_planning_batch_size=_positive_int_env(
                "WIKI_PLANNING_BATCH_SIZE",
                5,
            ),
            wiki_planning_workers=_positive_int_env("WIKI_PLANNING_WORKERS", 8),
            wiki_work_package_top_k=_positive_int_env(
                "WIKI_WORK_PACKAGE_TOP_K",
                20,
            ),
            wiki_relation_batch_size=_positive_int_env(
                "WIKI_RELATION_BATCH_SIZE",
                24,
            ),
            wiki_relation_workers=_positive_int_env("WIKI_RELATION_WORKERS", 8),
            wiki_graph_pagerank_alpha=_finite_float_env(
                "WIKI_GRAPH_PAGERANK_ALPHA",
                0.85,
            ),
            wiki_max_outgoing_links=_positive_int_env(
                "WIKI_MAX_OUTGOING_LINKS",
                8,
            ),
            wiki_link_always_include_similarity=_finite_float_env(
                "WIKI_LINK_ALWAYS_INCLUDE_SIMILARITY",
                0.92,
            ),
            wiki_link_semantic_weight=_finite_float_env(
                "WIKI_LINK_SEMANTIC_WEIGHT",
                0.50,
            ),
            wiki_link_personalized_pagerank_weight=_finite_float_env(
                "WIKI_LINK_PERSONALIZED_PAGERANK_WEIGHT",
                0.35,
            ),
            wiki_link_global_pagerank_weight=_finite_float_env(
                "WIKI_LINK_GLOBAL_PAGERANK_WEIGHT",
                0.15,
            ),
            wiki_max_page_candidates=_positive_int_env(
                "WIKI_MAX_PAGE_CANDIDATES",
                12,
            ),
            wiki_max_page_work_items=_positive_int_env(
                "WIKI_MAX_PAGE_WORK_ITEMS",
                20,
            ),
            wiki_page_plan_workers=_positive_int_env("WIKI_PAGE_PLAN_WORKERS", 8),
            wiki_page_workers=_positive_int_env("WIKI_PAGE_WORKERS", 4),
            wiki_page_transient_retries=_nonnegative_int_env(
                "WIKI_PAGE_TRANSIENT_RETRIES",
                2,
            ),
            wiki_finalizer_batch_size=_positive_int_env(
                "WIKI_FINALIZER_BATCH_SIZE",
                20,
            ),
            wiki_finalizer_timeout_seconds=_finite_float_env(
                "WIKI_FINALIZER_TIMEOUT_SECONDS",
                900.0,
            ),
            bot_name=os.getenv("WIKI_GIT_AUTHOR_NAME", "Wiki Agent"),
            bot_email=os.getenv("WIKI_GIT_AUTHOR_EMAIL", "wiki-agent@localhost"),
            similar_concepts_top_k=_positive_int_env(
                "WIKI_SIMILAR_CONCEPTS_TOP_K",
                5,
            ),
            qdrant_url=os.getenv("QDRANT_URL"),
            qdrant_path=os.getenv("QDRANT_PATH"),
            qdrant_api_key=os.getenv("QDRANT_API_KEY"),
            qdrant_collection=os.getenv("QDRANT_COLLECTION", "wiki-concepts"),
            qdrant_work_package_collection=os.getenv(
                "QDRANT_WORK_PACKAGE_COLLECTION",
                "wiki-work-packages",
            ),
            embedding_model=os.getenv(
                "EMBEDDING_DEFAULT_MODEL",
                "embeddinggemma-300m",
            ),
            embedding_base_url=os.getenv("EMBEDDING_BASE_URL"),
            embedding_api_key=os.getenv("EMBEDDING_API_KEY"),
            embedding_batch_size=_positive_int_env("EMBEDDING_BATCH_SIZE", 64),
            embedding_max_input_chars=_positive_int_env(
                "EMBEDDING_MAX_INPUT_CHARS",
                1200,
            ),
        )
