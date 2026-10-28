"""Persist durable Wiki mapping outcomes."""

import json
from hashlib import sha256
from pathlib import Path
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from wiki_agents.agents.wiki.models import (
    WikiCandidatePlan,
    WikiPageDefinition,
    WikiPageManifest,
    WikiPageRetryState,
    WikiRelationResolution,
    WorkPackageGraphEdge,
)

_CORPUS_CHECKPOINT_VERSION = 4

UnitFloat = Annotated[FiniteFloat, Field(ge=0.0, le=1.0)]


class WikiCandidatePageMapping(BaseModel):
    """Map one original candidate to its fixed final page path."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_id: UUID
    page_id: str = Field(min_length=1)
    target_path: str = Field(pattern=r"^wiki/concepts/.+\.md$")


class WikiPageAttempt(BaseModel):
    """Record one page-centric workflow outcome."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    page_id: str = Field(min_length=1)
    attempts: int = Field(ge=1)
    accepted: bool
    last_error: str | None = None
    retry_state: WikiPageRetryState | None = None

    @model_validator(mode="after")
    def validate_error(self) -> Self:
        """Require errors only for failed attempts."""
        if self.accepted and self.last_error is not None:
            raise ValueError("an accepted page attempt cannot contain an error")
        if self.accepted and self.retry_state is not None:
            raise ValueError("an accepted page attempt cannot contain retry state")
        if not self.accepted and not self.last_error:
            raise ValueError("a failed page attempt requires an error")
        return self


class WikiCandidatePageRank(BaseModel):
    """Store structural centrality for one work-package candidate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_id: UUID
    pagerank: UnitFloat = Field(
        description="Structural graph centrality, not scientific importance.",
    )


class WikiPagePageRank(BaseModel):
    """Store structural centrality for one final Wiki page."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    page_id: str = Field(min_length=1)
    pagerank: UnitFloat = Field(
        description="Structural graph centrality, not scientific importance.",
    )


class WikiPreliminaryGraphCheckpoint(BaseModel):
    """Persist the canonical sparse graph and its preliminary PageRank."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    upstream_input_fingerprint: str = Field(min_length=1)
    nodes: list[UUID]
    edges: list[WorkPackageGraphEdge]
    content_fingerprint: str = Field(min_length=1)
    pagerank_config_fingerprint: str = Field(min_length=1)
    centrality_scores: list[WikiCandidatePageRank]

    @model_validator(mode="after")
    def validate_snapshot(self) -> Self:
        """Require complete graph state in stable node and edge order."""
        expected_nodes = sorted(self.nodes, key=str)
        if self.nodes != expected_nodes or len(self.nodes) != len(set(self.nodes)):
            raise ValueError("preliminary graph nodes must be sorted and unique")
        node_ids = set(self.nodes)
        edge_pairs = [edge.candidate_ids for edge in self.edges]
        expected_edges = sorted(edge_pairs, key=lambda pair: tuple(map(str, pair)))
        if edge_pairs != expected_edges or len(edge_pairs) != len(set(edge_pairs)):
            raise ValueError("preliminary graph edges must be sorted and unique")
        if any(not set(pair).issubset(node_ids) for pair in edge_pairs):
            raise ValueError("preliminary graph edges must reference graph nodes")
        centrality_ids = [item.candidate_id for item in self.centrality_scores]
        if centrality_ids != expected_nodes:
            raise ValueError(
                "preliminary PageRank scores must match sorted graph nodes",
            )
        return self


class WikiFinalGraphCheckpoint(BaseModel):
    """Persist the accepted page graph and complete ranked manifest."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    input_fingerprint: str = Field(min_length=1)
    content_fingerprint: str = Field(min_length=1)
    ranking_fingerprint: str = Field(min_length=1)
    global_pagerank: list[WikiPagePageRank]
    page_manifest: WikiPageManifest

    @model_validator(mode="after")
    def validate_scores(self) -> Self:
        """Require one stable global PageRank score for every final page."""
        page_ids = sorted(page.page_id for page in self.page_manifest.pages)
        score_ids = [item.page_id for item in self.global_pagerank]
        if score_ids != page_ids:
            raise ValueError(
                "global PageRank scores must match sorted manifest pages",
            )
        return self


class WikiCorpusCheckpoint(BaseModel):
    """Persist resumable full-corpus stages without publishing partial pages."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: Literal[4] = 4
    candidate_planning_fingerprint: str | None = None
    candidate_plans: list[WikiCandidatePlan] = Field(default_factory=list)
    preliminary_graph: WikiPreliminaryGraphCheckpoint | None = None
    relation_input_fingerprint: str | None = None
    relation_resolutions: list[WikiRelationResolution] = Field(default_factory=list)
    page_definition_input_fingerprint: str | None = None
    page_definition_ids: list[str] = Field(default_factory=list)
    page_definitions: list[WikiPageDefinition] = Field(
        default_factory=list,
        exclude=True,
    )
    partition_relation_input_fingerprint: str | None = None
    partition_relation_resolutions: list[WikiRelationResolution] = Field(
        default_factory=list,
    )
    final_graph: WikiFinalGraphCheckpoint | None = None
    page_generation_fingerprint: str | None = None
    candidate_pages: list[WikiCandidatePageMapping] = Field(default_factory=list)
    page_attempts: list[WikiPageAttempt] = Field(default_factory=list)
    accepted_page_ids: list[str] = Field(default_factory=list)
    finalizer_state: dict[str, object] = Field(default_factory=dict)
    final_commit_sha: str | None = None


class WikiMappingCheckpointStore:
    """Load and atomically save Wiki mapping checkpoints."""

    def __init__(
        self,
        checkpoints_path: Path,
        corpus_fingerprint: str,
    ) -> None:
        """Create a checkpoint store for one Wiki configuration.

        Args:
            checkpoints_path: Root directory for ignored checkpoint files.
            corpus_fingerprint: Stable key for the staged corpus checkpoint.

        """
        self._checkpoints_path = checkpoints_path
        self._corpus_fingerprint = corpus_fingerprint

    def corpus_path(self) -> Path:
        """Return the checkpoint path for the full-corpus mapping state."""
        directory = (
            self._checkpoints_path / "wiki" / "corpus" / self._corpus_fingerprint
        )
        directory.mkdir(parents=True, exist_ok=True)
        return directory / "state.json"

    def load_corpus(self) -> WikiCorpusCheckpoint:
        """Load the full-corpus checkpoint or return empty state."""
        path = self.corpus_path()
        if not path.exists():
            return WikiCorpusCheckpoint()
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise TypeError(f"Wiki corpus checkpoint must be an object in {path}")
        version = payload.get("version")
        if version != _CORPUS_CHECKPOINT_VERSION:
            raise ValueError(f"Unsupported Wiki corpus checkpoint version in {path}")
        checkpoint = WikiCorpusCheckpoint.model_validate(payload)
        definitions = [
            self.load_page_plan(page_id) for page_id in checkpoint.page_definition_ids
        ]
        return checkpoint.model_copy(update={"page_definitions": definitions})

    def save_corpus(self, checkpoint: WikiCorpusCheckpoint) -> None:
        """Save full-corpus state with an atomic file replacement."""
        definitions = sorted(checkpoint.page_definitions, key=lambda item: item.page_id)
        self._write_page_plans(definitions)
        checkpoint = checkpoint.model_copy(
            update={
                "page_definition_ids": [item.page_id for item in definitions],
                "page_definitions": definitions,
            },
        )
        path = self.corpus_path()
        temporary_path = path.with_suffix(".json.tmp")
        temporary_path.write_text(checkpoint.model_dump_json(indent=2) + "\n")
        temporary_path.replace(path)
        self._remove_stale_page_plans({item.page_id for item in definitions})

    def page_plans_path(self) -> Path:
        """Return the directory for individual page definition plans."""
        return self.corpus_path().parent / "page-plans"

    def load_page_plan(self, page_id: str) -> WikiPageDefinition:
        """Load one page definition plan by its stable page ID."""
        path = self.page_plans_path() / f"{page_id}.json"
        if not path.is_file():
            raise ValueError(f"Wiki page plan is missing: {path}")
        return WikiPageDefinition.model_validate_json(path.read_text())

    def _write_page_plans(self, definitions: list[WikiPageDefinition]) -> None:
        """Write page plans before the state starts to reference them."""
        directory = self.page_plans_path()
        directory.mkdir(parents=True, exist_ok=True)
        for definition in definitions:
            path = directory / f"{definition.page_id}.json"
            temporary_path = path.with_suffix(".json.tmp")
            temporary_path.write_text(definition.model_dump_json(indent=2) + "\n")
            temporary_path.replace(path)

    def _remove_stale_page_plans(self, expected_ids: set[str]) -> None:
        """Remove unreferenced plans after the new state is durable."""
        directory = self.page_plans_path()
        expected_paths = {directory / f"{page_id}.json" for page_id in expected_ids}
        for stale_path in set(directory.glob("*.json")) - expected_paths:
            stale_path.unlink()

    def page_attempts_path(self, generation_fingerprint: str) -> Path:
        """Return the directory for one generation's page attempt checkpoints."""
        return self.corpus_path().parent / "page-attempts" / generation_fingerprint

    def load_page_attempts(
        self,
        generation_fingerprint: str,
    ) -> list[WikiPageAttempt]:
        """Load page attempts for one compatible page generation."""
        directory = self.page_attempts_path(generation_fingerprint)
        return [
            WikiPageAttempt.model_validate_json(path.read_text())
            for path in sorted(directory.glob("*.json"))
        ]

    def save_page_attempt(
        self,
        generation_fingerprint: str,
        attempt: WikiPageAttempt,
    ) -> Path:
        """Atomically save one completed page attempt."""
        directory = self.page_attempts_path(generation_fingerprint)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{attempt.page_id}.json"
        temporary_path = path.with_suffix(".json.tmp")
        temporary_path.write_text(attempt.model_dump_json(indent=2) + "\n")
        temporary_path.replace(path)
        return path

    def save_page_proposal(
        self,
        generation_fingerprint: str,
        page_id: str,
        content: bytes,
    ) -> Path:
        """Save accepted page Markdown for one compatible generation."""
        directory = (
            self.corpus_path().parent / "page-proposals" / generation_fingerprint
        )
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{page_id}.md"
        temporary_path = path.with_suffix(".md.tmp")
        temporary_path.write_bytes(content)
        temporary_path.replace(path)
        return path

    def load_page_proposal(
        self,
        generation_fingerprint: str,
        page_id: str,
    ) -> bytes | None:
        """Load one accepted page proposal for a compatible generation."""
        path = (
            self.corpus_path().parent
            / "page-proposals"
            / generation_fingerprint
            / f"{page_id}.md"
        )
        return path.read_bytes() if path.is_file() else None


def wiki_stage_fingerprint(metadata: object) -> str:
    """Return a stable fingerprint for one independently reusable stage."""
    encoded = json.dumps(
        metadata,
        default=_fingerprint_json_value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return sha256(encoded).hexdigest()


def _fingerprint_json_value(value: object) -> object:
    """Convert supported structured values to canonical JSON data."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, UUID | Path):
        return str(value)
    if isinstance(value, set | frozenset):
        return sorted(value, key=str)
    raise TypeError(f"Unsupported fingerprint metadata type: {type(value).__name__}")
