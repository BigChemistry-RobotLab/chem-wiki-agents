"""Define plans and results used by the wiki agents."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Annotated, Self
from uuid import (
    UUID,  # ruff: ignore[typing-only-standard-library-import] - Pydantic needs this type at runtime.
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    field_validator,
    model_validator,
)

from wiki_agents.concept.models import (  # ruff: ignore[typing-only-first-party-import] - Pydantic resolves this model at runtime.
    ConceptCandidate,
)

_MIN_WIKI_PATH_PARTS = 3
_RESERVED_WIKI_NAMES = {"index.md", "log.md"}

type UnitFloat = Annotated[FiniteFloat, Field(ge=0.0, le=1.0)]


def _is_concept_path(path: PurePosixPath, *, workspace_path: bool) -> bool:
    """Return whether a path identifies a non-reserved concept document."""
    expected_prefix = ("wiki", "concepts") if workspace_path else ("/", "concepts")
    return (
        ".." not in path.parts
        and path.parts[:2] == expected_prefix
        and len(path.parts) >= _MIN_WIKI_PATH_PARTS
        and path.suffix.lower() == ".md"
        and path.name.lower() not in _RESERVED_WIKI_NAMES
    )


class WikiPlanDecision(StrEnum):
    """Describe the action that the wiki planner selected."""

    SKIP = "SKIP"
    CREATE = "CREATE"
    UPDATE = "UPDATE"


class WikiCandidateDecision(StrEnum):
    """Describe whether a candidate has knowledge for the generated Wiki."""

    SKIP = "SKIP"
    CREATE = "CREATE"


class WikiPageIntent(BaseModel):
    """Describe the normalized subject and work from one candidate."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    canonical_subject: str = Field(min_length=1)
    scope: str = Field(min_length=1)
    work_summary: list[Annotated[str, Field(min_length=1)]] = Field(min_length=1)


class WikiCandidatePlan(BaseModel):
    """Keep one candidate decision without assigning a page or path."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    candidate_id: UUID
    decision: WikiCandidateDecision
    rationale: str = Field(min_length=1)
    page_intent: WikiPageIntent | None = None

    @model_validator(mode="after")
    def validate_decision_fields(self) -> Self:
        """Require an intent only for a candidate that must be represented."""
        if self.decision is WikiCandidateDecision.SKIP:
            if self.page_intent is not None:
                raise ValueError("a SKIP candidate plan cannot have a page intent")
        elif self.page_intent is None:
            raise ValueError("a CREATE candidate plan requires a page intent")
        return self


class WikiCandidateBatchPlan(BaseModel):
    """Return one independent result for each candidate in a request batch."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    items: list[WikiCandidatePlan] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_candidate_ids(self) -> Self:
        """Reject duplicate candidate results."""
        candidate_ids = [item.candidate_id for item in self.items]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("a candidate batch cannot repeat a candidate ID")
        return self


class WikiRelationDecision(StrEnum):
    """Describe the page-level relationship between two work packages."""

    MERGE = "MERGE"
    LINK = "LINK"
    NONE = "NONE"


class WorkPackageNeighbor(BaseModel):
    """Keep one directed top-K work-package observation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_candidate_id: UUID
    target_candidate_id: UUID
    similarity_score: FiniteFloat

    @model_validator(mode="after")
    def reject_self_observation(self) -> Self:
        """Reject a work-package observation that points to itself."""
        if self.source_candidate_id == self.target_candidate_id:
            raise ValueError("a work-package neighbor cannot reference itself")
        return self


class WorkPackageGraphEdge(BaseModel):
    """Keep one canonical undirected work-package graph edge."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_ids: tuple[UUID, UUID]
    raw_similarity: FiniteFloat
    similarity: UnitFloat
    reciprocal: bool

    @model_validator(mode="after")
    def validate_candidate_pair(self) -> Self:
        """Require distinct candidate IDs in stable order."""
        first, second = self.candidate_ids
        if first == second or str(first) > str(second):
            raise ValueError("graph edges require sorted distinct candidate IDs")
        return self


class WikiRelationCandidate(BaseModel):
    """Keep one unique unordered work-package pair for model resolution."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_ids: tuple[UUID, UUID]
    similarity_score: FiniteFloat
    ranking_similarity: UnitFloat = 0.0
    reciprocal: bool = False

    @model_validator(mode="after")
    def validate_pair(self) -> Self:
        """Require two distinct IDs in stable order."""
        first, second = self.candidate_ids
        if first == second or str(first) > str(second):
            raise ValueError("relation candidates require sorted distinct IDs")
        return self


class WikiDirectedRelation(BaseModel):
    """Describe one directed candidate link and its meaning."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    source_candidate_id: UUID
    target_candidate_id: UUID
    relationship: str = Field(min_length=1)

    @model_validator(mode="after")
    def reject_self_relation(self) -> Self:
        """Reject a candidate link to itself."""
        if self.source_candidate_id == self.target_candidate_id:
            raise ValueError("a directed relation cannot link a candidate to itself")
        return self


class WikiRelationResolution(BaseModel):
    """Keep one typed result for an unordered candidate pair."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    candidate_ids: tuple[UUID, ...] = Field(min_length=2, max_length=2)
    decision: WikiRelationDecision
    rationale: str = Field(min_length=1)
    links: list[WikiDirectedRelation] = Field(default_factory=list, max_length=2)

    @model_validator(mode="after")
    def validate_pair_and_links(self) -> Self:
        """Require sorted distinct IDs and links only for LINK results."""
        first, second = self.candidate_ids
        if first == second:
            raise ValueError("a relation pair must contain two candidates")
        if str(first) > str(second):
            raise ValueError("relation candidate IDs must use stable sorted order")
        if self.decision is WikiRelationDecision.LINK:
            if not self.links:
                raise ValueError("a LINK relation requires directed link intent")
            pair = {first, second}
            if any(
                {link.source_candidate_id, link.target_candidate_id} != pair
                for link in self.links
            ):
                raise ValueError("directed links must connect the resolved pair")
        elif self.links:
            raise ValueError("only a LINK relation can contain directed links")
        return self


class WikiRelationBatch(BaseModel):
    """Return relation results for one bounded pair batch."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    items: list[WikiRelationResolution] = Field(min_length=1)


class WikiPageDefinition(BaseModel):
    """Define one final page before a target path is assigned."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    page_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    scope: str = Field(min_length=1)
    work_summary: list[Annotated[str, Field(min_length=1)]] = Field(min_length=1)
    member_candidate_ids: list[UUID] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_members(self) -> Self:
        """Require stable unique page members."""
        if len(self.member_candidate_ids) != len(set(self.member_candidate_ids)):
            raise ValueError("a page definition cannot repeat a candidate ID")
        return self


class WikiPageDefinitionDraft(BaseModel):
    """Define one proposed page before code assigns its stable identity."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    title: str = Field(min_length=1)
    scope: str = Field(min_length=1)
    work_summary: list[Annotated[str, Field(min_length=1)]] = Field(min_length=1)
    member_candidate_ids: list[UUID] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_members(self) -> Self:
        """Require unique page members."""
        if len(self.member_candidate_ids) != len(set(self.member_candidate_ids)):
            raise ValueError("a page definition draft cannot repeat a candidate ID")
        return self


class WikiPageDefinitionBatch(BaseModel):
    """Partition one provisional merge group into one or more page drafts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    items: list[WikiPageDefinitionDraft] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_members(self) -> Self:
        """Require each candidate to occur in only one page draft."""
        members = [
            candidate_id
            for item in self.items
            for candidate_id in item.member_candidate_ids
        ]
        if len(members) != len(set(members)):
            raise ValueError("page definition drafts cannot share a candidate ID")
        return self


class WikiCandidatePairContribution(BaseModel):
    """Keep one directed candidate LINK contribution to a page edge."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_candidate_id: UUID
    target_candidate_id: UUID
    similarity: UnitFloat

    @model_validator(mode="after")
    def reject_self_contribution(self) -> Self:
        """Reject a contribution that points to the same candidate."""
        if self.source_candidate_id == self.target_candidate_id:
            raise ValueError("a candidate contribution cannot reference itself")
        return self


def candidate_contribution_sort_key(
    contribution: WikiCandidatePairContribution,
) -> tuple[str, str]:
    """Return the stable key for one directed candidate contribution."""
    return str(contribution.source_candidate_id), str(contribution.target_candidate_id)


class WikiPageRelation(BaseModel):
    """Keep one collapsed accepted directed page relation."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    source_page_id: str = Field(min_length=1)
    target_page_id: str = Field(min_length=1)
    relationships: tuple[Annotated[str, Field(min_length=1)], ...] = Field(min_length=1)
    semantic_score: UnitFloat
    contributing_candidate_pairs: tuple[WikiCandidatePairContribution, ...] = Field(
        min_length=1,
    )

    @model_validator(mode="after")
    def validate_relation(self) -> Self:
        """Require distinct pages and stable unique relation metadata."""
        if self.source_page_id == self.target_page_id:
            raise ValueError("a page relation cannot link a page to itself")
        if tuple(sorted(set(self.relationships))) != self.relationships:
            raise ValueError("page relationships must use stable unique order")
        ordered = tuple(
            sorted(
                set(self.contributing_candidate_pairs),
                key=candidate_contribution_sort_key,
            ),
        )
        if ordered != self.contributing_candidate_pairs:
            raise ValueError("page contributions must use stable unique order")
        return self


class WikiGraphRankingConfig(BaseModel):
    """Configure deterministic structural ranking and accepted-link selection."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    alpha: FiniteFloat = Field(default=0.85, gt=0.0, lt=1.0)
    max_outgoing_links: int = Field(default=8, ge=1)
    always_include_similarity: UnitFloat = 0.92
    semantic_weight: FiniteFloat = Field(default=0.50, ge=0.0)
    personalized_pagerank_weight: FiniteFloat = Field(default=0.35, ge=0.0)
    global_pagerank_weight: FiniteFloat = Field(default=0.15, ge=0.0)

    @model_validator(mode="after")
    def validate_weight_sum(self) -> Self:
        """Require ranking weights to form one finite convex combination."""
        total = (
            self.semantic_weight
            + self.personalized_pagerank_weight
            + self.global_pagerank_weight
        )
        if not math.isclose(total, 1.0, abs_tol=1e-9, rel_tol=0.0):
            raise ValueError("Wiki link ranking weights must sum to 1.0")
        return self


class RankedWikiLink(BaseModel):
    """Keep ranking metadata for one accepted direct outgoing page link."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    target_page_id: str = Field(min_length=1)
    relationships: tuple[Annotated[str, Field(min_length=1)], ...] = Field(min_length=1)
    semantic_score: UnitFloat
    global_pagerank: UnitFloat = Field(
        description=(
            "Structural graph centrality for ranking; this is not scientific importance."
        ),
    )
    personalized_pagerank: UnitFloat = Field(
        description=(
            "Source-personalized structural centrality; this is not scientific "
            "importance."
        ),
    )
    ranking_score: UnitFloat
    selected_for_page: bool
    contributing_candidate_pairs: tuple[WikiCandidatePairContribution, ...] = Field(
        min_length=1,
    )

    @model_validator(mode="after")
    def validate_metadata_order(self) -> Self:
        """Require stable unique relationship and contribution metadata."""
        if tuple(sorted(set(self.relationships))) != self.relationships:
            raise ValueError("ranked relationships must use stable unique order")
        ordered = tuple(
            sorted(
                set(self.contributing_candidate_pairs),
                key=candidate_contribution_sort_key,
            ),
        )
        if ordered != self.contributing_candidate_pairs:
            raise ValueError("ranked contributions must use stable unique order")
        return self


class WikiManifestLink(BaseModel):
    """Describe one accepted final page-to-page link and its ranking metadata."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    target_page_id: str = Field(min_length=1)
    target_path: str
    relationships: list[Annotated[str, Field(min_length=1)]] = Field(min_length=1)
    semantic_score: UnitFloat
    global_pagerank: UnitFloat = Field(
        description=(
            "Structural graph centrality for ranking; this is not scientific importance."
        ),
    )
    personalized_pagerank: UnitFloat = Field(
        description=(
            "Source-personalized structural centrality; this is not scientific "
            "importance."
        ),
    )
    ranking_score: UnitFloat
    selected_for_page: bool
    contributing_candidate_pairs: list[WikiCandidatePairContribution] = Field(
        min_length=1,
    )

    @field_validator("target_path", mode="before")
    @classmethod
    def validate_target_path(cls, value: object) -> str:
        """Require a safe bundle-root concept target."""
        if not isinstance(value, str):
            raise TypeError("target_path must be a string")
        path = PurePosixPath(value)
        if not path.is_absolute() or not _is_concept_path(path, workspace_path=False):
            raise ValueError("target_path must be a non-reserved /concepts/*.md path")
        return path.as_posix()

    @model_validator(mode="after")
    def validate_relation_metadata(self) -> Self:
        """Require stable unique relation text and candidate provenance."""
        if sorted(set(self.relationships)) != self.relationships:
            raise ValueError("manifest relationships must use stable unique order")
        ordered = sorted(
            set(self.contributing_candidate_pairs),
            key=candidate_contribution_sort_key,
        )
        if ordered != self.contributing_candidate_pairs:
            raise ValueError("manifest contributions must use stable unique order")
        return self


class WikiManifestPage(BaseModel):
    """Describe one path-fixed page in the complete page manifest."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    page_id: str = Field(min_length=1)
    target_path: str
    bundle_path: str
    title: str = Field(min_length=1)
    scope: str = Field(min_length=1)
    work_summary: list[Annotated[str, Field(min_length=1)]] = Field(min_length=1)
    member_candidate_ids: list[UUID] = Field(min_length=1)
    global_pagerank: UnitFloat = Field(
        description=(
            "Structural graph centrality for pipeline ranking; this is not scientific "
            "importance."
        ),
    )
    links: list[WikiManifestLink] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_page(self) -> Self:
        """Require matching paths and unique candidates and link targets."""
        target = PurePosixPath(self.target_path)
        if target.is_absolute() or not _is_concept_path(target, workspace_path=True):
            raise ValueError("target_path must be below wiki/concepts/")
        expected_bundle = "/" + self.target_path.removeprefix("wiki/")
        if self.bundle_path != expected_bundle:
            raise ValueError("bundle_path does not match target_path")
        if len(self.member_candidate_ids) != len(set(self.member_candidate_ids)):
            raise ValueError("a manifest page cannot repeat a candidate ID")
        targets = [link.target_path for link in self.links]
        if len(targets) != len(set(targets)):
            raise ValueError("a manifest page cannot repeat a link target")
        if self.bundle_path in targets:
            raise ValueError("a manifest page cannot link to itself")
        return self


class WikiPageManifest(BaseModel):
    """Keep the immutable target and link graph for a corpus build."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pages: list[WikiManifestPage]

    @model_validator(mode="after")
    def validate_unique_pages(self) -> Self:
        """Reject duplicate page IDs and target paths."""
        page_ids = [page.page_id for page in self.pages]
        target_paths = [page.target_path for page in self.pages]
        if len(page_ids) != len(set(page_ids)):
            raise ValueError("a page manifest cannot repeat a page ID")
        if len(target_paths) != len(set(target_paths)):
            raise ValueError("a page manifest cannot repeat a target path")
        return self


class WikiPageJob(BaseModel):
    """Bind one final Wiki plan to all member candidate IDs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    page_id: str = Field(min_length=1)
    plan: WikiPlan
    member_candidate_ids: list[UUID] = Field(min_length=1)


class WikiSelectedLinkContext(BaseModel):
    """Give a page worker one selected direct outgoing link."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    target_page_id: str = Field(min_length=1)
    target_path: str
    target_title: str = Field(min_length=1)
    relationship: str = Field(min_length=1)


class WikiPageContext(BaseModel):
    """Keep a page job with original candidates and selected links."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    page_job: WikiPageJob
    candidates: list[ConceptCandidate] = Field(min_length=1)
    selected_links: list[WikiSelectedLinkContext] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_members(self) -> Self:
        """Require the page job to contain exactly its original candidates."""
        actual = [candidate.candidate_id for candidate in self.candidates]
        expected = self.page_job.member_candidate_ids
        if actual != expected:
            raise ValueError("page candidates do not match the page job members")
        return self


class WikiLinkPlan(BaseModel):
    """Describe one outbound link that the creator must add."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    target_path: str
    relationship: str = Field(min_length=1)

    @field_validator("target_path", mode="before")
    @classmethod
    def validate_target_path(cls, value: object) -> str:
        """Normalize a safe concept link to its bundle-root path."""
        if not isinstance(value, str):
            raise TypeError("target_path must be a string")
        if value.startswith("wiki/concepts/"):
            value = "/" + value.removeprefix("wiki/")
        elif value.startswith("concepts/"):
            value = f"/{value}"
        path = PurePosixPath(value)
        if not path.is_absolute() or not _is_concept_path(path, workspace_path=False):
            raise ValueError("target_path must be a non-reserved /concepts/*.md path")
        return path.as_posix()


class WikiPlan(BaseModel):
    """Give the editor one validated wiki action."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    decision: WikiPlanDecision
    rationale: str = Field(min_length=1)
    target_path: str | None = None
    title: str | None = Field(default=None, min_length=1)
    work_summary: list[Annotated[str, Field(min_length=1)]] = Field(
        default_factory=list,
    )
    links: list[WikiLinkPlan] = Field(default_factory=list)

    @field_validator("target_path", mode="before")
    @classmethod
    def validate_target_path(cls, value: object) -> str | None:
        """Normalize a safe concept target to its workspace path."""
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("target_path must be a string")
        if value.startswith("/concepts/"):
            value = f"wiki{value}"
        elif value.startswith("concepts/"):
            value = f"wiki/{value}"
        path = PurePosixPath(value)
        if path.is_absolute() or not _is_concept_path(path, workspace_path=True):
            raise ValueError(
                "target_path must be a non-reserved Markdown file below wiki/concepts/",
            )
        return path.as_posix()

    @model_validator(mode="after")
    def validate_decision_fields(self) -> Self:
        """Require fields that match the selected action."""
        if self.decision is WikiPlanDecision.SKIP:
            if self.target_path is not None or self.title is not None:
                raise ValueError("a SKIP plan cannot have a target path or title")
            if self.work_summary:
                raise ValueError("a SKIP plan cannot have work items")
            if self.links:
                raise ValueError("a SKIP plan cannot have links")
            return self

        if self.target_path is None or self.title is None or not self.title.strip():
            raise ValueError("a CREATE or UPDATE plan requires a target path and title")
        if not self.work_summary:
            raise ValueError("a CREATE or UPDATE plan requires one or more work items")
        link_targets = [link.target_path for link in self.links]
        if len(link_targets) != len(set(link_targets)):
            raise ValueError(
                "a plan cannot contain the same link target more than once",
            )
        return self


class WikiChangeStatus(StrEnum):
    """Describe one wiki file change made in the sandbox."""

    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"


@dataclass(frozen=True, slots=True)
class WikiFileChange:
    """Keep one file change and its resulting content."""

    path: str
    status: WikiChangeStatus
    content: bytes | None


@dataclass(frozen=True, slots=True)
class WikiCreationResult:
    """Return the plan and sandbox wiki changes."""

    plan: WikiPlan
    changes: tuple[WikiFileChange, ...]
    raw_output: str


class WikiReviewDecision(StrEnum):
    """Describe whether the proposed wiki changes can be accepted."""

    ACCEPT = "ACCEPT"
    REVISE = "REVISE"


class WikiReviewFindingType(StrEnum):
    """Classify one required Wiki correction."""

    PLAN = "PLAN"
    FACTUAL = "FACTUAL"
    SOURCE = "SOURCE"
    OKF = "OKF"
    LINK = "LINK"
    SCOPE = "SCOPE"


class WikiReviewFinding(BaseModel):
    """Describe one required correction to proposed wiki changes."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    finding_type: WikiReviewFindingType
    target: str = Field(min_length=1)
    problem: str = Field(min_length=1)
    required_change: str = Field(min_length=1)
    basis: str = Field(min_length=1)


class WikiPageRetryState(BaseModel):
    """Keep a valid page proposal for review or a later correction run."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        ser_json_bytes="base64",
        val_json_bytes="base64",
    )

    content: bytes
    raw_output: str
    findings: list[WikiReviewFinding] = Field(default_factory=list)
    correction_runs: int = Field(ge=0)


class WikiReview(BaseModel):
    """Return the review decision and required corrections."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    decision: WikiReviewDecision
    summary: str = Field(min_length=1)
    findings: list[WikiReviewFinding] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_decision(self) -> Self:
        """Require findings only when the editor must revise the files."""
        if self.decision is WikiReviewDecision.ACCEPT and self.findings:
            raise ValueError("an ACCEPT review cannot have findings")
        if self.decision is WikiReviewDecision.REVISE and not self.findings:
            raise ValueError("a REVISE review requires one or more findings")
        return self
