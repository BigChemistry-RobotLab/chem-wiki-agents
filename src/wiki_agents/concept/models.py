from dataclasses import dataclass
from typing import Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from wiki_agents.document.models import (
    AnchoredPage,
    Evidence,
    SourceRef,
)


class Claim(BaseModel):
    """One assertion about a concept, with direct source evidence."""

    model_config = ConfigDict(extra="forbid")

    statement: str = Field(
        min_length=1,
        description="A complete assertion that keeps all applicable source qualifications.",
    )
    evidence: list[Evidence] = Field(
        min_length=1,
        description="One or more source passages that directly support the complete assertion.",
    )


class Concept(BaseModel):
    """A reusable chemistry concept in the paper."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(
        min_length=1,
        description="A short, standard title for the concept.",
    )
    description: str = Field(
        min_length=1,
        description="A concise explanation based only on the supported claims.",
    )
    claims: list[Claim] = Field(
        min_length=1,
        description="The distinct claims about the concept. Each claim has direct source evidence.",
    )


class ConceptExtraction(BaseModel):
    """The structured result of concept extraction."""

    model_config = ConfigDict(extra="forbid")

    concepts: list[Concept] = Field(
        description="The distinct, reusable chemistry concepts in the paper.",
    )


@dataclass(frozen=True, slots=True)
class SchemaIssue:
    location: tuple[str | int, ...]
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class EvidenceIssue:
    location: tuple[str | int, ...]
    message: str


class ConceptCandidate(BaseModel):
    """Add source provenance to an extracted concept."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_id: UUID
    source: SourceRef
    concept: Concept


class ConceptMatch(BaseModel):
    """Describe one related stored concept."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate: ConceptCandidate
    score: float
    concept_score: float | None = None
    claim_score: float | None = None
    matched_claim: Claim | None = None


class ConceptContext(BaseModel):
    """Keep a candidate with its related concepts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate: ConceptCandidate
    similar_concepts: list[ConceptMatch]


class ConceptExtractionRetryState(BaseModel):
    """Keep the latest reviewed candidate so a later run can resume fixing it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    raw_output: str
    extraction: ConceptExtraction | None
    critic_feedback: str


class ConceptExtractionChunk(BaseModel):
    """Record the durable extraction outcome for one source-page chunk."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str
    page_numbers: list[int] = Field(min_length=1)
    status: Literal["succeeded", "failed"]
    run_attempts: int = Field(ge=1)
    extraction: ConceptExtraction | None = None
    error_code: str | None = None
    error_message: str | None = None
    retry_state: ConceptExtractionRetryState | None = None

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        """Require success and failure records to contain matching fields."""
        if self.status == "succeeded":
            if self.extraction is None:
                raise ValueError("a succeeded chunk requires an extraction")
            if (
                self.error_code is not None
                or self.error_message is not None
                or self.retry_state is not None
            ):
                raise ValueError("a succeeded chunk cannot contain an error")
        else:
            if self.extraction is not None:
                raise ValueError("a failed chunk cannot contain an extraction")
            if self.error_code is None or self.error_message is None:
                raise ValueError("a failed chunk requires an error code and message")
        return self


class ConceptExtractionCheckpoint(BaseModel):
    """Persist resumable concept-extraction state for one source."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: Literal[1] = 1
    source: SourceRef
    chunks: list[ConceptExtractionChunk]


@dataclass(frozen=True, slots=True)
class ConceptChunkWork:
    """Describe one deterministic unit of concept-extraction work."""

    chunk_id: str
    page_numbers: list[int]
    anchored_text: str


@dataclass(frozen=True, slots=True)
class ExtractedSourceConcepts:
    """Keep one source's parsed pages and checkpointed concept results."""

    source: SourceRef
    pages: list[AnchoredPage]
    candidates: list[ConceptCandidate]
    concept_extraction_status: Literal["succeeded", "partial", "failed"]
    concept_extraction_chunks: list[ConceptExtractionChunk]
