from typing import Protocol, cast
from uuid import uuid4

import pytest
from pydantic import ValidationError

from wiki_agents.concept.models import (
    Claim,
    Concept,
    ConceptExtraction,
    ConceptExtractionChunk,
    ConceptExtractionRetryState,
)
from wiki_agents.document.models import (
    AnchoredPage,
    EvidenceSpan,
    FigureEvidence,
    RegionEvidence,
    SourceRef,
    TableEvidence,
)


class _MutableAnchoredPage(Protocol):
    text: str


def _extraction() -> ConceptExtraction:
    return ConceptExtraction(
        concepts=[
            Concept(
                title="Catalysis",
                description="A grounded description.",
                claims=[
                    Claim(
                        statement="The catalyst forms the product.",
                        evidence=[
                            EvidenceSpan(
                                start_line_id="p0004-l0012",
                                end_line_id="p0004-l0012",
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


def test_concept_extraction_accepts_grounded_nested_output() -> None:
    extraction = _extraction()

    evidence = extraction.concepts[0].claims[0].evidence[0]
    assert isinstance(evidence, EvidenceSpan)
    assert evidence.start_line_id == "p0004-l0012"
    assert evidence.end_line_id == "p0004-l0012"
    assert ConceptExtraction(concepts=[]).concepts == []


@pytest.mark.parametrize(
    ("start_line_id", "end_line_id"),
    [
        ("page-1-line-2", "p0001-l0003"),
        ("p0002-l0001", "p0001-l9999"),
    ],
)
def test_evidence_span_rejects_invalid_ranges(
    start_line_id: str,
    end_line_id: str,
) -> None:
    with pytest.raises(ValidationError):
        EvidenceSpan(start_line_id=start_line_id, end_line_id=end_line_id)


def test_claim_accepts_visual_evidence_variants() -> None:
    claim = Claim(
        statement="Visual evidence supports the result.",
        evidence=[
            FigureEvidence(
                anchor_id="p0007-figure-0002",
                label="Figure 2 on page 7",
            ),
            TableEvidence(
                anchor_id="p0008-table-0001",
                label="Table 1 on page 8",
            ),
            RegionEvidence(
                anchor_id="p0008-region-0003",
                label="Plot region on page 8",
                bbox=(0.12, 0.18, 0.84, 0.72),
            ),
        ],
    )

    assert [evidence.kind for evidence in claim.evidence] == [
        "figure",
        "table",
        "region",
    ]


@pytest.mark.parametrize(
    "bbox",
    [
        (0.5, 0.2, 0.4, 0.8),
        (0.1, 0.9, 0.8, 0.2),
        (-0.1, 0.2, 0.8, 0.9),
    ],
)
def test_region_evidence_rejects_invalid_bbox(
    bbox: tuple[float, float, float, float],
) -> None:
    with pytest.raises(ValidationError):
        RegionEvidence(
            anchor_id="p0008-region-0003",
            label="Invalid region",
            bbox=bbox,
        )


def test_concept_rejects_empty_claims_independently() -> None:
    with pytest.raises(ValidationError) as error:
        Concept(title="Catalysis", description="A grounded description.", claims=[])

    assert error.value.errors()[0]["loc"] == ("claims",)


def test_concept_rejects_empty_title_independently() -> None:
    valid_claim = _extraction().concepts[0].claims[0]

    with pytest.raises(ValidationError) as error:
        Concept(title="", description="A grounded description.", claims=[valid_claim])

    assert error.value.errors()[0]["loc"] == ("title",)


def test_concept_rejects_empty_description_independently() -> None:
    valid_claim = _extraction().concepts[0].claims[0]

    with pytest.raises(ValidationError) as error:
        Concept(title="Catalysis", description="", claims=[valid_claim])

    assert error.value.errors()[0]["loc"] == ("description",)


def test_claim_rejects_empty_statement_independently() -> None:
    valid_evidence = EvidenceSpan(
        start_line_id="p0001-l0001",
        end_line_id="p0001-l0001",
    )

    with pytest.raises(ValidationError) as error:
        Claim(statement="", evidence=[valid_evidence])

    assert error.value.errors()[0]["loc"] == ("statement",)


def test_claim_rejects_empty_evidence_independently() -> None:
    with pytest.raises(ValidationError) as error:
        Claim(statement="A supported statement.", evidence=[])

    assert error.value.errors()[0]["loc"] == ("evidence",)


def test_concept_rejects_unknown_fields_independently() -> None:
    data = _extraction().concepts[0].model_dump()
    data["source_id"] = "not-owned-by-this-agent"

    with pytest.raises(ValidationError) as error:
        Concept.model_validate(data)

    assert error.value.errors()[0]["loc"] == ("source_id",)


@pytest.mark.parametrize("sha256", ["abc", "A" * 64, "g" * 64])
def test_source_ref_rejects_invalid_sha256(sha256: str) -> None:
    with pytest.raises(ValidationError):
        SourceRef(source_id="source", sha256=sha256, resource="/references/paper.md")


def test_source_ref_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        SourceRef.model_validate(
            {
                "source_id": "source",
                "sha256": "a" * 64,
                "uri": "paper.pdf",
                "unexpected": True,
            },
        )


@pytest.mark.parametrize("page_number", [0, -1])
def test_anchored_page_requires_positive_page_number(page_number: int) -> None:
    with pytest.raises(ValidationError):
        AnchoredPage(page_number=page_number, text="page text")


def test_anchored_page_is_frozen() -> None:
    page = AnchoredPage(page_number=1, text="page text")

    with pytest.raises(ValidationError):
        cast("_MutableAnchoredPage", page).text = "changed"


def _retry_state() -> ConceptExtractionRetryState:
    return ConceptExtractionRetryState(
        raw_output="{}",
        extraction=None,
        critic_feedback="Repair it.",
    )


@pytest.mark.parametrize(
    "updates",
    [
        {"extraction": None},
        {"error_code": "error"},
        {"error_message": "failed"},
        {"retry_state": _retry_state()},
    ],
)
def test_succeeded_chunk_rejects_inconsistent_outcome_fields(
    updates: dict[str, object],
) -> None:
    values = {
        "chunk_id": "chunk-1",
        "page_numbers": [1],
        "status": "succeeded",
        "run_attempts": 1,
        "extraction": _extraction(),
    }
    values.update(updates)

    with pytest.raises(ValidationError):
        ConceptExtractionChunk.model_validate(values)


@pytest.mark.parametrize(
    "updates",
    [
        {"extraction": _extraction()},
        {"error_code": None},
        {"error_message": None},
    ],
)
def test_failed_chunk_rejects_inconsistent_outcome_fields(
    updates: dict[str, object],
) -> None:
    values = {
        "chunk_id": "chunk-1",
        "page_numbers": [1],
        "status": "failed",
        "run_attempts": 1,
        "error_code": "model_error",
        "error_message": "failed",
    }
    values.update(updates)

    with pytest.raises(ValidationError):
        ConceptExtractionChunk.model_validate(values)


@pytest.mark.parametrize(
    "updates",
    [
        {"page_numbers": []},
        {"run_attempts": 0},
    ],
)
def test_chunk_rejects_invalid_progress_metadata(
    updates: dict[str, object],
) -> None:
    values = {
        "chunk_id": str(uuid4()),
        "page_numbers": [1],
        "status": "succeeded",
        "run_attempts": 1,
        "extraction": _extraction(),
    }
    values.update(updates)

    with pytest.raises(ValidationError):
        ConceptExtractionChunk.model_validate(values)
