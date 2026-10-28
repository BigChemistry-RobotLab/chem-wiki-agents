"""Validate concept extraction output and source evidence."""

from __future__ import annotations

import re

from pydantic import ValidationError

from wiki_agents.concept.models import (
    ConceptExtraction,
    EvidenceIssue,
    SchemaIssue,
)
from wiki_agents.document.models import EvidenceSpan


def schema_issues(error: BaseException) -> tuple[SchemaIssue, ...]:
    """Convert a nested Pydantic failure into serializable issues.

    Args:
        error: Error raised while Pydantic validates model output.

    Returns:
        Schema issues with stable locations and error codes.

    """
    current: BaseException | None = error

    while current is not None:
        if isinstance(current, ValidationError):
            return tuple(
                SchemaIssue(
                    location=tuple(issue["loc"]),
                    code=issue["type"],
                    message=issue["msg"],
                )
                for issue in current.errors(include_url=False)
            )
        current = current.__cause__

    return (
        SchemaIssue(
            location=(),
            code="invalid_final_output",
            message=str(error),
        ),
    )


def validate_concept_extraction_json(
    raw_output: str,
) -> tuple[ConceptExtraction | None, tuple[SchemaIssue, ...]]:
    """Parse JSON against the canonical concept extraction schema.

    Args:
        raw_output: JSON text to validate.

    Returns:
        The parsed extraction and no issues, or no extraction and its issues.

    """
    try:
        return ConceptExtraction.model_validate_json(raw_output), ()
    except ValidationError as error:
        return None, schema_issues(error)


def format_schema_validation(
    extraction: ConceptExtraction | None,
    issues: tuple[SchemaIssue, ...],
) -> str:
    """Format schema validation for a model-facing tool.

    Args:
        extraction: Parsed extraction, if validation succeeded.
        issues: Schema issues from failed validation.

    Returns:
        Concise validation text for the concept fixer.

    """
    if extraction is not None:
        return "VALID: extraction.json matches the ConceptExtraction schema."

    lines = ["INVALID: extraction.json does not match the ConceptExtraction schema."]
    for issue in issues:
        location = ".".join(str(part) for part in issue.location) or "root"
        lines.append(f"- {location} [{issue.code}]: {issue.message}")
    return "\n".join(lines)


_SOURCE_ANCHOR_PATTERN = re.compile(
    r'^<a id="(?P<anchor_id>p\d{4,}-(?:l\d{4,}|(?:figure|table|region)-\d{4,}))"></a>(?:[ \t]|$)',
    re.MULTILINE,
)


def evidence_issues(
    anchored_text: str,
    extraction: ConceptExtraction,
) -> tuple[EvidenceIssue, ...]:
    """Find evidence anchors that do not exist in the source.

    Args:
        anchored_text: Source text with page and line anchors.
        extraction: Concept extraction to validate.

    Returns:
        Evidence issues with their locations in the extraction.

    """
    source_anchor_ids = {
        match.group("anchor_id")
        for match in _SOURCE_ANCHOR_PATTERN.finditer(anchored_text)
    }

    issues: list[EvidenceIssue] = []
    for concept_index, concept in enumerate(extraction.concepts):
        for claim_index, claim in enumerate(concept.claims):
            evidence_location = (
                "concepts",
                concept_index,
                "claims",
                claim_index,
                "evidence",
            )

            for evidence_index, evidence in enumerate(claim.evidence):
                item_location = (*evidence_location, evidence_index)
                anchor_fields = (
                    (
                        ("start_line_id", evidence.start_line_id),
                        ("end_line_id", evidence.end_line_id),
                    )
                    if isinstance(evidence, EvidenceSpan)
                    else (("anchor_id", evidence.anchor_id),)
                )

                for field_name, anchor_id in anchor_fields:
                    if anchor_id not in source_anchor_ids:
                        issues.append(
                            EvidenceIssue(
                                location=(*item_location, field_name),
                                message=f"Source anchor {anchor_id!r} does not exist",
                            ),
                        )

    return tuple(issues)
