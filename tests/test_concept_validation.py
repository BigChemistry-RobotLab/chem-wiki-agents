import pytest
from pydantic import ValidationError

from wiki_agents.agents.concept.validation import (
    format_schema_validation,
    schema_issues,
    validate_concept_extraction_json,
)
from wiki_agents.concept.models import ConceptExtraction, SchemaIssue


def test_shared_schema_validator_accepts_a_concept_extraction() -> None:
    extraction, issues = validate_concept_extraction_json('{"concepts": []}')

    assert extraction == ConceptExtraction(concepts=[])
    assert issues == ()
    assert format_schema_validation(extraction, issues).startswith("VALID:")


def test_shared_schema_validator_returns_actionable_locations() -> None:
    extraction, issues = validate_concept_extraction_json(
        '{"concepts": [{"title": "Incomplete"}]}',
    )

    assert extraction is None
    assert {issue.location for issue in issues} == {
        ("concepts", 0, "description"),
        ("concepts", 0, "claims"),
    }
    output = format_schema_validation(extraction, issues)
    assert output.startswith("INVALID:")
    assert "concepts.0.description [missing]" in output
    assert "concepts.0.claims [missing]" in output


def test_schema_issues_unwraps_nested_validation_error() -> None:
    with pytest.raises(ValidationError) as caught:
        ConceptExtraction.model_validate({"concepts": [{}]})
    wrapped_error = RuntimeError("SDK wrapper")
    wrapped_error.__cause__ = caught.value

    issues = schema_issues(wrapped_error)

    assert {issue.location for issue in issues} == {
        ("concepts", 0, "title"),
        ("concepts", 0, "description"),
        ("concepts", 0, "claims"),
    }
    assert {issue.code for issue in issues} == {"missing"}


def test_schema_issues_falls_back_for_non_validation_error() -> None:
    error = RuntimeError("model returned unusable output")

    assert schema_issues(error) == (
        SchemaIssue(
            location=(),
            code="invalid_final_output",
            message="model returned unusable output",
        ),
    )
