from collections.abc import Callable
from types import SimpleNamespace

import pytest
from agents import AgentOutputSchema, MessageOutputItem
from agents.exceptions import ModelBehaviorError
from openai.types.responses import ResponseOutputMessage, ResponseOutputText
from pydantic import ValidationError

from wiki_agents.agents.concept.extractor import (
    Claim,
    Concept,
    ConceptExtraction,
    EvidenceSpan,
    ExtractionAttempt,
    _evidence_issues,
    _last_message_text,
    create_concept_extractor_agent,
    run_concept_extractor_agent,
)
from wiki_agents.document.models import FigureEvidence


def test_agent_uses_strict_structured_output_and_preserves_model() -> None:
    agent = create_concept_extractor_agent(model="test-model")
    schema = AgentOutputSchema(ConceptExtraction).json_schema()

    assert agent.model == "test-model"
    assert agent.output_type is ConceptExtraction
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert isinstance(agent.instructions, str)
    assert "p####-l####" in agent.instructions
    assert "Every claim must have one or more evidence spans" in agent.instructions


def test_last_message_text_returns_most_recent_message() -> None:
    agent = create_concept_extractor_agent(model="test-model")

    def message(text: str, message_id: str) -> MessageOutputItem:
        return MessageOutputItem(
            agent=agent,
            raw_item=ResponseOutputMessage(
                id=message_id,
                content=[
                    ResponseOutputText(
                        annotations=[],
                        logprobs=[],
                        text=text,
                        type="output_text",
                    ),
                ],
                role="assistant",
                status="completed",
                type="message",
            ),
        )

    assert (
        _last_message_text([message("first", "msg_1"), message("last", "msg_2")])
        == "last"
    )
    assert not _last_message_text([])


def test_evidence_issues_reports_anchors_that_are_not_in_the_source() -> None:
    extraction = ConceptExtraction(
        concepts=[
            Concept(
                title="Catalysis",
                description="A supported description.",
                claims=[
                    Claim(
                        statement="A claim.",
                        evidence=[
                            EvidenceSpan(
                                start_line_id="p0001-l0002",
                                end_line_id="p0001-l0003",
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )

    issues = _evidence_issues(
        '<a id="p0001-l0001"></a> Only a real line.',
        extraction,
    )

    assert [issue.location for issue in issues] == [
        ("concepts", 0, "claims", 0, "evidence", 0, "start_line_id"),
        ("concepts", 0, "claims", 0, "evidence", 0, "end_line_id"),
    ]
    assert all("does not exist" in issue.message for issue in issues)


def test_evidence_issues_accepts_visual_anchors() -> None:
    extraction = ConceptExtraction(
        concepts=[
            Concept(
                title="Visual result",
                description="A supported description.",
                claims=[
                    Claim(
                        statement="The figure supports the result.",
                        evidence=[
                            FigureEvidence(
                                anchor_id="p0002-figure-0001",
                                label="Figure 1 on page 2",
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )

    issues = _evidence_issues(
        '<a id="p0002-figure-0001"></a> Figure 1',
        extraction,
    )

    assert issues == ()


def test_evidence_issues_allows_repeated_overlapping_and_unordered_ranges() -> None:
    extraction = ConceptExtraction(
        concepts=[
            Concept(
                title="Catalysis",
                description="A supported description.",
                claims=[
                    Claim(
                        statement="A claim.",
                        evidence=[
                            EvidenceSpan(
                                start_line_id="p0001-l0002",
                                end_line_id="p0001-l0003",
                            ),
                            EvidenceSpan(
                                start_line_id="p0001-l0001",
                                end_line_id="p0001-l0002",
                            ),
                            EvidenceSpan(
                                start_line_id="p0001-l0002",
                                end_line_id="p0001-l0003",
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )
    source = "\n".join(
        f'<a id="p0001-l{line:04d}"></a> Line {line}.' for line in range(1, 4)
    )

    issues = _evidence_issues(source, extraction)

    assert issues == ()


def test_extraction_attempt_exposes_evidence_validity() -> None:
    attempt = ExtractionAttempt(
        raw_output="{}",
        extraction=ConceptExtraction(concepts=[]),
        evidence_issues=(),
    )

    assert attempt.schema_valid
    assert attempt.evidence_valid

    invalid_attempt = ExtractionAttempt(
        raw_output="not JSON",
        extraction=None,
        evidence_issues=(),
    )
    assert not invalid_attempt.evidence_valid


def _message(agent, text: str) -> MessageOutputItem:
    return MessageOutputItem(
        agent=agent,
        raw_item=ResponseOutputMessage(
            id="msg_1",
            content=[
                ResponseOutputText(
                    annotations=[],
                    logprobs=[],
                    text=text,
                    type="output_text",
                ),
            ],
            role="assistant",
            status="completed",
            type="message",
        ),
    )


def test_run_extractor_returns_structured_output_and_raw_message(monkeypatch) -> None:
    agent = create_concept_extractor_agent(model="test-model")
    extraction = ConceptExtraction(concepts=[])
    raw_output = '{"concepts": []}'

    class Result:
        def __init__(self) -> None:
            self.new_items = [_message(agent, raw_output)]

        def final_output_as(
            self,
            output_type: object,
            *,
            raise_if_incorrect_type: bool,
        ) -> ConceptExtraction:
            assert output_type is ConceptExtraction
            assert raise_if_incorrect_type is True
            return extraction

    def fake_run_sync(run_agent, run_input, *, error_handlers):
        assert run_agent is agent
        assert run_input == "anchored source"
        assert "invalid_final_output" in error_handlers
        return Result()

    monkeypatch.setattr(
        "wiki_agents.agents.concept.extractor.Runner.run_sync", fake_run_sync,
    )

    attempt = run_concept_extractor_agent(agent, "anchored source")

    assert attempt.raw_output == raw_output
    assert attempt.extraction is extraction
    assert attempt.evidence_issues == ()


def test_run_extractor_falls_back_to_serialized_output_without_message(
    monkeypatch,
) -> None:
    agent = create_concept_extractor_agent(model="test-model")
    extraction = ConceptExtraction(concepts=[])
    result = SimpleNamespace(
        new_items=[],
        final_output_as=lambda *_args, **_kwargs: extraction,
    )
    monkeypatch.setattr(
        "wiki_agents.agents.concept.extractor.Runner.run_sync",
        lambda *_args, **_kwargs: result,
    )

    attempt = run_concept_extractor_agent(agent, "anchored source")

    assert attempt.raw_output == extraction.model_dump_json()
    assert attempt.extraction is extraction


def test_run_extractor_returns_invalid_final_output_details(monkeypatch) -> None:
    agent = create_concept_extractor_agent(model="test-model")
    raw_output = '{"concepts": [{"title": "Incomplete"}]}'

    def fake_run_sync(
        *_args: object,
        error_handlers: dict[str, Callable[[object], None]],
        **_kwargs: object,
    ) -> None:
        with pytest.raises(ValidationError) as caught:
            ConceptExtraction.model_validate_json(raw_output)
        error = ModelBehaviorError("invalid final output")
        error.__cause__ = caught.value
        error_handlers["invalid_final_output"](
            SimpleNamespace(
                error=error,
                run_data=SimpleNamespace(new_items=[_message(agent, raw_output)]),
            ),
        )
        raise error

    monkeypatch.setattr(
        "wiki_agents.agents.concept.extractor.Runner.run_sync", fake_run_sync,
    )

    attempt = run_concept_extractor_agent(agent, "anchored source")

    assert attempt.raw_output == raw_output
    assert attempt.extraction is None
    assert {issue.location for issue in attempt.schema_issues} == {
        ("concepts", 0, "description"),
        ("concepts", 0, "claims"),
    }


def test_run_extractor_propagates_unrelated_model_behavior_error(monkeypatch) -> None:
    agent = create_concept_extractor_agent(model="test-model")

    def fake_run_sync(*_args: object, **_kwargs: object) -> None:
        raise ModelBehaviorError("invalid tool call")

    monkeypatch.setattr(
        "wiki_agents.agents.concept.extractor.Runner.run_sync", fake_run_sync,
    )

    with pytest.raises(ModelBehaviorError, match="invalid tool call"):
        run_concept_extractor_agent(agent, "anchored source")


def test_run_extractor_reports_evidence_issues(monkeypatch) -> None:
    agent = create_concept_extractor_agent(model="test-model")
    extraction = ConceptExtraction(
        concepts=[
            Concept(
                title="Catalysis",
                description="A supported description.",
                claims=[
                    Claim(
                        statement="A claim.",
                        evidence=[
                            EvidenceSpan(
                                start_line_id="p0001-l0001",
                                end_line_id="p0001-l0002",
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )
    result = SimpleNamespace(
        new_items=[],
        final_output_as=lambda *_args, **_kwargs: extraction,
    )
    monkeypatch.setattr(
        "wiki_agents.agents.concept.extractor.Runner.run_sync",
        lambda *_args, **_kwargs: result,
    )

    attempt = run_concept_extractor_agent(
        agent,
        '<a id="p0001-l0001"></a> Only one line.',
    )

    assert [issue.location for issue in attempt.evidence_issues] == [
        ("concepts", 0, "claims", 0, "evidence", 0, "end_line_id"),
    ]
