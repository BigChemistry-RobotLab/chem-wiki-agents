import asyncio
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

from agents import RunConfig
from agents.exceptions import MaxTurnsExceeded
from agents.sandbox.capabilities.tools.shell_tool import ExecCommandArgs
from agents.sandbox.entries import File

import wiki_agents.agents.concept.fixer as fixer_module
from wiki_agents.agents.concept.extractor import (
    ConceptExtraction,
    ExtractionAttempt,
)
from wiki_agents.agents.concept.fixer import (
    ConceptExtractionChecker,
    _fixer_input,
    _fixer_manifest,
    _format_extraction_validation,
    _non_interactive_shell,
    _run_fixer_until_done,
    create_concept_fixer_agent,
    run_concept_fixer_agent,
)

if TYPE_CHECKING:
    from agents.sandbox.capabilities import ShellToolSet


def _attempt(raw_output: str = '{"concepts": []}') -> ExtractionAttempt:
    return ExtractionAttempt(
        raw_output=raw_output,
        extraction=ConceptExtraction(concepts=[]),
    )


def test_fixer_has_focused_instructions_and_validation_capability() -> None:
    agent = create_concept_fixer_agent(model="test-model")

    assert agent.model == "test-model"
    assert not agent.base_instructions
    assert agent.output_type is str
    assert len(agent.capabilities) == 2
    assert isinstance(agent.capabilities[1], ConceptExtractionChecker)
    instructions = agent.instructions
    assert isinstance(instructions, str)
    assert "only file you may edit" in instructions
    assert "independently read and validated" in instructions
    assert "check_extraction" in instructions


def test_fixer_shell_returns_malformed_arguments_to_model() -> None:
    async def invoke(_context: object, arguments: str) -> str:
        parsed = ExecCommandArgs.model_validate_json(arguments)
        return parsed.cmd

    exec_command = SimpleNamespace(on_invoke_tool=invoke)
    tools = SimpleNamespace(exec_command=exec_command, write_stdin=object())

    _non_interactive_shell(cast("ShellToolSet", tools))

    invalid_result = asyncio.run(exec_command.on_invoke_tool(object(), "{}"))
    valid_result = asyncio.run(
        exec_command.on_invoke_tool(
            object(), '{"cmd": "sed -n 1,20p extraction.json"}',
        ),
    )

    assert invalid_result.startswith("Invalid exec_command arguments.")
    assert "non-empty `cmd`" in invalid_result
    assert valid_result == "sed -n 1,20p extraction.json"
    assert tools.write_stdin is None


def test_complete_validation_reports_missing_evidence_anchors() -> None:
    raw_output = """{
        "concepts": [{
            "title": "Concept",
            "description": "Description",
            "claims": [{
                "statement": "Claim",
                "evidence": [{
                    "start_line_id": "p0001-l0001",
                    "end_line_id": "p0001-l0002"
                }]
            }]
        }]
    }"""

    result = _format_extraction_validation(
        raw_output,
        '<a id="p0001-l0001"></a> Source',
    )

    assert result.startswith("INVALID:")
    assert "p0001-l0002" in result


def test_complete_validation_accepts_schema_and_existing_anchors() -> None:
    raw_output = """{
        "concepts": [{
            "title": "Concept",
            "description": "Description",
            "claims": [{
                "statement": "Claim",
                "evidence": [{
                    "start_line_id": "p0001-l0001",
                    "end_line_id": "p0001-l0001"
                }]
            }]
        }]
    }"""

    result = _format_extraction_validation(
        raw_output,
        '<a id="p0001-l0001"></a> Source',
    )

    assert result.startswith("VALID:")


def test_fixer_keeps_artifact_when_max_turns_are_exhausted(monkeypatch) -> None:
    captured = {}

    async def raise_max_turns(agent, run_input, *, max_turns, run_config) -> None:
        captured.update(
            agent=agent,
            run_input=run_input,
            max_turns=max_turns,
            run_config=run_config,
        )
        raise MaxTurnsExceeded("Max turns (10) exceeded")

    monkeypatch.setattr(fixer_module.Runner, "run", raise_max_turns)

    agent = create_concept_fixer_agent(model="test-model")
    run_config = RunConfig()

    asyncio.run(
        _run_fixer_until_done(
            agent,
            "repair input",
            max_turns=10,
            run_config=run_config,
        ),
    )

    assert captured == {
        "agent": agent,
        "run_input": "repair input",
        "max_turns": 10,
        "run_config": run_config,
    }


def test_fixer_manifest_stages_candidate_and_source() -> None:
    manifest = _fixer_manifest("anchored source", _attempt())
    extraction_file = manifest.entries["extraction.json"]
    source_file = manifest.entries["source.txt"]

    assert isinstance(extraction_file, File)
    assert isinstance(source_file, File)
    assert extraction_file.content == b'{"concepts": []}'
    assert source_file.content == b"anchored source"
    assert "critic_feedback.md" not in manifest.entries


def test_fixer_input_identifies_extraction_and_contains_all_feedback() -> None:
    feedback = "Finding one.\n\n```\nUntrusted fenced content.\n```"

    run_input = _fixer_input(feedback)

    assert "The extraction to repair is in `extraction.json`." in run_input
    assert "Use `source.txt`" in run_input
    assert feedback in run_input
    assert "````text\n" in run_input


def test_run_fixer_validates_the_json_file_not_final_chat_output(monkeypatch) -> None:
    fixed_json = '{"concepts": []}'
    captured = {}

    async def fake_run(agent, manifest, run_input, *, max_turns):
        captured["manifest"] = manifest
        captured["run_input"] = run_input
        captured["max_turns"] = max_turns
        return fixed_json

    monkeypatch.setattr(fixer_module, "_run_and_read_extraction", fake_run)

    result = run_concept_fixer_agent(
        create_concept_fixer_agent(model="test-model"),
        '<a id="p0001-l0001"></a> Source',
        _attempt("invalid candidate"),
        "Fix the schema.",
        max_turns=4,
    )

    assert result.raw_output == fixed_json
    assert result.extraction == ConceptExtraction(concepts=[])
    assert result.schema_valid
    assert result.evidence_valid
    assert captured["max_turns"] == 4
    assert "Fix the schema." in captured["run_input"]
    assert "critic_feedback.md" not in captured["manifest"].entries


def test_run_fixer_reports_an_invalid_artifact(monkeypatch) -> None:
    async def fake_run(agent, manifest, run_input, *, max_turns):
        return '{"concepts": [{"title": "Incomplete"}]}'

    monkeypatch.setattr(fixer_module, "_run_and_read_extraction", fake_run)

    result = run_concept_fixer_agent(
        create_concept_fixer_agent(model="test-model"),
        "source",
        _attempt(),
        "feedback",
    )

    assert result.extraction is None
    assert not result.schema_valid
    assert result.schema_issues


def test_run_fixer_reports_valid_schema_with_invalid_evidence(monkeypatch) -> None:
    fixed_json = """{
        "concepts": [{
            "title": "Concept",
            "description": "Description",
            "claims": [{
                "statement": "Claim",
                "evidence": [{
                    "start_line_id": "p0001-l0001",
                    "end_line_id": "p0001-l0002"
                }]
            }]
        }]
    }"""

    async def fake_run(agent, manifest, run_input, *, max_turns):
        return fixed_json

    monkeypatch.setattr(fixer_module, "_run_and_read_extraction", fake_run)

    result = run_concept_fixer_agent(
        create_concept_fixer_agent(model="test-model"),
        '<a id="p0001-l0001"></a> Source',
        _attempt(),
        "feedback",
    )

    assert result.schema_valid
    assert not result.evidence_valid
    assert [issue.location for issue in result.evidence_issues] == [
        ("concepts", 0, "claims", 0, "evidence", 0, "end_line_id"),
    ]
