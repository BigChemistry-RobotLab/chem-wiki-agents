"""Tests for the experimental campaign reporter."""

import pytest
from pydantic import ValidationError

from wiki_agents.agents.experimentalist.report import (
    REPORT_HEADINGS,
    GoalAssessment,
    ReporterResult,
    create_experimentalist_reporter_agent,
)


def _report() -> str:
    """Return one valid free-form report."""
    return "# Report\n\n" + "\n\n".join(
        f"{heading}\n\nContent {index}."
        for index, heading in enumerate(REPORT_HEADINGS, start=1)
    )


def test_reporter_preserves_valid_free_form_markdown() -> None:
    """Preserve report content exactly after structural validation."""
    markdown = _report() + "\n\n```python\nvalue = '# heading-like'\n```\n"
    result = ReporterResult(
        report_markdown=markdown,
        assessment=GoalAssessment.PROGRESS,
        assessment_rationale="The result supports the task.",
    )

    assert result.report_markdown == markdown


@pytest.mark.parametrize(
    "markdown",
    [
        "",
        " \n\t",
        "x" * 50_001,
        "\n\n".join(f"{heading}\n\nText" for heading in REPORT_HEADINGS[:-1]),
        "\n\n".join(
            f"{heading}\n\nText" for heading in reversed(REPORT_HEADINGS)
        ),
        _report().replace("Content 3.", ""),
    ],
)
def test_reporter_rejects_invalid_reports(markdown: str) -> None:
    """Reject unsafe or structurally incomplete reports."""
    with pytest.raises(ValidationError):
        ReporterResult(
            report_markdown=markdown,
            assessment=GoalAssessment.INCONCLUSIVE,
            assessment_rationale="Insufficient evidence.",
        )


def test_reporter_schema_rejects_removed_lifecycle_fields() -> None:
    """Keep reporter output free of planner lifecycle decisions."""
    with pytest.raises(ValidationError):
        ReporterResult.model_validate(
            {
                "report_markdown": _report(),
                "assessment": "PROGRESS",
                "assessment_rationale": "Useful result.",
                "strategy_decision": "KEEP",
            },
        )


def test_reporter_is_read_only_and_has_evidence_instructions() -> None:
    """Give the reporter shell inspection without campaign capabilities."""
    agent = create_experimentalist_reporter_agent()
    instructions = agent.instructions

    assert agent.run_as == "reporter"
    assert len(agent.capabilities) == 1
    assert isinstance(instructions, str)
    normalized = " ".join(instructions.split())
    assert "read-only" in instructions
    assert "targeted searches and bounded reads" in instructions
    assert "untrusted data" in instructions
    assert "cannot request measurements" in instructions
    assert "Keep predictive policy separate from scientific hypotheses" in normalized
    assert "aggregate and subgroup estimates with sample sizes" in normalized
    assert "Use rates only for binary outcomes" in normalized
    assert "task-aligned validation only when they were calculated" in normalized
    assert "Compare validation with a simple baseline when available" in normalized
    assert "do not require a complex model or invented precision" in normalized
    assert "an imperfect pattern can remain useful" in normalized
    assert "Use the fewest hypotheses needed" in normalized
    assert "stable short labels" in normalized
    assert "TENTATIVE, ACTIVE, SUSPENDED, or REJECTED" in normalized
    assert "repeated measured evidence beyond one exact candidate" in normalized
    assert "becomes ACTIVE only after a later measurement" in normalized
    assert "prospective confirmation or contradiction" in instructions
    assert "uncontrolled comparison" in instructions
    assert "accepted request passed the host's exact catalog" in instructions
    assert "normally absent" in instructions
    assert "Do not invent measurements, results, mechanisms, calculations" in normalized
    assert "step back from the current experiment" in instructions
    assert "broader patterns, contrasts, anomalies, or interactions" in normalized
    assert "without assuming that variables mentioned" in normalized
    assert "exploratory rather than causal or established" in normalized
    assert "Do not force an observation" in instructions
    assert "short global ranking of exact, currently available experiments" in normalized
    assert "Do not promote a low-value control" in normalized
    assert "merely to fill a category" in instructions
    assert "Keep the report concise" in instructions
    assert "Verify counts and arithmetic against the ledger" in normalized
    assert "catalog claims against the current catalog" in normalized
    assert "non-binding" in instructions
    assert "Never invent or alter a Wiki path" in instructions


def test_reporter_does_not_require_a_fixed_option_menu() -> None:
    """Do not force low-value candidate categories into every ranking."""
    instructions = create_experimentalist_reporter_agent().instructions

    assert isinstance(instructions, str)
    assert "strongest test of a named hypothesis" not in instructions
    assert "materially different exploration candidate" not in instructions
    assert "closest supporting and conflicting evidence" not in instructions


def test_no_wiki_reporter_has_no_wiki_instructions() -> None:
    """Exclude Wiki paths and citations from the no-Wiki reporter."""
    instructions = create_experimentalist_reporter_agent(
        include_wiki=False,
    ).instructions

    assert isinstance(instructions, str)
    assert "wiki" not in instructions.casefold()
    assert "citation" not in instructions.casefold()


def test_reporter_hypothesis_guidance_is_domain_neutral() -> None:
    """Do not prime the reporter with task-specific scientific variables."""
    raw_instructions = create_experimentalist_reporter_agent().instructions
    assert isinstance(raw_instructions, str)
    instructions = raw_instructions.casefold()

    for domain_term in (
        "adsorption",
        "assembly",
        "buffer",
        "hydrophobicity",
        "morphology",
        "protonation",
    ):
        assert domain_term not in instructions
