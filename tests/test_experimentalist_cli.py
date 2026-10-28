"""Tests for the experimentalist command-line entry point."""

from __future__ import annotations

from typing import TYPE_CHECKING

from wiki_agents.orchestrator import experimentalist_cli

if TYPE_CHECKING:
    from wiki_agents.orchestrator.experimentalist import (
        ExperimentalistOrchestratorConfig,
    )


def test_cli_loads_dotenv_before_starting_orchestrator(monkeypatch) -> None:
    """Load `.env` before the orchestrator reads process environment variables."""
    events: list[str] = []

    class FakeOrchestrator:
        """Record the orchestrator lifecycle without starting a campaign."""

        def __init__(self, config: ExperimentalistOrchestratorConfig) -> None:
            _ = config
            events.append("initialize")

        def run(self) -> None:
            """Record the simulated campaign start."""
            events.append("run")

    monkeypatch.setattr(
        experimentalist_cli,
        "load_dotenv",
        lambda: events.append("load_dotenv"),
    )
    monkeypatch.setattr(
        experimentalist_cli,
        "ExperimentalistOrchestrator",
        FakeOrchestrator,
    )
    monkeypatch.setattr(experimentalist_cli, "configure_logging", lambda: None)

    result = experimentalist_cli.main(
        ["wiki", "campaign", "--task", "Test objective"],
    )

    assert result == 0
    assert events == ["load_dotenv", "initialize", "run"]


def test_cli_removes_planner_settings(monkeypatch) -> None:
    """Build campaign configuration without planner controls."""
    received: list[ExperimentalistOrchestratorConfig] = []

    class FakeOrchestrator:
        """Capture campaign configuration without running agents."""

        def __init__(self, config: ExperimentalistOrchestratorConfig) -> None:
            received.append(config)

        def run(self) -> None:
            """Skip the simulated campaign run."""

    monkeypatch.setattr(experimentalist_cli, "load_dotenv", lambda: None)
    monkeypatch.setattr(experimentalist_cli, "configure_logging", lambda: None)
    monkeypatch.setattr(
        experimentalist_cli,
        "ExperimentalistOrchestrator",
        FakeOrchestrator,
    )

    result = experimentalist_cli.main(
        [
            "wiki",
            "campaign",
            "--task",
            "Test objective",
            "--researcher-model",
            "researcher-model",
        ],
    )

    assert result == 0
    assert received[0].researcher_model == "researcher-model"
    assert not hasattr(received[0], "planner_model")
    parser = experimentalist_cli.build_parser()
    for removed in (
        "--planner-model",
        "--planner-turns",
        "--strategy-review-patience",
    ):
        assert removed not in parser.format_help()
