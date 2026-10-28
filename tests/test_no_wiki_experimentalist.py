"""Tests for the standalone no-Wiki experimentalist command."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, cast

import httpx

from wiki_agents.agents.experimentalist.lab_tools import create_lab_capability
from wiki_agents.agents.experimentalist.research import create_experimentalist_agent
from wiki_agents.orchestrator.experimentalist import ExperimentalistOrchestratorConfig

_SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "no_wiki_experimentalist.py"
_SPEC = importlib.util.spec_from_file_location("no_wiki_experimentalist", _SCRIPT_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Could not load {_SCRIPT_PATH}")
no_wiki = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = no_wiki
_SPEC.loader.exec_module(no_wiki)

if TYPE_CHECKING:
    from agents.sandbox.entries import File


def _transport(request: httpx.Request) -> httpx.Response:
    """Serve a small lab schema and material catalog."""
    if request.url.path == "/v1/schema":
        return httpx.Response(
            200,
            json={
                "inputs": [{"name": "sample", "type": "string"}],
                "outputs": [{"name": "score", "type": "number"}],
                "materials": [{"name": "sample", "type": "string"}],
                "required_inputs": ["sample"],
                "max_batch_size": 1,
            },
        )
    if request.url.path == "/v1/materials":
        return httpx.Response(
            200,
            json={"items": [{"sample": "A"}], "total": 1, "offset": 0, "limit": 1000},
        )
    return httpx.Response(404)


def test_cli_has_no_wiki_or_planner_options() -> None:
    """Expose only shared researcher-reporter campaign controls."""
    parser = no_wiki.build_parser()
    args = parser.parse_args(["campaign", "--task", "Test objective"])

    assert not hasattr(args, "wiki_path")
    assert not hasattr(args, "planner_turns")
    assert not hasattr(args, "planner_model")
    assert not hasattr(args, "strategy_review_patience")
    assert args.researcher_turns == 30


def test_main_builds_shared_no_wiki_configuration(tmp_path, monkeypatch) -> None:
    """Use the shared orchestrator with a null Wiki path."""
    received: list[ExperimentalistOrchestratorConfig] = []
    monkeypatch.setattr(no_wiki, "load_dotenv", lambda: None)
    monkeypatch.setattr(no_wiki, "configure_logging", lambda: None)
    monkeypatch.setattr(
        no_wiki.NoWikiExperimentalistOrchestrator,
        "run",
        lambda self: received.append(self.config),
    )

    assert no_wiki.main([str(tmp_path / "campaign"), "--task", "Objective"]) == 0
    assert received[0].wiki_path is None
    assert received[0].task == "Objective"


def test_no_wiki_manifest_and_prompts_have_no_wiki_content(tmp_path) -> None:
    """Omit Wiki files and Wiki-specific researcher instructions."""
    capability = create_lab_capability(
        "http://lab.test",
        transport=httpx.MockTransport(_transport),
    )
    try:
        orchestrator = no_wiki.NoWikiExperimentalistOrchestrator(
            ExperimentalistOrchestratorConfig(
                wiki_path=None,
                campaign_path=tmp_path / "campaign",
                task="Find the best sample.",
            ),
            capability=capability,
        )
        orchestrator._load_or_initialize_campaign(capability)
        artifact = orchestrator.campaign_path / "work" / "analysis.txt"
        artifact.write_text("analysis\n", encoding="utf-8")
        manifest = capability.process_manifest(
            orchestrator._build_manifest([], capability.material_records),
        )
        paths = {str(path) for path in manifest.entries}
        instructions = create_experimentalist_agent(
            capability,
            include_wiki=False,
        ).instructions

        assert not any(PurePath(path).parts[0].casefold() == "wiki" for path in paths)
        assert "inputs/report.md" in paths
        assert "inputs/planning_context.json" in paths
        assert "work/analysis.txt" in paths
        entry = cast("File", manifest.entries["work/analysis.txt"])
        assert not entry.permissions.other & 0o2
        assert isinstance(instructions, str)
        assert "wiki" not in instructions.casefold()
        assert "advisory" in instructions
        first_message = orchestrator._researcher_input(0, 100)
        assert "plan the initial global predictive policy" in first_message
        assert "Wiki" not in first_message
        assert "Use the task and available materials" in first_message
        assert "evaluate the remaining catalog" in first_message
    finally:
        capability.client.close()
