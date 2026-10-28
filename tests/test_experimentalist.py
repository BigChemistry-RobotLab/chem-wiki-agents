"""Tests for report-centered experimental campaign state."""

from __future__ import annotations

import asyncio
import io
import json
import tarfile
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import httpx
import pytest

import wiki_agents.orchestrator.experimentalist as experimentalist_module
from wiki_agents.agents.experimentalist.lab_tools import (
    LabCapability,
    create_lab_capability,
)
from wiki_agents.agents.experimentalist.research import (
    _finish_after_accepted_request,
    create_experimentalist_agent,
)
from wiki_agents.orchestrator.experimentalist import (
    ExperimentalistOrchestrator,
    ExperimentalistOrchestratorConfig,
)

if TYPE_CHECKING:
    from collections.abc import Coroutine, Iterator
    from pathlib import Path

    from agents import FunctionToolResult
    from agents.run_context import RunContextWrapper
    from agents.sandbox import Manifest
    from agents.sandbox.entries import File
    from agents.sandbox.session.base_sandbox_session import BaseSandboxSession


def _transport(request: httpx.Request) -> httpx.Response:
    """Serve a two-row lab configuration."""
    if request.url.path == "/v1/schema":
        return httpx.Response(
            200,
            json={
                "inputs": [
                    {"name": "sequence", "type": "string"},
                    {"name": "ph", "type": "number"},
                ],
                "outputs": [{"name": "score", "type": "number"}],
                "materials": [
                    {"name": "sequence", "type": "string"},
                    {"name": "ph", "type": "number"},
                ],
                "required_inputs": ["sequence", "ph"],
                "max_batch_size": 2,
            },
        )
    if request.url.path == "/v1/materials":
        return httpx.Response(
            200,
            json={
                "items": [
                    {"sequence": "A", "ph": 3.0},
                    {"sequence": "B", "ph": 7.0},
                ],
                "total": 2,
                "offset": 0,
                "limit": 1000,
            },
        )
    return httpx.Response(404)


@pytest.fixture
def campaign(
    tmp_path: Path,
) -> Iterator[tuple[ExperimentalistOrchestrator, LabCapability]]:
    """Create initialized campaign state and its lab capability.

    Yields:
        The orchestrator and configured lab capability.

    """
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    (wiki / "knowledge.md").write_text("# Knowledge\n", encoding="utf-8")
    config = ExperimentalistOrchestratorConfig(
        wiki_path=wiki,
        campaign_path=tmp_path / "campaign",
        task="Find high-scoring experiments.",
        request_budget=2,
    )
    capability = create_lab_capability(
        "http://lab.test",
        transport=httpx.MockTransport(_transport),
    )
    orchestrator = ExperimentalistOrchestrator(config, capability=capability)
    orchestrator._load_or_initialize_campaign(capability)
    yield orchestrator, capability
    capability.client.close()


def test_initialization_is_deterministic_and_has_reduced_artifacts(campaign) -> None:
    """Create the central report without planner artifacts or model calls."""
    orchestrator, _ = campaign
    report = (orchestrator.campaign_path / "report.md").read_text(encoding="utf-8")

    assert report == orchestrator._initial_report(orchestrator.config.task)
    assert not (orchestrator.campaign_path / "strategy.json").exists()
    assert [line for line in report.splitlines() if line.startswith("## ")] == [
        "## Objective and progress",
        "## Predictive evidence and current policy",
        "## Scientific hypotheses and limitations",
        "## General Observations",
        "## Next ranked options and open questions",
    ]
    assert "No predictive evidence or policy yet." in report
    assert "No scientific hypotheses or limitations yet." in report
    assert "No observations yet." in report


def test_first_researcher_message_uses_wiki_for_initial_policy(campaign) -> None:
    """Ask the first researcher to plan an initial Wiki-informed policy."""
    orchestrator, _ = campaign

    message = orchestrator._researcher_input(0, 2)

    assert "plan the initial global predictive policy" in message
    assert "Use the Wiki, task, and available materials" in message
    assert "evaluate the remaining catalog" in message


def test_later_researcher_message_continues_the_campaign(campaign) -> None:
    """Keep the continuation instruction after a valid experiment exists."""
    orchestrator, _ = campaign

    message = orchestrator._researcher_input(1, 1)

    assert message.startswith("Continue the immutable objective")
    assert "initial global predictive policy" not in message
    assert "Update the global predictive policy from all evidence" in message


def test_manifest_has_shared_evidence_and_no_planner_inputs(campaign) -> None:
    """Expose all campaign evidence and keep work writable only by researchers."""
    orchestrator, capability = campaign
    artifact = orchestrator.campaign_path / "work" / "analysis" / "trend.py"
    artifact.parent.mkdir()
    artifact.write_text("print('analysis')\n", encoding="utf-8")
    manifest = capability.process_manifest(
        orchestrator._build_manifest([], capability.material_records),
    )
    paths = {str(path) for path in manifest.entries}

    assert [user.name for user in manifest.users] == ["researcher", "reporter"]
    assert {
        "inputs/task.md",
        "inputs/report.md",
        "inputs/planning_context.json",
        "inputs/observations.jsonl",
        "inputs/available_materials.csv",
        "wiki/knowledge.md",
        "work/analysis/trend.py",
    } <= paths
    assert "inputs/strategy.md" not in paths
    assert "inputs/measurement_contract.json" not in paths
    entry = cast("File", manifest.entries["work/analysis/trend.py"])
    assert entry.group is not None
    assert entry.group.name == "researcher"
    assert entry.permissions.group & 0o2
    assert not entry.permissions.other & 0o2


def test_session_manifest_does_not_disclose_workspace_files(campaign) -> None:
    """Keep the model-facing session manifest data-minimal."""
    orchestrator, _ = campaign
    manifest = orchestrator._build_session_manifest()

    assert manifest.entries == {}
    assert [user.name for user in manifest.users] == ["researcher", "reporter"]


def test_workspace_materialization_archives_wiki(campaign, monkeypatch) -> None:
    """Materialize Wiki files without putting them in the session manifest."""
    orchestrator, capability = campaign
    manifest = capability.process_manifest(
        orchestrator._build_manifest([], capability.material_records),
    )
    session = SimpleNamespace()
    applied: list[Manifest] = []
    writes: list[tuple[Path, bytes]] = []

    async def apply_manifest(received, *, manifest, provision_accounts) -> None:
        _ = received, provision_accounts
        applied.append(manifest)

    async def write(path, data) -> None:
        writes.append((path, data.read()))

    async def execute(*args: object, **kwargs: object):
        _ = args, kwargs
        return SimpleNamespace(ok=lambda: True, stderr=b"")

    async def remove(path) -> None:
        _ = path

    session.write = write
    session.exec = execute
    session.rm = remove
    monkeypatch.setattr(experimentalist_module.manifest_ops, "apply_manifest", apply_manifest)

    asyncio.run(
        orchestrator._materialize_workspace(
            cast("BaseSandboxSession", session),
            manifest,
        ),
    )

    assert "wiki/knowledge.md" not in applied[0].entries
    with tarfile.open(fileobj=io.BytesIO(writes[0][1]), mode="r:") as archive:
        extracted = archive.extractfile("wiki/knowledge.md")
        assert extracted is not None
        assert extracted.read() == b"# Knowledge\n"


def test_accepted_batch_persists_only_reporter_neutral_fields(campaign) -> None:
    """Record a measurement without strategy revision metadata."""
    orchestrator, _ = campaign
    orchestrator._start_curation_group("test-group")
    requests = [{"sequence": "A", "ph": 3.0}]
    response = {"results": [{"request": requests[0], "found": True, "values": {"score": 4}}]}

    orchestrator._record_accepted(requests, "Tests local transfer.", response)

    record = orchestrator._read_observations()[0]
    assert record["reasoning"] == "Tests local transfer."
    assert "selected_by_strategy_revision" not in record
    pending = orchestrator._require_pending_collation()
    assert pending["phase"] == "measurement_recorded"
    assert pending["observation_indices"] == [0]


def test_resume_rejects_planner_era_campaign(campaign) -> None:
    """Require a new directory when planner-era state is present."""
    orchestrator, capability = campaign
    (orchestrator.campaign_path / "strategy.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"planner-era.*new campaign directory"):
        orchestrator._load_or_initialize_campaign(capability)


def test_resume_rejects_immutable_task_change(campaign) -> None:
    """Keep the task immutable across resumed runs."""
    orchestrator, capability = campaign
    changed = ExperimentalistOrchestrator(
        ExperimentalistOrchestratorConfig(
            wiki_path=orchestrator.wiki_path,
            campaign_path=orchestrator.campaign_path,
            task="Different task.",
            request_budget=2,
        ),
        capability=capability,
    )

    with pytest.raises(ValueError, match="resume mismatch for task"):
        changed._load_or_initialize_campaign(capability)


def test_researcher_owns_strategy_and_explains_requests(campaign) -> None:
    """Require a global predictive policy and an honest request rationale."""
    _, capability = campaign
    instructions = create_experimentalist_agent(capability).instructions

    assert isinstance(instructions, str)
    assert "own the evolving experimental strategy" in instructions.casefold()
    assert "advisory" in instructions
    assert "Evaluate the remaining catalog globally" in instructions
    normalized = " ".join(instructions.split())
    assert "Prioritize immediate expected progress" in normalized
    assert "aggregate evidence" in instructions
    assert "simple baseline" in instructions
    assert "held-out observations" in normalized
    assert "could leak information" in instructions
    assert "With sparse data, prefer transparent rules" in normalized
    assert "Interpret evidence probabilistically" in instructions
    assert "does not automatically make" in normalized
    assert "Predictive associations may guide selection" in instructions
    assert "Do not create or narrow a scientific hypothesis" in instructions
    assert "strongest global alternatives" in normalized
    assert "only if it materially affected the decision" in normalized
    assert "second authoritative strategy record" in normalized
    assert "`wiki/...` source path" in instructions
    assert "fixed HIGH, MEDIUM, or LOW" not in instructions
    assert "materially different catalog region" not in instructions


def test_researcher_hypothesis_guidance_is_domain_neutral(campaign) -> None:
    """Do not prime the researcher with task-specific scientific variables."""
    _, capability = campaign
    raw_instructions = create_experimentalist_agent(capability).instructions
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


def test_only_found_results_count_toward_budget(campaign) -> None:
    """Preserve the host-owned request budget rule."""
    orchestrator, _ = campaign
    observations = [
        {"result": {"found": True}},
        {"result": {"found": False}},
    ]
    assert orchestrator._valid_request_count(observations) == 1


def test_only_accepted_lab_result_finishes_researcher_session() -> None:
    """Continue after duplicate and fully missing requests."""
    context = cast("RunContextWrapper[None]", None)
    for status in ("duplicate", "not_found"):
        results = cast(
            "list[FunctionToolResult]",
            [SimpleNamespace(output=json.dumps({"status": status}))],
        )
        assert not _finish_after_accepted_request(context, results).is_final_output
    accepted = cast(
        "list[FunctionToolResult]",
        [SimpleNamespace(output='{"status":"accepted"}')],
    )
    assert _finish_after_accepted_request(context, accepted).is_final_output


def test_run_initializes_tracing_before_campaign_execution(campaign, monkeypatch) -> None:
    """Initialize tracing before starting asynchronous campaign work."""
    orchestrator, _ = campaign
    events: list[str] = []

    def run_coroutine(coroutine: Coroutine[object, object, object]) -> None:
        events.append("run")
        coroutine.close()

    monkeypatch.setattr(experimentalist_module, "init_weave_tracing", lambda: events.append("trace"))
    monkeypatch.setattr(experimentalist_module.asyncio, "run", run_coroutine)
    orchestrator.run()

    assert events == ["trace", "run"]
