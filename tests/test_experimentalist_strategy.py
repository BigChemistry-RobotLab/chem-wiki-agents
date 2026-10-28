"""Tests for report-centered curation and recovery."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import httpx
import pytest

from wiki_agents.agents.experimentalist.lab_tools import create_lab_capability
from wiki_agents.agents.experimentalist.report import (
    GoalAssessment,
    ReporterResult,
)
from wiki_agents.orchestrator.experimentalist import (
    ExperimentalistOrchestrator,
    ExperimentalistOrchestratorConfig,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from agents.result import RunResult
    from agents.sandbox import SandboxAgent
    from agents.sandbox.session.base_sandbox_session import BaseSandboxSession

    from wiki_agents.agents.experimentalist.lab_tools import LabCapability


def _transport(request: httpx.Request) -> httpx.Response:
    """Serve a three-row lab configuration."""
    if request.url.path == "/v1/schema":
        return httpx.Response(
            200,
            json={
                "inputs": [{"name": "sample", "type": "string"}],
                "outputs": [{"name": "score", "type": "number"}],
                "materials": [{"name": "sample", "type": "string"}],
                "required_inputs": ["sample"],
                "max_batch_size": 3,
            },
        )
    if request.url.path == "/v1/materials":
        return httpx.Response(
            200,
            json={
                "items": [{"sample": "A"}, {"sample": "B"}, {"sample": "C"}],
                "total": 3,
                "offset": 0,
                "limit": 1000,
            },
        )
    return httpx.Response(404)


def _report(label: str) -> str:
    """Return one valid rolling report."""
    return f"""# Experimental campaign report

## Objective and progress

Immutable task: Find the best sample.

## Predictive evidence and current policy

{label} predictive evidence.

## Scientific hypotheses and limitations

{label} hypothesis evidence.

## General Observations

{label} observations.

## Next ranked options and open questions

Test the next exact catalog candidate.
"""


@pytest.fixture
def campaign(
    tmp_path: Path,
) -> Iterator[tuple[ExperimentalistOrchestrator, LabCapability]]:
    """Create an initialized no-Wiki campaign.

    Yields:
        The orchestrator and lab capability.

    """
    capability = create_lab_capability(
        "http://lab.test",
        transport=httpx.MockTransport(_transport),
    )
    orchestrator = ExperimentalistOrchestrator(
        ExperimentalistOrchestratorConfig(
            wiki_path=None,
            campaign_path=tmp_path / "campaign",
            task="Find the best sample.",
            request_budget=3,
            batch_size=3,
        ),
        capability=capability,
    )
    orchestrator._load_or_initialize_campaign(capability)
    yield orchestrator, capability
    capability.client.close()


class _Session:
    """Store host writes used to refresh reporter inputs."""

    def __init__(self) -> None:
        self.writes: dict[str, bytes] = {}

    async def write(self, path: Path, data) -> None:
        """Store one sandbox file replacement."""
        self.writes[path.as_posix()] = data.read()


def _record(
    orchestrator: ExperimentalistOrchestrator,
    requests: list[dict[str, object]],
    results: list[dict[str, object]],
) -> None:
    """Record one accepted batch under a new curation group."""
    orchestrator._start_curation_group("group")
    orchestrator._record_accepted(
        requests,
        "Tests a local hypothesis against the strongest alternative.",
        {"results": results},
    )


def test_missing_transcript_recovery_and_found_false_normalization(
    campaign,
    monkeypatch,
) -> None:
    """Report from stored evidence and force missing data to inconclusive."""
    orchestrator, _ = campaign
    _record(
        orchestrator,
        [{"sample": "A"}],
        [{"request": {"sample": "A"}, "found": False}],
    )
    payloads: list[dict[str, object]] = []

    async def run_reporter(reporter, payload, record, count, session):
        _ = reporter, record, count, session
        payloads.append(payload)
        return ReporterResult(
            report_markdown=_report("A"),
            assessment=GoalAssessment.NON_PROGRESS,
            assessment_rationale="Reporter tried to classify the absent result.",
        )

    monkeypatch.setattr(orchestrator, "_run_reporter", run_reporter)
    asyncio.run(
        orchestrator._process_pending_collation(
            reporter=cast("SandboxAgent[None]", SimpleNamespace()),
            session=cast("BaseSandboxSession", _Session()),
        ),
    )

    notice = cast("str", payloads[0]["researcher_transcript"])
    assert "Unavailable" in notice
    assert "stored rationale, ledger, report, and work files" in notice
    curation = cast("dict[str, object]", orchestrator._read_observations()[0]["curation"])
    assert curation == {
        "assessment": "INCONCLUSIVE",
        "assessment_rationale": (
            "The lab returned found: false, so the measurement is inconclusive "
            "against the immutable task."
        ),
    }
    assert orchestrator._pending_collation() is None


def test_multi_observation_batch_uses_one_transcript_and_rolling_reports(
    campaign,
    monkeypatch,
) -> None:
    """Run one reporter pass per result against the report from the prior pass."""
    orchestrator, _ = campaign
    requests: list[dict[str, object]] = [{"sample": "A"}, {"sample": "B"}]
    _record(
        orchestrator,
        requests,
        [
            {"request": requests[0], "found": True, "values": {"score": 1}},
            {"request": requests[1], "found": True, "values": {"score": 2}},
        ],
    )
    transcript = orchestrator.campaign_path / "sessions" / "session-0001.json"
    orchestrator._write_json(transcript, {"items": [{"role": "user", "content": "x"}]})
    pending = orchestrator._require_pending_collation()
    pending["transcript_path"] = "sessions/session-0001.json"
    pending["phase"] = "transcript_persisted"
    orchestrator._write_pending_collation(pending)
    session = _Session()
    calls: list[str] = []

    async def run_reporter(reporter, payload, record, count, received_session):
        _ = reporter, payload, count, received_session
        previous = session.writes["inputs/report.md"].decode()
        calls.append(previous)
        label = cast("dict[str, str]", record["request"])["sample"]
        return ReporterResult(
            report_markdown=_report(label),
            assessment=GoalAssessment.PROGRESS,
            assessment_rationale=f"{label} supplies task evidence.",
        )

    monkeypatch.setattr(orchestrator, "_run_reporter", run_reporter)
    asyncio.run(
        orchestrator._process_pending_collation(
            reporter=cast("SandboxAgent[None]", SimpleNamespace()),
            session=cast("BaseSandboxSession", session),
        ),
    )

    assert len(calls) == 2
    assert calls[0] == orchestrator._initial_report(orchestrator.config.task)
    assert calls[1] == _report("A")
    assert session.writes["inputs/latest_researcher_session.json"] == transcript.read_bytes()
    assert (orchestrator.campaign_path / "report.md").read_text() == _report("B")
    observations = orchestrator._read_observations()
    assert all(
        set(cast("dict[str, object]", item["curation"]))
        == {"assessment", "assessment_rationale"}
        for item in observations
    )


def test_recovery_discovers_transcript_before_journal_association(
    campaign,
    monkeypatch,
) -> None:
    """Recover a transcript saved before its journal path was written."""
    orchestrator, _ = campaign
    request: dict[str, object] = {"sample": "A"}
    _record(
        orchestrator,
        [request],
        [{"request": request, "found": True, "values": {"score": 1}}],
    )
    transcript = orchestrator.campaign_path / "sessions" / "session-0001.json"
    orchestrator._write_json(
        transcript,
        {"curation_group_id": "group", "items": [{"role": "user"}]},
    )
    payloads: list[dict[str, object]] = []

    async def run_reporter(reporter, payload, record, count, session):
        _ = reporter, record, count, session
        payloads.append(payload)
        return ReporterResult(
            report_markdown=_report("Recovered transcript"),
            assessment=GoalAssessment.PROGRESS,
            assessment_rationale="The score informs the task.",
        )

    monkeypatch.setattr(orchestrator, "_run_reporter", run_reporter)
    asyncio.run(
        orchestrator._process_pending_collation(
            reporter=cast("SandboxAgent[None]", SimpleNamespace()),
            session=cast("BaseSandboxSession", _Session()),
        ),
    )

    assert payloads[0]["researcher_transcript"] == (
        "inputs/latest_researcher_session.json"
    )


@pytest.mark.parametrize(
    "phase",
    ["reporter_draft_persisted", "assessment_attached", "report_published"],
)
def test_recovery_reuses_persisted_reporter_draft(
    campaign,
    monkeypatch,
    phase: str,
) -> None:
    """Do not repeat a reporter call after its complete result is journaled."""
    orchestrator, _ = campaign
    request: dict[str, object] = {"sample": "A"}
    _record(
        orchestrator,
        [request],
        [{"request": request, "found": True, "values": {"score": 1}}],
    )
    draft = ReporterResult(
        report_markdown=_report("Recovered"),
        assessment=GoalAssessment.PROGRESS,
        assessment_rationale="The score informs the task.",
    )
    pending = orchestrator._require_pending_collation()
    pending["reporter_draft"] = draft.model_dump(mode="json")
    pending["phase"] = phase
    orchestrator._write_pending_collation(pending)
    if phase in {"assessment_attached", "report_published"}:
        observations = orchestrator._read_observations()
        orchestrator._attach_curation(observations, 0, draft)
        orchestrator._write_observations(observations)
    if phase == "report_published":
        orchestrator._publish_report(draft.report_markdown)

    async def fail(*args: object, **kwargs: object):
        _ = args, kwargs
        pytest.fail("Reporter was called again")

    monkeypatch.setattr(orchestrator, "_run_reporter", fail)
    asyncio.run(
        orchestrator._process_pending_collation(
            reporter=cast("SandboxAgent[None]", SimpleNamespace()),
            session=cast("BaseSandboxSession", _Session()),
        ),
    )

    assert orchestrator._pending_collation() is None
    assert (orchestrator.campaign_path / "report.md").read_text() == _report("Recovered")


def test_complete_visible_transcript_excludes_hidden_reasoning(campaign) -> None:
    """Persist visible user, shell, and tool items under the curation group."""
    orchestrator, _ = campaign
    orchestrator._start_curation_group("transcript-group")
    result = SimpleNamespace(
        to_input_list=lambda: [
            {"role": "user", "content": "task"},
            {"type": "shell_call", "command": "rg score"},
            {"type": "shell_call_output", "output": "match"},
            {"type": "function_call", "name": "request_data"},
            {"type": "function_call_output", "output": "accepted"},
            {"type": "reasoning", "content": "hidden"},
        ],
    )

    path = orchestrator._write_transcript(cast("RunResult", result))
    payload = json.loads(path.read_text())

    assert payload["curation_group_id"] == "transcript-group"
    assert len(payload["items"]) == 5
    assert all(item.get("type") != "reasoning" for item in payload["items"])
