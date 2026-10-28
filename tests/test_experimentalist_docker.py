"""Gated Docker isolation test for experimental campaigns."""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING

import httpx
import pytest
from agents.sandbox.config import DEFAULT_PYTHON_SANDBOX_IMAGE
from agents.sandbox.sandboxes.docker import (
    DockerSandboxClient,
    DockerSandboxClientOptions,
)
from docker import from_env as docker_from_env

from wiki_agents.agents.experimentalist.lab_tools import create_lab_capability
from wiki_agents.orchestrator.experimentalist import (
    ExperimentalistOrchestrator,
    ExperimentalistOrchestratorConfig,
)

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.skipif(
    os.getenv("EXPERIMENTALIST_DOCKER_SMOKE") != "1",
    reason=(
        "Set EXPERIMENTALIST_DOCKER_SMOKE=1 to run the Docker-backed isolation test."
    ),
)


def _transport(request: httpx.Request) -> httpx.Response:
    """Serve public lab metadata without exposing experiment outputs."""
    if request.url.path == "/v1/schema":
        return httpx.Response(
            200,
            json={
                "inputs": [
                    {"name": "sequence", "type": "string"},
                    {"name": "ph", "type": "number"},
                ],
                "outputs": [
                    {"name": "concentration", "type": "number"},
                    {"name": "surface_tension", "type": "number"},
                ],
                "materials": [
                    {"name": "sequence", "type": "string"},
                    {"name": "ph", "type": "number"},
                ],
                "required_inputs": ["sequence", "ph"],
                "max_batch_size": 1,
            },
        )
    if request.url.path == "/v1/materials":
        return httpx.Response(
            200,
            json={
                "items": [{"sequence": "AHG", "ph": 3.0}],
                "total": 1,
                "offset": 0,
                "limit": 1000,
            },
        )
    return httpx.Response(404)


async def _run_isolation_check(tmp_path: Path) -> None:
    """Create the campaign sandbox and inspect it without a model call."""
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    (wiki / "knowledge.md").write_text("# Public knowledge\n", encoding="utf-8")
    config = ExperimentalistOrchestratorConfig(
        wiki_path=wiki,
        campaign_path=tmp_path / "campaign",
        task="Select one experiment.",
        request_budget=1,
    )
    capability = create_lab_capability(
        "http://lab.test",
        transport=httpx.MockTransport(_transport),
    )
    orchestrator = ExperimentalistOrchestrator(config, capability=capability)
    orchestrator._load_or_initialize_campaign(capability)
    (orchestrator.campaign_path / "work" / "research.txt").write_text(
        "research evidence\n",
        encoding="utf-8",
    )
    manifest = capability.process_manifest(
        orchestrator._build_manifest([], capability.material_records),
    )
    client = DockerSandboxClient(docker_from_env())
    session = await client.create(
        manifest=orchestrator._build_session_manifest(),
        options=DockerSandboxClientOptions(
            image=DEFAULT_PYTHON_SANDBOX_IMAGE,
            network_mode="none",
        ),
    )
    try:
        async with session:
            await orchestrator._materialize_workspace(session, manifest)
            assert session.describe() == "/workspace\n"
            network = await session.exec(
                "python",
                "-c",
                (
                    "import socket; "
                    "socket.create_connection(('1.1.1.1', 53), timeout=0.2)"
                ),
                shell=False,
                user="researcher",
            )
            assert not network.ok()
            writable = await session.exec(
                "sh",
                "-lc",
                "printf analysis > work/check.txt",
                shell=False,
                user="researcher",
            )
            assert writable.ok()
            reporter_reads = await session.exec(
                "sh",
                "-lc",
                (
                    "test -r wiki/knowledge.md && test -r inputs/report.md && "
                    "test -r inputs/observations.jsonl && test -r work/research.txt"
                ),
                shell=False,
                user="reporter",
            )
            assert reporter_reads.ok()
            reporter_writes = await session.exec(
                "sh",
                "-lc",
                "printf forbidden > work/reporter.txt",
                shell=False,
                user="reporter",
            )
            assert not reporter_writes.ok()
            reporter_input_write = await session.exec(
                "sh",
                "-lc",
                "printf forbidden > inputs/report.md",
                shell=False,
                user="reporter",
            )
            assert not reporter_input_write.ok()
            reporter_network = await session.exec(
                "python",
                "-c",
                (
                    "import socket; "
                    "socket.create_connection(('1.1.1.1', 53), timeout=0.2)"
                ),
                shell=False,
                user="reporter",
            )
            assert not reporter_network.ok()
            inventory = await session.exec(
                "sh",
                "-lc",
                "find . -type f -maxdepth 5 -print; grep -R -E '47[.]97541879|0[.]5,47' .",
                shell=False,
                user="researcher",
            )
            text = (inventory.stdout + inventory.stderr).decode("utf-8")
            assert "dataset" not in text.casefold()
            assert "47.97541879" not in text
            assert "0.5,47" not in text
    finally:
        await client.delete(session)
        capability.client.close()


def test_sandbox_has_no_network_or_hidden_lab_data(tmp_path: Path) -> None:
    """Prove that the researcher receives only public campaign inputs."""
    asyncio.run(_run_isolation_check(tmp_path))
