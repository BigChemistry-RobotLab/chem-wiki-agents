import asyncio
import os

import pytest
from agents.sandbox import Manifest
from agents.sandbox.config import DEFAULT_PYTHON_SANDBOX_IMAGE
from agents.sandbox.entries import Dir
from agents.sandbox.sandboxes.docker import (
    DockerSandboxClient,
    DockerSandboxClientOptions,
)
from docker import from_env as docker_from_env

from wiki_agents.agents.wiki.creator import create_wiki_agent
from wiki_agents.agents.wiki.reviewer import create_wiki_reviewer_agent

pytestmark = pytest.mark.skipif(
    os.getenv("WIKI_AGENT_DOCKER_SMOKE") != "1",
    reason="Set WIKI_AGENT_DOCKER_SMOKE=1 to run Docker-backed Wiki smoke tests.",
)


async def _run_workspace_smoke() -> None:
    """Start one agent workspace without making a model call."""
    client = DockerSandboxClient(docker_from_env())
    session = await client.create(
        manifest=Manifest(entries={"wiki": Dir()}),
        options=DockerSandboxClientOptions(image=DEFAULT_PYTHON_SANDBOX_IMAGE),
    )
    try:
        async with session:
            result = await session.exec("find", "wiki", "-type", "d", shell=False)
            assert result.ok()
    finally:
        await client.delete(session)


def test_wiki_agent_definitions_and_docker_workspace_smoke() -> None:
    agents = [
        create_wiki_agent(model="test-model"),
        create_wiki_reviewer_agent(model="test-model"),
    ]

    assert all(agent.capabilities for agent in agents)
    asyncio.run(_run_workspace_smoke())
