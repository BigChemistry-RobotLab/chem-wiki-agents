"""Manage Docker sandbox clients and their connection pools."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

from agents.sandbox.sandboxes.docker import DockerSandboxClient
from docker import from_env as docker_from_env

if TYPE_CHECKING:
    from collections.abc import Iterator


@contextmanager
def managed_docker_sandbox_client() -> Iterator[DockerSandboxClient]:
    """Create a sandbox client and close its Docker connection pool.

    Yields:
        A sandbox client that owns the temporary Docker connection pool.

    """
    docker_client = docker_from_env()
    try:
        yield DockerSandboxClient(docker_client)
    finally:
        docker_client.close()
