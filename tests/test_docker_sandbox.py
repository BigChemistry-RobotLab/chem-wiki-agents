from contextlib import nullcontext

import pytest

from wiki_agents import docker_sandbox


@pytest.mark.parametrize("raise_inside", [False, True])
def test_managed_docker_sandbox_client_closes_connection_pool(
    monkeypatch,
    *,
    raise_inside: bool,
) -> None:
    class FakeDockerClient:
        closed = False

        def close(self) -> None:
            self.closed = True

    docker_client = FakeDockerClient()
    monkeypatch.setattr(docker_sandbox, "docker_from_env", lambda: docker_client)

    with (
        pytest.raises(RuntimeError) if raise_inside else nullcontext(),
        docker_sandbox.managed_docker_sandbox_client() as sandbox_client,
    ):
        assert sandbox_client.docker_client is docker_client
        if raise_inside:
            raise RuntimeError("sandbox failed")

    assert docker_client.closed
