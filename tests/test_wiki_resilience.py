"""Test bounded Wiki page infrastructure retries."""

from types import SimpleNamespace

import pytest
from agents.sandbox.errors import ConfigurationError, ErrorCode, SandboxError
from docker.errors import APIError, ContainerError, DockerException, InvalidArgument

from wiki_agents.agents.wiki.resilience import (
    classify_wiki_transient_error,
    run_with_infrastructure_retries,
)


@pytest.mark.parametrize(
    ("error", "retryable", "category", "status"),
    [
        (
            SandboxError(
                "transport failed",
                ErrorCode.EXEC_TRANSPORT_ERROR,
                "exec",
                {},
            ),
            True,
            "sandbox",
            None,
        ),
        (
            ConfigurationError(
                "bad manifest",
                ErrorCode.SANDBOX_CONFIG_INVALID,
                "materialize",
                {},
            ),
            False,
            "sandbox",
            None,
        ),
        (DockerException("daemon disconnected"), True, "docker_transport", None),
        (
            APIError("rate limited", response=SimpleNamespace(status_code=429)),
            True,
            "docker_http",
            429,
        ),
        (
            APIError("bad request", response=SimpleNamespace(status_code=400)),
            False,
            "docker_http",
            400,
        ),
        (
            APIError("authentication failed", response=SimpleNamespace(status_code=401)),
            False,
            "docker_http",
            401,
        ),
        (
            APIError("daemon unavailable", response=SimpleNamespace(status_code=503)),
            True,
            "docker_http",
            503,
        ),
        (
            InvalidArgument("bad Docker setting"),
            False,
            "docker_configuration",
            None,
        ),
        (
            ContainerError(object(), 1, "false", "image", "failed"),
            False,
            "docker_configuration",
            None,
        ),
        (ConnectionError("connection refused"), True, "transport", None),
        (TimeoutError("timed out"), True, "transport", None),
        (RuntimeError("invalid output"), False, "other", None),
    ],
)
def test_transient_error_classification(error, retryable, category, status) -> None:
    classification = classify_wiki_transient_error(error)

    assert classification.retryable is retryable
    assert classification.category == category
    assert classification.status == status


def test_infrastructure_retry_limit_uses_fresh_operations(monkeypatch) -> None:
    calls = 0
    sleeps: list[float] = []

    def operation() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise ConnectionError("connection refused")
        return "accepted"

    monkeypatch.setattr(
        "wiki_agents.agents.wiki.resilience._retry_delay",
        float,
    )
    monkeypatch.setattr(
        "wiki_agents.agents.wiki.resilience.time.sleep",
        sleeps.append,
    )

    result = run_with_infrastructure_retries(
        operation,
        phase="reviewer",
        retries=2,
        correction_count=1,
    )

    assert result == "accepted"
    assert calls == 3
    assert sleeps == [1.0, 2.0]
