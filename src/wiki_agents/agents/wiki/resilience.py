"""Classify and retry Wiki page infrastructure operations."""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from agents import ModelRetrySettings, ModelSettings, RetryDecision
from agents.sandbox.errors import (
    ConfigurationError,
    ExecNonZeroError,
    SandboxError,
)
from docker.errors import APIError, DockerException

if TYPE_CHECKING:
    from collections.abc import Callable

    from agents import RetryPolicyContext

logger = logging.getLogger("wiki_agents.orchestrator")
_RATE_LIMIT_STATUS = 429
_SERVER_ERROR_STATUS = 500


class WikiPageFailureCategory(StrEnum):
    """Classify one terminal page outcome."""

    REVISION = "revision"
    INFRASTRUCTURE = "infrastructure"
    AGENT_OUTPUT = "agent_output"


class WikiEditorOutputError(RuntimeError):
    """Report an editor action that does not match the fixed plan."""

    def __init__(self, code: str, message: str) -> None:
        """Store a stable editor-output code.

        Args:
            code: Stable failure code.
            message: Human-readable failure description.

        """
        super().__init__(message)
        self.code = code


class WikiInfrastructureUnavailableError(RuntimeError):
    """Report repeated infrastructure failures across page jobs."""


@dataclass(frozen=True, slots=True)
class WikiTransientError:
    """Describe the structured facts for one retry decision."""

    retryable: bool
    category: str
    code: str | None = None
    status: int | None = None


def _exception_chain(error: BaseException) -> tuple[BaseException, ...]:
    """Return the error and its explicit cause or context chain."""
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ or current.__context__
    return tuple(chain)


def _status_code(error: BaseException) -> int | None:
    """Return an HTTP status from a structured exception when available."""
    status = getattr(error, "status_code", None)
    if isinstance(status, int):
        return status
    response = getattr(error, "response", None)
    response_status = getattr(response, "status_code", None)
    return response_status if isinstance(response_status, int) else None


def classify_wiki_transient_error(  # ruff: ignore[too-many-return-statements] - Each structured error family has a separate policy.
    error: BaseException,
) -> WikiTransientError:
    """Classify retryable sandbox and Docker transport failures."""
    chain = _exception_chain(error)
    sandbox_error = next(
        (item for item in chain if isinstance(item, SandboxError)),
        None,
    )
    if isinstance(sandbox_error, ConfigurationError | ExecNonZeroError):
        return WikiTransientError(
            retryable=False,
            category="sandbox",
            code=str(sandbox_error.error_code),
        )

    docker_error = next(
        (item for item in chain if isinstance(item, DockerException)),
        None,
    )
    status = next(
        (value for item in chain if (value := _status_code(item)) is not None),
        None,
    )
    if docker_error is not None and type(docker_error).__name__ in {
        "BuildError",
        "ContainerError",
        "ContextAlreadyExists",
        "ContextNotFound",
        "DeprecatedMethod",
        "InvalidArgument",
        "InvalidConfigFile",
        "InvalidRepository",
        "InvalidVersion",
        "MissingContextParameter",
        "NullResource",
        "TLSParameterError",
    }:
        return WikiTransientError(
            retryable=False,
            category="docker_configuration",
            code=type(docker_error).__name__,
            status=status,
        )
    if isinstance(docker_error, APIError):
        return WikiTransientError(
            retryable=(
                status == _RATE_LIMIT_STATUS
                or (status is not None and status >= _SERVER_ERROR_STATUS)
            ),
            category="docker_http",
            code=type(docker_error).__name__,
            status=status,
        )

    if status is not None:
        return WikiTransientError(
            retryable=(
                status in {408, 409, _RATE_LIMIT_STATUS}
                or status >= _SERVER_ERROR_STATUS
            ),
            category="provider_http",
            code=type(error).__name__,
            status=status,
        )

    connection_error = next(
        (
            item
            for item in chain
            if isinstance(item, ConnectionError | TimeoutError)
            or item.__class__.__name__
            in {
                "ConnectError",
                "ConnectionError",
                "ConnectTimeout",
                "NewConnectionError",
                "ProtocolError",
                "ReadTimeout",
                "ReadTimeoutError",
                "Timeout",
                "TimeoutError",
            }
        ),
        None,
    )
    if connection_error is not None:
        return WikiTransientError(
            retryable=True,
            category="transport",
            code=type(connection_error).__name__,
            status=status,
        )
    if sandbox_error is not None:
        code = str(sandbox_error.error_code)
        retryable = sandbox_error.retryable is True or code in {
            "exec_timeout",
            "exec_transport_error",
            "workspace_archive_read_error",
            "workspace_archive_write_error",
            "workspace_start_error",
            "workspace_stop_error",
        }
        return WikiTransientError(
            retryable=retryable,
            category="sandbox",
            code=code,
            status=status,
        )
    if docker_error is not None:
        return WikiTransientError(
            retryable=True,
            category="docker_transport",
            code=type(docker_error).__name__,
            status=status,
        )
    return WikiTransientError(retryable=False, category="other", status=status)


def _retry_delay(retry_number: int) -> float:
    """Return full-jitter exponential backoff capped at eight seconds."""
    return random.SystemRandom().uniform(
        0.0,
        min(8.0, 2.0 ** (retry_number - 1)),
    )


def wiki_model_settings(transient_retries: int) -> ModelSettings:
    """Configure bounded Agents SDK retries for transient model failures."""

    def retry_policy(context: RetryPolicyContext) -> RetryDecision:
        normalized = context.normalized
        provider_suggested = (
            context.provider_advice is not None
            and context.provider_advice.suggested is True
        )
        retryable = (
            provider_suggested
            or normalized.is_network_error
            or normalized.is_timeout
            or normalized.retry_after is not None
            or normalized.status_code in {408, 409, 429}
            or (
                normalized.status_code is not None
                and normalized.status_code >= _SERVER_ERROR_STATUS
            )
        )
        delay = (
            normalized.retry_after
            if normalized.retry_after is not None
            else _retry_delay(context.attempt)
        )
        if retryable:
            logger.warning(
                "Wiki page retry phase=model category=provider code=%s status=%s "
                "retry=%d delay=%.3f correction_count=unknown",
                normalized.error_code,
                normalized.status_code,
                context.attempt,
                delay,
            )
        return RetryDecision(retry=retryable, delay=delay)

    return ModelSettings(
        retry=ModelRetrySettings(
            max_retries=transient_retries,
            policy=retry_policy,
        ),
    )


def run_with_infrastructure_retries[T](
    operation: Callable[[], T],
    *,
    phase: str,
    retries: int,
    correction_count: int,
) -> T:
    """Run an operation again only after a transient infrastructure failure."""
    for retry_number in range(retries + 1):
        try:
            return operation()
        except Exception as error:
            classification = classify_wiki_transient_error(error)
            if not classification.retryable or retry_number >= retries:
                raise
            delay = _retry_delay(retry_number + 1)
            logger.warning(
                "Wiki page retry phase=%s category=%s code=%s status=%s "
                "retry=%d delay=%.3f correction_count=%d",
                phase,
                classification.category,
                classification.code,
                classification.status,
                retry_number + 1,
                delay,
                correction_count,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")
