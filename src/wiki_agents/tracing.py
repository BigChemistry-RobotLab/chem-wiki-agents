import os
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from importlib import import_module
from importlib.util import find_spec
from pathlib import PurePosixPath
from threading import Lock
from types import ModuleType
from typing import Any
from uuid import uuid4

from agents import custom_span, get_current_trace, set_trace_processors, trace

from wiki_agents.document.models import SourceRef

weave: ModuleType | None = (
    import_module("weave") if find_spec("weave") is not None else None
)

_INITIALIZED = False
_TRACING_ACTIVE = False
_INITIALIZATION_LOCK = Lock()
_RUN_ID: str | None = None


def init_weave_tracing() -> None:
    """Use Weave as the trace exporter when the package is installed."""
    global _INITIALIZED, _TRACING_ACTIVE  # ruff: ignore[global-statement] - Process-wide tracing state.
    if _INITIALIZED:
        return

    with _INITIALIZATION_LOCK:
        if _INITIALIZED:
            return

        # The Agents SDK installs its OpenAI exporter when code first accesses the
        # trace provider. It always posts to api.openai.com. An OpenAI-compatible
        # provider key can cause a 401 response. Start with no processors. If
        # Weave is available, weave.init() registers the Weave processor.
        set_trace_processors([])
        if weave is None:
            _INITIALIZED = True
            return
        weave.init(
            os.getenv("WEAVE_PROJECT", "agents-test"),
            settings={
                "log_level": os.getenv("THIRD_PARTY_LOG_LEVEL", "WARNING"),
                "print_call_link": False,
            },
        )
        _TRACING_ACTIVE = True
        _INITIALIZED = True


def get_run_id() -> str:
    """Return the process-wide identifier that links one pipeline run."""
    global _RUN_ID  # ruff: ignore[global-statement] - Process-wide trace correlation.
    if _RUN_ID is None:
        with _INITIALIZATION_LOCK:
            if _RUN_ID is None:
                _RUN_ID = os.getenv("WIKI_RUN_ID") or str(uuid4())
    return _RUN_ID


@contextmanager
def traced_workflow(
    name: str,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    """Record one named workflow and link it to the complete pipeline run.

    Args:
        name: Clear operation name for the trace viewer.
        metadata: Searchable identifiers and provenance for the operation.

    Yields:
        Mutable span metadata. The caller can add outcome fields before exit.

    """
    run_id = get_run_id()
    span_metadata = {"run_id": run_id, **dict(metadata or {})}
    if not _TRACING_ACTIVE:
        yield span_metadata
        return
    if get_current_trace() is not None:
        with custom_span(name, span_metadata):
            yield span_metadata
        return

    with trace(name, group_id=run_id), custom_span(name, span_metadata):
        yield span_metadata


def record_trace_event(name: str, metadata: Mapping[str, Any]) -> None:
    """Record an instantaneous semantic outcome in the active workflow."""
    if not _TRACING_ACTIVE:
        return
    with traced_workflow(name, metadata=metadata):
        pass


def pdf_source_metadata(sources: Sequence[SourceRef]) -> dict[str, object]:
    """Build searchable PDF provenance fields for a trace span."""
    unique_sources = {source.source_id: source for source in sources}
    ordered = [unique_sources[source_id] for source_id in sorted(unique_sources)]
    return {
        "pdf_source_ids": [source.source_id for source in ordered],
        "pdf_sha256": [source.sha256 for source in ordered],
        "pdf_reference_resources": [source.resource for source in ordered],
        "pdf_artifact_resources": [
            artifact
            for source in ordered
            if (artifact := _pdf_artifact_resource(source)) is not None
        ],
    }


def _pdf_artifact_resource(source: SourceRef) -> str | None:
    """Derive the managed PDF artifact from a standard paper reference."""
    reference = PurePosixPath(source.resource)
    if reference.parent.name != "papers" or not source.source_id.startswith("paper_"):
        return None
    filename = source.source_id.removeprefix("paper_") + ".pdf"
    return str(reference.parent / "artifacts" / filename)
