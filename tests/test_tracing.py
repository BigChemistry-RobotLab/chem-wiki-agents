from contextlib import contextmanager

import wiki_agents.tracing as tracing_module
from wiki_agents.document.models import SourceRef


def test_traced_workflow_links_root_span_to_run(monkeypatch) -> None:
    events = []

    @contextmanager
    def fake_trace(name, *, group_id):
        events.append(("trace_start", name, group_id))
        try:
            yield
        finally:
            events.append(("trace_end", name, group_id))

    @contextmanager
    def fake_span(name, data):
        events.append(("span_start", name, data.copy()))
        try:
            yield
        finally:
            events.append(("span_end", name, data.copy()))

    monkeypatch.setattr(tracing_module, "_TRACING_ACTIVE", True)
    monkeypatch.setattr(tracing_module, "_RUN_ID", "run-123")
    monkeypatch.setattr(tracing_module, "get_current_trace", lambda: None)
    monkeypatch.setattr(tracing_module, "trace", fake_trace)
    monkeypatch.setattr(tracing_module, "custom_span", fake_span)

    with tracing_module.traced_workflow(
        "Plan wiki page",
        metadata={"page_id": "page-1"},
    ) as metadata:
        metadata["status"] = "accepted"

    assert events == [
        ("trace_start", "Plan wiki page", "run-123"),
        (
            "span_start",
            "Plan wiki page",
            {"run_id": "run-123", "page_id": "page-1"},
        ),
        (
            "span_end",
            "Plan wiki page",
            {
                "run_id": "run-123",
                "page_id": "page-1",
                "status": "accepted",
            },
        ),
        ("trace_end", "Plan wiki page", "run-123"),
    ]


def test_pdf_source_metadata_keeps_linked_provenance() -> None:
    first = SourceRef(
        source_id="paper_a",
        sha256="a" * 64,
        resource="/references/papers/paper-a.md",
    )
    second = SourceRef(
        source_id="paper_b",
        sha256="b" * 64,
        resource="/references/papers/paper-b.md",
    )

    metadata = tracing_module.pdf_source_metadata([second, first, second])

    assert metadata == {
        "pdf_source_ids": ["paper_a", "paper_b"],
        "pdf_sha256": ["a" * 64, "b" * 64],
        "pdf_reference_resources": [
            "/references/papers/paper-a.md",
            "/references/papers/paper-b.md",
        ],
        "pdf_artifact_resources": [
            "/references/papers/artifacts/a.pdf",
            "/references/papers/artifacts/b.pdf",
        ],
    }
