import logging
import tempfile
from importlib.metadata import entry_points
from pathlib import Path
from threading import Barrier, Event, Lock, get_ident
from types import SimpleNamespace

import pytest

import wiki_agents.concept.workflow as concept_workflow_module
import wiki_agents.orchestrator.concept_store as concept_store_module
import wiki_agents.orchestrator.wiki as orchestrator_module
import wiki_agents.orchestrator.wiki_extraction as extraction_module
import wiki_agents.tracing as tracing_module
from wiki_agents.agents.concept.critic import ConceptCritique, CritiqueDecision
from wiki_agents.agents.concept.extractor import (
    Claim,
    Concept,
    ConceptExtraction,
    ExtractionAttempt,
)
from wiki_agents.concept.checkpoints import ConceptCheckpointStore
from wiki_agents.concept.chunk_runner import ConceptChunkRunner
from wiki_agents.concept.models import (
    ConceptExtractionCheckpoint,
    ConceptExtractionChunk,
    ConceptExtractionRetryState,
)
from wiki_agents.concept.pipeline import ConceptExtractionPipeline
from wiki_agents.concept.store import ConceptStore
from wiki_agents.document.models import EvidenceSpan, SourceRef
from wiki_agents.orchestrator import (
    ConceptChunkWork,
    ConceptExtractionExhausted,
    ExperimentalistOrchestrator,
    ExperimentalistOrchestratorConfig,
    WikiOrchestrator,
    WikiOrchestratorConfig,
)


def _make_wiki_orchestrator(  # ruff: ignore[too-many-arguments] - Mirrors config.
    *,
    repository_path: str | Path,
    extractor_model: str,
    critic_model: str,
    fixer_model: str,
    max_wiki_turns: int = 5,
    max_concept_turns: int = 5,
    num_extraction_workers: int = 4,
    num_concept_pages: int = 3,
    wiki_planning_batch_size: int = 5,
    similar_concepts_top_k: int = 5,
    sources_path: str | Path | None = None,
    concept_store: ConceptStore | None = None,
    bot_name: str = "Wiki Agent",
    bot_email: str = "wiki-agent@localhost",
) -> WikiOrchestrator:
    return WikiOrchestrator(
        WikiOrchestratorConfig(
            repository_path=repository_path,
            extractor_model=extractor_model,
            critic_model=critic_model,
            fixer_model=fixer_model,
            max_wiki_turns=max_wiki_turns,
            max_concept_turns=max_concept_turns,
            num_extraction_workers=num_extraction_workers,
            num_concept_pages=num_concept_pages,
            wiki_planning_batch_size=wiki_planning_batch_size,
            similar_concepts_top_k=similar_concepts_top_k,
            sources_path=sources_path,
            concept_store=concept_store,
            bot_name=bot_name,
            bot_email=bot_email,
        ),
    )


def test_packaged_cli_entry_point_displays_help(capsys) -> None:
    (entry_point,) = entry_points(group="console_scripts", name="wiki-agent")

    assert entry_point.value == "wiki_agents.orchestrator:main"
    with pytest.raises(SystemExit) as exc_info:
        entry_point.load()(["--help"])

    assert exc_info.value.code == 0
    assert "Run the wiki build workflow." in capsys.readouterr().out


def test_wiki_orchestrator_config_resolves_environment(monkeypatch, tmp_path) -> None:
    values = {
        "EXTRACTOR_MODEL": "extractor-env",
        "CRITIC_MODEL": "critic-env",
        "FIXER_MODEL": "fixer-env",
        "WIKI_PLANNER_MODEL": "wiki-planner-env",
        "WIKI_CREATOR_MODEL": "wiki-creator-env",
        "WIKI_REVIEWER_MODEL": "wiki-reviewer-env",
        "WIKI_PAGE_TRANSIENT_RETRIES": "0",
        "WIKI_FINALIZER_BATCH_SIZE": "7",
        "WIKI_FINALIZER_TIMEOUT_SECONDS": "45.5",
        "WIKI_MAX_TURNS": "6",
        "CONCEPT_MAX_TURNS": "7",
        "WIKI_EXTRACTION_WORKERS": "8",
        "WIKI_NUM_CONCEPT_PAGES": "9",
        "WIKI_PLANNING_BATCH_SIZE": "4",
        "WIKI_GRAPH_PAGERANK_ALPHA": "0.77",
        "WIKI_MAX_OUTGOING_LINKS": "12",
        "WIKI_LINK_ALWAYS_INCLUDE_SIMILARITY": "0.88",
        "WIKI_LINK_SEMANTIC_WEIGHT": "0.60",
        "WIKI_LINK_PERSONALIZED_PAGERANK_WEIGHT": "0.25",
        "WIKI_LINK_GLOBAL_PAGERANK_WEIGHT": "0.15",
        "WIKI_MAX_PAGE_CANDIDATES": "14",
        "WIKI_MAX_PAGE_WORK_ITEMS": "22",
        "WIKI_GIT_AUTHOR_NAME": "Environment Bot",
        "WIKI_GIT_AUTHOR_EMAIL": "environment@example.test",
        "WIKI_SIMILAR_CONCEPTS_TOP_K": "10",
        "QDRANT_URL": "https://qdrant.example.test",
        "QDRANT_API_KEY": "qdrant-secret",
        "QDRANT_COLLECTION": "environment-concepts",
        "EMBEDDING_DEFAULT_MODEL": "embedding-env",
        "EMBEDDING_BASE_URL": "https://embedding.example.test/v1",
        "EMBEDDING_API_KEY": "embedding-secret",
        "EMBEDDING_BATCH_SIZE": "11",
        "EMBEDDING_MAX_INPUT_CHARS": "900",
    }
    monkeypatch.delenv("QDRANT_PATH", raising=False)
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    config = WikiOrchestratorConfig.from_env(tmp_path / "repository")

    assert config.extractor_model == "extractor-env"
    assert config.critic_model == "critic-env"
    assert config.fixer_model == "fixer-env"
    assert config.wiki_planner_model == "wiki-planner-env"
    assert config.wiki_creator_model == "wiki-creator-env"
    assert config.wiki_reviewer_model == "wiki-reviewer-env"
    assert config.wiki_page_transient_retries == 0
    assert config.wiki_finalizer_batch_size == 7
    assert config.wiki_finalizer_timeout_seconds == pytest.approx(45.5)
    assert config.max_wiki_turns == 6
    assert config.max_concept_turns == 7
    assert config.num_extraction_workers == 8
    assert config.num_concept_pages == 9
    assert config.wiki_planning_batch_size == 4
    assert config.wiki_graph_pagerank_alpha == pytest.approx(0.77)
    assert config.wiki_max_outgoing_links == 12
    assert config.wiki_link_always_include_similarity == pytest.approx(0.88)
    assert config.wiki_link_semantic_weight == pytest.approx(0.60)
    assert config.wiki_link_personalized_pagerank_weight == pytest.approx(0.25)
    assert config.wiki_link_global_pagerank_weight == pytest.approx(0.15)
    assert config.wiki_max_page_candidates == 14
    assert config.wiki_max_page_work_items == 22
    assert config.bot_name == "Environment Bot"
    assert config.bot_email == "environment@example.test"
    assert config.similar_concepts_top_k == 10
    assert config.qdrant_url == "https://qdrant.example.test"
    assert config.qdrant_path is None
    assert config.qdrant_api_key == "qdrant-secret"
    assert config.qdrant_collection == "environment-concepts"
    assert config.embedding_model == "embedding-env"
    assert config.embedding_base_url == "https://embedding.example.test/v1"
    assert config.embedding_api_key == "embedding-secret"
    assert config.embedding_batch_size == 11
    assert config.embedding_max_input_chars == 900
    assert "qdrant-secret" not in repr(config)
    assert "embedding-secret" not in repr(config)

    orchestrator = WikiOrchestrator(config)
    assert "qdrant-secret" not in repr(orchestrator.concept_store_provider.settings)
    assert "embedding-secret" not in repr(orchestrator.concept_store_provider.settings)


@pytest.mark.parametrize(
    "name",
    [
        "max_wiki_turns",
        "max_concept_turns",
        "num_extraction_workers",
        "num_concept_pages",
        "wiki_planning_batch_size",
        "wiki_max_outgoing_links",
        "wiki_max_page_candidates",
        "wiki_max_page_work_items",
        "wiki_finalizer_batch_size",
        "similar_concepts_top_k",
        "embedding_batch_size",
        "embedding_max_input_chars",
    ],
)
def test_wiki_orchestrator_config_rejects_nonpositive_settings(tmp_path, name) -> None:
    config = WikiOrchestratorConfig(
        repository_path=tmp_path / "repository",
        extractor_model="extractor",
        critic_model="critic",
        fixer_model="fixer",
    )
    object.__setattr__(  # ruff: ignore[unnecessary-dunder-call] - Make invalid data.
        config,
        name,
        0,
    )

    with pytest.raises(ValueError, match=rf"{name} must be positive"):
        config.__post_init__()


def test_wiki_orchestrator_config_rejects_two_qdrant_locations(tmp_path) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        WikiOrchestratorConfig(
            repository_path=tmp_path / "repository",
            extractor_model="extractor",
            critic_model="critic",
            fixer_model="fixer",
            qdrant_url="https://qdrant.example.test",
            qdrant_path=tmp_path / "qdrant",
        )


@pytest.mark.parametrize("timeout", [0.0, -1.0, float("nan"), float("inf")])
def test_wiki_orchestrator_config_rejects_invalid_finalizer_timeout(
    tmp_path,
    timeout,
) -> None:
    with pytest.raises(ValueError, match="must be finite and positive"):
        WikiOrchestratorConfig(
            repository_path=tmp_path / "repository",
            extractor_model="extractor",
            critic_model="critic",
            fixer_model="fixer",
            wiki_finalizer_timeout_seconds=timeout,
        )


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.1, float("nan"), float("inf")])
def test_wiki_orchestrator_config_rejects_invalid_pagerank_alpha(
    tmp_path,
    alpha,
) -> None:
    with pytest.raises(ValueError, match="pagerank_alpha must be between 0 and 1"):
        WikiOrchestratorConfig(
            repository_path=tmp_path / "repository",
            extractor_model="extractor",
            critic_model="critic",
            fixer_model="fixer",
            wiki_graph_pagerank_alpha=alpha,
        )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("wiki_link_semantic_weight", -0.1),
        ("wiki_link_personalized_pagerank_weight", float("nan")),
        ("wiki_link_global_pagerank_weight", float("inf")),
    ],
)
def test_wiki_orchestrator_config_rejects_invalid_link_weight(
    tmp_path,
    name,
    value,
) -> None:
    settings = {name: value}

    with pytest.raises(ValueError, match=rf"{name} must be finite and nonnegative"):
        WikiOrchestratorConfig(
            repository_path=tmp_path / "repository",
            extractor_model="extractor",
            critic_model="critic",
            fixer_model="fixer",
            **settings,
        )


def test_wiki_orchestrator_config_requires_link_weights_to_sum_to_one(
    tmp_path,
) -> None:
    with pytest.raises(ValueError, match=r"ranking weights must sum to 1\.0"):
        WikiOrchestratorConfig(
            repository_path=tmp_path / "repository",
            extractor_model="extractor",
            critic_model="critic",
            fixer_model="fixer",
            wiki_link_semantic_weight=0.51,
        )


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), float("inf")])
def test_wiki_orchestrator_config_rejects_invalid_always_include_threshold(
    tmp_path,
    threshold,
) -> None:
    with pytest.raises(
        ValueError,
        match="always_include_similarity must be between 0 and 1",
    ):
        WikiOrchestratorConfig(
            repository_path=tmp_path / "repository",
            extractor_model="extractor",
            critic_model="critic",
            fixer_model="fixer",
            wiki_link_always_include_similarity=threshold,
        )


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_wiki_orchestrator_config_rejects_nonfinite_environment_float(
    monkeypatch,
    tmp_path,
    value,
) -> None:
    monkeypatch.setenv("WIKI_GRAPH_PAGERANK_ALPHA", value)

    with pytest.raises(ValueError, match="WIKI_GRAPH_PAGERANK_ALPHA must be finite"):
        WikiOrchestratorConfig.from_env(tmp_path / "repository")


def test_wiki_orchestrator_construction_does_not_access_docker(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        extraction_module,
        "docker_from_env",
        lambda: pytest.fail("construction must not access Docker"),
    )

    _make_wiki_orchestrator(
        repository_path=tmp_path / "repository",
        extractor_model="extractor",
        critic_model="critic",
        fixer_model="fixer",
    )


def test_operational_settings_do_not_change_page_fingerprint(tmp_path) -> None:
    """Check that execution policy does not change content compatibility."""

    def fingerprint(
        *,
        page_workers: int = 4,
        transient_retries: int = 2,
        finalizer_model: str = "finalizer-a",
        finalizer_policy: tuple[int, float] = (20, 900.0),
        max_correction_runs: int = 5,
    ) -> str:
        with WikiOrchestrator(
            WikiOrchestratorConfig(
                repository_path=tmp_path / "repository",
                extractor_model="extractor",
                critic_model="critic",
                fixer_model="fixer",
                wiki_page_workers=page_workers,
                wiki_page_transient_retries=transient_retries,
                wiki_finalizer_model=finalizer_model,
                wiki_finalizer_batch_size=finalizer_policy[0],
                wiki_finalizer_timeout_seconds=finalizer_policy[1],
                max_wiki_turns=max_correction_runs,
            ),
        ) as orchestrator:
            return orchestrator.page_generation_config_fingerprint

    baseline = fingerprint()

    assert fingerprint(page_workers=12) == baseline
    assert fingerprint(transient_retries=0) == baseline
    assert fingerprint(finalizer_model="finalizer-b") == baseline
    assert fingerprint(finalizer_policy=(5, 60.0)) == baseline
    assert fingerprint(max_correction_runs=9) == baseline


def test_experimentalist_orchestrator_construction_does_not_access_docker(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "wiki_agents.orchestrator.experimentalist.docker_from_env",
        lambda: pytest.fail("construction must not access Docker"),
    )
    config = ExperimentalistOrchestratorConfig(
        wiki_path=tmp_path / "wiki",
        campaign_path=tmp_path / "campaign",
        task="Find useful experiments.",
    )

    ExperimentalistOrchestrator(config)


def test_wiki_orchestrator_owns_a_concept_extraction_pipeline(tmp_path) -> None:
    orchestrator = _make_wiki_orchestrator(
        repository_path=tmp_path / "repository",
        extractor_model="extractor",
        critic_model="critic",
        fixer_model="fixer",
    )

    assert not isinstance(orchestrator, ConceptExtractionPipeline)
    assert isinstance(orchestrator.concept_pipeline, ConceptExtractionPipeline)


@pytest.mark.parametrize("kind", ["missing", "directory", "text"])
def test_parse_document_rejects_invalid_input(monkeypatch, tmp_path, kind) -> None:
    orchestrator = _make_wiki_orchestrator(
        extractor_model="test-model",
        critic_model="test-model",
        fixer_model="test-model",
        repository_path=tmp_path / ".wiki",
    )
    path = tmp_path / "source.pdf"
    if kind == "directory":
        path.mkdir()
    elif kind == "text":
        path = tmp_path / "source.txt"
        path.write_text("not a PDF")

    message = "is not a PDF file" if kind == "text" else "is not a file"
    with pytest.raises(ValueError, match=message):
        orchestrator.concept_pipeline.parse_document(path)


def test_parse_document_does_not_publish_a_tracked_source(
    monkeypatch, tmp_path,
) -> None:
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"PDF")
    monkeypatch.setattr(
        extraction_module,
        "pdf_to_okf_source",
        lambda value: ("---\ntitle: Paper\n---", ["page one", "page two"]),
    )
    orchestrator = _make_wiki_orchestrator(
        extractor_model="test-model",
        critic_model="test-model",
        fixer_model="test-model",
        repository_path=tmp_path / ".wiki",
    )

    pages = orchestrator.concept_pipeline.parse_document(path)

    assert pages == ["page one", "page two"]
    assert not (orchestrator.sources_path / "paper.md").exists()


def test_parse_document_reuses_unchanged_parsed_source(monkeypatch, tmp_path) -> None:
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"PDF")
    parsed = ("frontmatter", ["page one"])
    monkeypatch.setattr(extraction_module, "pdf_to_okf_source", lambda value: parsed)
    orchestrator = _make_wiki_orchestrator(
        extractor_model="test-model",
        critic_model="test-model",
        fixer_model="test-model",
        repository_path=tmp_path / ".wiki",
    )
    orchestrator.concept_pipeline.parse_document(path)
    source_path = orchestrator.sources_path / "paper.md"
    original_write_text = Path.write_text
    writes = []

    def record_write(
        target: Path,
        content: str,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> int:
        if target == source_path:
            writes.append(content)
        return original_write_text(
            target,
            content,
            encoding=encoding,
            errors=errors,
            newline=newline,
        )

    monkeypatch.setattr(Path, "write_text", record_write)

    assert orchestrator.concept_pipeline.parse_document(path) == ["page one"]
    assert writes == []


def test_parse_document_returns_changed_parser_output_without_writing(
    monkeypatch,
    tmp_path,
) -> None:
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"PDF")
    parsed_results = iter(
        [("frontmatter", ["old page"]), ("frontmatter", ["new page"])],
    )
    monkeypatch.setattr(
        extraction_module,
        "pdf_to_okf_source",
        lambda value: next(parsed_results),
    )
    orchestrator = _make_wiki_orchestrator(
        extractor_model="test-model",
        critic_model="test-model",
        fixer_model="test-model",
        repository_path=tmp_path / ".wiki",
    )
    orchestrator.concept_pipeline.parse_document(path)

    assert orchestrator.concept_pipeline.parse_document(path) == ["new page"]
    assert not (orchestrator.sources_path / "paper.md").exists()


@pytest.fixture(autouse=True)
def _disable_external_tracing(monkeypatch):
    monkeypatch.setattr(tracing_module, "_INITIALIZED", True)
    monkeypatch.setattr(tracing_module, "_TRACING_ACTIVE", False)


def test_weave_replaces_the_default_agents_trace_exporter(monkeypatch) -> None:
    events = []
    monkeypatch.setattr(tracing_module, "_INITIALIZED", False)
    monkeypatch.setattr(
        tracing_module,
        "weave",
        SimpleNamespace(
            init=lambda project, **kwargs: events.append(("weave", project, kwargs)),
        ),
    )
    monkeypatch.setattr(
        tracing_module,
        "set_trace_processors",
        lambda processors: events.append(("processors", processors)),
    )
    monkeypatch.setenv("WEAVE_PROJECT", "test-project")

    tracing_module.init_weave_tracing()
    tracing_module.init_weave_tracing()

    assert events == [
        ("processors", []),
        (
            "weave",
            "test-project",
            {"settings": {"log_level": "WARNING", "print_call_link": False}},
        ),
    ]


def test_missing_weave_disables_external_tracing(monkeypatch) -> None:
    events = []
    monkeypatch.setattr(tracing_module, "_INITIALIZED", False)
    monkeypatch.setattr(tracing_module, "_TRACING_ACTIVE", False)
    monkeypatch.setattr(tracing_module, "weave", None)
    monkeypatch.setattr(
        tracing_module,
        "set_trace_processors",
        lambda processors: events.append(("processors", processors)),
    )

    tracing_module.init_weave_tracing()
    tracing_module.init_weave_tracing()

    assert events == [("processors", [])]
    assert tracing_module._INITIALIZED
    assert not tracing_module._TRACING_ACTIVE


def _attempt() -> ExtractionAttempt:
    return ExtractionAttempt(
        raw_output='{"concepts": []}',
        extraction=ConceptExtraction(concepts=[]),
    )


def _critique(decision: CritiqueDecision) -> ConceptCritique:
    marker = f"<decision>{decision.value}</decision>"
    return ConceptCritique(
        decision=decision,
        feedback="# Summary\n\nTest feedback.",
        raw_output=f"# Summary\n\nTest feedback.\n\n{marker}",
    )


def _orchestrator(*, max_concept_turns: int = 5) -> WikiOrchestrator:
    return _make_wiki_orchestrator(
        repository_path=Path(tempfile.mkdtemp(prefix="wiki-agent-test-")),
        extractor_model="test-model",
        critic_model="test-model",
        fixer_model="test-model",
        max_concept_turns=max_concept_turns,
    )


def test_close_closes_owned_qdrant_client_once(monkeypatch) -> None:
    class Client:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    client = Client()
    monkeypatch.setattr(concept_store_module, "QdrantClient", lambda **kwargs: client)

    orchestrator = _orchestrator()
    orchestrator._get_concept_store()
    orchestrator.close()
    orchestrator.close()

    assert client.close_calls == 1


def test_context_manager_does_not_close_caller_owned_store(tmp_path) -> None:
    class Store:
        def __init__(self) -> None:
            self.close_calls = 0

        def replace_source_candidates(self, source, candidates) -> None:
            raise AssertionError("context management must not use the concept store")

        def top_k_similar_concepts(self, candidate, *, top_k):
            raise AssertionError("context management must not use the concept store")

        def close(self) -> None:
            self.close_calls += 1

    store = Store()
    with _make_wiki_orchestrator(
        extractor_model="test-model",
        critic_model="test-model",
        fixer_model="test-model",
        concept_store=store,
        repository_path=tmp_path / "repository",
    ):
        pass

    assert store.close_calls == 0


def test_concept_extraction_stops_when_critic_accepts(monkeypatch) -> None:
    orchestrator = _orchestrator()
    attempt = _attempt()
    fixer_called = False

    monkeypatch.setattr(
        concept_workflow_module,
        "run_concept_extractor_agent",
        lambda **kwargs: attempt,
    )
    monkeypatch.setattr(
        concept_workflow_module,
        "run_concept_critic_agent",
        lambda **kwargs: _critique(CritiqueDecision.ACCEPT),
    )

    def fail_if_called(**kwargs):
        nonlocal fixer_called
        fixer_called = True
        return attempt

    monkeypatch.setattr(
        concept_workflow_module,
        "run_concept_fixer_agent",
        fail_if_called,
    )

    orchestrator.concept_pipeline.run_concept_extraction("paper.pdf")

    assert fixer_called is False


def test_concept_extraction_retries_when_critic_requests_revision(
    monkeypatch,
) -> None:
    orchestrator = _orchestrator()
    initial_attempt = _attempt()
    fixed_attempt = _attempt()
    critiques = iter(
        [
            _critique(CritiqueDecision.REVISE),
            _critique(CritiqueDecision.ACCEPT),
        ],
    )
    fixed_inputs = []
    fixed_feedback = []

    monkeypatch.setattr(
        concept_workflow_module,
        "run_concept_extractor_agent",
        lambda **kwargs: initial_attempt,
    )
    monkeypatch.setattr(
        concept_workflow_module,
        "run_concept_critic_agent",
        lambda **kwargs: next(critiques),
    )

    def fix(**kwargs):
        fixed_inputs.append(kwargs["extraction_attempt"])
        fixed_feedback.append(kwargs["critic_feedback"])
        return fixed_attempt

    monkeypatch.setattr(concept_workflow_module, "run_concept_fixer_agent", fix)

    orchestrator.concept_pipeline.run_concept_extraction("paper.pdf")

    assert fixed_inputs == [initial_attempt]
    assert fixed_feedback == ["# Summary\n\nTest feedback."]


def test_concept_extraction_uses_a_fresh_turn_limit_for_each_call(
    monkeypatch,
) -> None:
    orchestrator = _orchestrator(max_concept_turns=1)
    attempt = _attempt()
    critiques = iter(
        [
            _critique(CritiqueDecision.REVISE),
            _critique(CritiqueDecision.ACCEPT),
            _critique(CritiqueDecision.REVISE),
            _critique(CritiqueDecision.ACCEPT),
        ],
    )
    fixer_calls = []

    monkeypatch.setattr(
        concept_workflow_module,
        "run_concept_extractor_agent",
        lambda **kwargs: attempt,
    )
    monkeypatch.setattr(
        concept_workflow_module,
        "run_concept_critic_agent",
        lambda **kwargs: next(critiques),
    )
    monkeypatch.setattr(
        concept_workflow_module,
        "run_concept_fixer_agent",
        lambda **kwargs: fixer_calls.append(kwargs["anchored_text"]) or attempt,
    )

    first = orchestrator.concept_pipeline.run_concept_extraction("page one")
    second = orchestrator.concept_pipeline.run_concept_extraction("page two")

    assert first == attempt.extraction
    assert second == attempt.extraction
    assert fixer_calls == ["page one", "page two"]


def test_concept_extraction_retry_resumes_from_latest_reviewed_output(
    monkeypatch,
) -> None:
    orchestrator = _orchestrator()
    latest_attempt = _attempt()
    fixed_attempt = _attempt()
    fixer_inputs = []

    def extractor_must_not_run(**kwargs):
        raise AssertionError("retry unexpectedly started a new extraction")

    def fix(**kwargs):
        fixer_inputs.append(
            (kwargs["extraction_attempt"], kwargs["critic_feedback"]),
        )
        return fixed_attempt

    monkeypatch.setattr(
        concept_workflow_module,
        "run_concept_extractor_agent",
        extractor_must_not_run,
    )
    monkeypatch.setattr(concept_workflow_module, "run_concept_fixer_agent", fix)
    monkeypatch.setattr(
        concept_workflow_module,
        "run_concept_critic_agent",
        lambda **kwargs: _critique(CritiqueDecision.ACCEPT),
    )

    result = orchestrator.concept_pipeline.run_concept_extraction(
        "page one",
        retry_state=ConceptExtractionRetryState(
            raw_output=latest_attempt.raw_output,
            extraction=latest_attempt.extraction,
            critic_feedback="Fix the latest candidate.",
        ),
    )

    assert result == fixed_attempt.extraction
    assert fixer_inputs == [(latest_attempt, "Fix the latest candidate.")]


def test_concept_extraction_rejects_invalid_attempt_accepted_by_critic(
    monkeypatch,
) -> None:
    orchestrator = _orchestrator()
    invalid_attempt = ExtractionAttempt(raw_output="not JSON", extraction=None)

    monkeypatch.setattr(
        concept_workflow_module,
        "run_concept_extractor_agent",
        lambda **kwargs: invalid_attempt,
    )
    monkeypatch.setattr(
        concept_workflow_module,
        "run_concept_critic_agent",
        lambda **kwargs: _critique(CritiqueDecision.ACCEPT),
    )

    with pytest.raises(RuntimeError, match="schema or evidence errors"):
        orchestrator.concept_pipeline.run_concept_extraction("anchored source")


def test_concept_extraction_keeps_existing_turn_limit(monkeypatch) -> None:
    orchestrator = _orchestrator(max_concept_turns=1)
    attempt = _attempt()

    monkeypatch.setattr(
        concept_workflow_module,
        "run_concept_extractor_agent",
        lambda **kwargs: attempt,
    )
    monkeypatch.setattr(
        concept_workflow_module,
        "run_concept_critic_agent",
        lambda **kwargs: _critique(CritiqueDecision.REVISE),
    )
    monkeypatch.setattr(
        concept_workflow_module,
        "run_concept_fixer_agent",
        lambda **kwargs: attempt,
    )

    with pytest.raises(
        ConceptExtractionExhausted,
        match="failed after 1 turns",
    ) as exc_info:
        orchestrator.concept_pipeline.run_concept_extraction("paper.pdf")

    assert exc_info.value.retry_state is not None
    assert exc_info.value.retry_state.raw_output == attempt.raw_output
    assert exc_info.value.retry_state.critic_feedback == "# Summary\n\nTest feedback."


def test_extraction_finishes_before_embedding_and_wiki_input_preparation(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    events = []

    class Store:
        def replace_source_candidates(self, source, candidates):
            events.append(("replace", list(candidates)))

        def top_k_similar_concepts(self, candidate, *, top_k):
            events.append(("search", candidate.concept.title, top_k))
            return []

    path = tmp_path / "paper.pdf"
    path.write_bytes(b"test PDF bytes")
    orchestrator = _make_wiki_orchestrator(
        repository_path=tmp_path / "repository",
        extractor_model="test-model",
        critic_model="test-model",
        fixer_model="test-model",
        concept_store=Store(),
        num_concept_pages=1,
        similar_concepts_top_k=4,
    )
    assert orchestrator.concept_pipeline.config.num_concept_pages == 1
    assert orchestrator.concept_pipeline.metadata["chunk_size"] == 1
    pages = ['<a id="p0001-l0001"></a> First', ""]
    concepts = iter(
        [
            ConceptExtraction(
                concepts=[
                    Concept(
                        title="Alpha",
                        description="Description",
                        claims=[
                            Claim(
                                statement="Claim",
                                evidence=[
                                    EvidenceSpan(
                                        start_line_id="p0001-l0001",
                                        end_line_id="p0001-l0001",
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )
    monkeypatch.setattr(
        orchestrator.concept_pipeline,
        "parse_document",
        lambda value: pages,
    )

    def extract(text):
        events.append(("extract", text))
        return next(concepts)

    monkeypatch.setattr(
        orchestrator.concept_pipeline,
        "run_concept_extraction",
        extract,
    )
    result = orchestrator.concept_pipeline.extract_concepts(path)

    assert [event[0] for event in events] == ["extract"]
    assert result.source.resource == "/references/papers/paper_paper.md"
    assert result.source.sha256
    assert [page.page_number for page in result.pages] == [1, 2]
    assert result.pages[0].text == pages[0]
    assert len(result.candidates) == 1
    assert result.candidates[0].concept.title == "Alpha"
    assert result.concept_extraction_status == "succeeded"
    assert [chunk.status for chunk in result.concept_extraction_chunks] == ["succeeded"]

    orchestrator.concept_pipeline.update_concept_embeddings([result])
    wiki_inputs = orchestrator.prepare_wiki_input([result])

    assert [event[0] for event in events] == [
        "extract",
        "replace",
        "search",
    ]
    assert len(wiki_inputs) == 1
    assert wiki_inputs[0].candidate == result.candidates[0]
    assert wiki_inputs[0].similar_concepts == []


def test_extract_concepts_continues_and_retries_only_failed_chunks(
    monkeypatch,
    tmp_path,
    caplog,
) -> None:
    caplog.set_level(logging.INFO, logger="wiki_agents.orchestrator")

    class Store:
        def replace_source_candidates(self, source, candidates):
            pass

        def top_k_similar_concepts(self, candidate, *, top_k):
            return []

    path = tmp_path / "paper.pdf"
    path.write_bytes(b"test PDF bytes")
    orchestrator = _make_wiki_orchestrator(
        extractor_model="test-model",
        critic_model="test-model",
        fixer_model="test-model",
        concept_store=Store(),
        repository_path=tmp_path / ".wiki",
        num_concept_pages=1,
    )
    pages = ["First page", "Second page", "Third page"]
    monkeypatch.setattr(
        orchestrator.concept_pipeline,
        "parse_document",
        lambda value: pages,
    )
    first_run_calls = []

    def first_run_extract(text):
        first_run_calls.append(text)
        if text == "First page":
            raise ConceptExtractionExhausted(
                5,
                ConceptExtractionRetryState(
                    raw_output='{"concepts": []}',
                    extraction=ConceptExtraction(concepts=[]),
                    critic_feedback="Retry this candidate.",
                ),
            )
        return ConceptExtraction(concepts=[])

    monkeypatch.setattr(
        orchestrator.concept_pipeline,
        "run_concept_extraction",
        first_run_extract,
    )

    first_result = orchestrator.concept_pipeline.extract_concepts(path)

    assert first_run_calls == pages
    assert first_result.concept_extraction_status == "partial"
    assert [chunk.status for chunk in first_result.concept_extraction_chunks] == [
        "failed",
        "succeeded",
        "succeeded",
    ]
    failed = first_result.concept_extraction_chunks[0]
    assert failed.page_numbers == [1]
    assert failed.error_code == "revision_limit_exceeded"
    assert failed.run_attempts == 1

    retry_calls = []

    def retry_extract(text, *, retry_state=None):
        assert retry_state is not None
        retry_calls.append(text)
        return ConceptExtraction(concepts=[])

    monkeypatch.setattr(
        orchestrator.concept_pipeline,
        "run_concept_extraction",
        retry_extract,
    )

    retry_result = orchestrator.concept_pipeline.extract_concepts(path)

    assert retry_calls == ["First page"]
    assert retry_result.concept_extraction_status == "succeeded"
    assert [chunk.run_attempts for chunk in retry_result.concept_extraction_chunks] == [
        2,
        1,
        1,
    ]
    assert "pages 1 is resuming attempt 2 from its latest candidate" in caplog.text
    assert "pages 2 reused its successful checkpoint" in caplog.text
    assert "finished with status=succeeded" in caplog.text


def test_extract_concepts_does_not_swallow_unexpected_errors(
    monkeypatch,
    tmp_path,
) -> None:
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"test PDF bytes")
    orchestrator = _make_wiki_orchestrator(
        extractor_model="test-model",
        critic_model="test-model",
        fixer_model="test-model",
        concept_store=None,
        repository_path=tmp_path / ".wiki",
    )
    monkeypatch.setattr(
        orchestrator.concept_pipeline,
        "parse_document",
        lambda value: ["Page"],
    )

    def fail_unexpectedly(text):
        raise ValueError("bad configuration")

    monkeypatch.setattr(
        orchestrator.concept_pipeline,
        "run_concept_extraction",
        fail_unexpectedly,
    )

    with pytest.raises(ValueError, match="bad configuration"):
        orchestrator.concept_pipeline.extract_concepts(path)


def test_load_concept_checkpoint_rejects_corrupt_json(tmp_path) -> None:
    checkpoint_store = ConceptCheckpointStore(tmp_path, "fingerprint")
    source = SourceRef(
        source_id="source",
        sha256="a" * 64,
        resource="/references/source.md",
    )
    checkpoint_store.path_for(source).write_text("{not valid JSON")

    with pytest.raises(ValueError, match="Invalid JSON"):
        checkpoint_store.load(source)


def test_concept_checkpoint_path_names_each_identity_level(tmp_path) -> None:
    checkpoint_store = ConceptCheckpointStore(tmp_path, "pipeline-fingerprint")
    source = SourceRef(
        source_id="paper-123",
        sha256="a" * 64,
        resource="/references/paper-123.md",
    )

    expected_path = (
        tmp_path
        / "concepts"
        / "paper-123"
        / ("a" * 64)
        / "pipeline-fingerprint"
        / "extraction.json"
    )
    assert checkpoint_store.path_for(source).as_posix() == expected_path.as_posix()


def test_load_concept_checkpoint_rejects_source_mismatch(tmp_path) -> None:
    checkpoint_store = ConceptCheckpointStore(tmp_path, "fingerprint")
    expected_source = SourceRef(
        source_id="source",
        sha256="a" * 64,
        resource="/references/source.md",
    )
    saved_source = SourceRef(
        source_id="different",
        sha256="b" * 64,
        resource="/references/different.md",
    )
    checkpoint = ConceptExtractionCheckpoint(source=saved_source, chunks=[])
    checkpoint_store.path_for(expected_source).write_text(
        checkpoint.model_dump_json(),
    )

    with pytest.raises(ValueError, match="checkpoint source mismatch"):
        checkpoint_store.load(expected_source)


def test_incremental_checkpoint_preserves_unvisited_chunks_on_error(
    monkeypatch,
    tmp_path,
) -> None:
    checkpoint_store = ConceptCheckpointStore(tmp_path, "fingerprint")
    source = SourceRef(
        source_id="source",
        sha256="a" * 64,
        resource="/references/source.md",
    )

    def failed_chunk(chunk_id, page_number):
        return ConceptExtractionChunk(
            chunk_id=chunk_id,
            page_numbers=[page_number],
            status="failed",
            run_attempts=1,
            error_code="revision_limit_exceeded",
            error_message="try again",
        )

    previous = {
        "second": failed_chunk("second", 2),
        "third": failed_chunk("third", 3),
    }
    checkpoint_store.save(source, list(previous.values()))
    work_items = [
        ConceptChunkWork("first", [1], "first page"),
        ConceptChunkWork("second", [2], "second page"),
        ConceptChunkWork("third", [3], "third page"),
    ]
    first_result = ConceptExtractionChunk(
        chunk_id="first",
        page_numbers=[1],
        status="succeeded",
        run_attempts=1,
        extraction=ConceptExtraction(concepts=[]),
    )
    calls = 0

    def extract(text, *, retry_state=None):
        return ConceptExtraction(concepts=[])

    runner = ConceptChunkRunner(extract, checkpoint_store)

    def run_chunk(source, work, prior):
        nonlocal calls
        assert source.source_id == "source"
        calls += 1
        if calls == 1:
            assert work.chunk_id == "first"
            assert prior is None
            return first_result
        raise RuntimeError("interrupted")

    monkeypatch.setattr(
        runner,
        "_run_chunk",
        run_chunk,
    )

    with pytest.raises(RuntimeError, match="interrupted"):
        runner.run(source, work_items)

    saved = checkpoint_store.load(source)
    assert [chunk.chunk_id for chunk in saved.chunks] == [
        "first",
        "second",
        "third",
    ]
    assert saved.chunks[0] == first_result
    assert saved.chunks[1:] == [previous["second"], previous["third"]]


def test_run_accepts_an_initialized_repository_without_pdfs(tmp_path) -> None:
    orchestrator = _make_wiki_orchestrator(
        repository_path=tmp_path / "repository",
        extractor_model="test-model",
        critic_model="test-model",
        fixer_model="test-model",
    )
    orchestrator.run()

    assert orchestrator.wiki_repository.repo.head.commit.message == (
        "chore: initialize wiki repository"
    )


def test_run_collects_all_extractions_before_corpus_mapping(
    monkeypatch,
    tmp_path,
) -> None:
    orchestrator = _make_wiki_orchestrator(
        repository_path=tmp_path / "repository",
        extractor_model="test-model",
        critic_model="test-model",
        fixer_model="test-model",
    )
    extracted_sources = [
        SimpleNamespace(candidates=[]),
        SimpleNamespace(candidates=[]),
    ]
    events = []

    def extract():
        events.append("extract")
        return extracted_sources

    monkeypatch.setattr(orchestrator, "extract_concepts", extract)
    monkeypatch.setattr(
        orchestrator_module,
        "init_weave_tracing",
        lambda: events.append("trace"),
    )
    assert not hasattr(orchestrator, "run_wiki_creation")

    orchestrator.run()

    assert events == ["trace", "extract"]


def test_extract_concepts_limits_concurrent_chunks_across_pdfs(
    monkeypatch,
    tmp_path,
) -> None:
    orchestrator = _make_wiki_orchestrator(
        repository_path=tmp_path / "repository",
        extractor_model="test-model",
        critic_model="test-model",
        fixer_model="test-model",
        num_extraction_workers=2,
        num_concept_pages=1,
    )
    paths = [
        orchestrator.sources_path / "first.pdf",
        orchestrator.sources_path / "second.pdf",
    ]
    for path in paths:
        path.write_bytes(b"PDF")

    barrier = Barrier(2)
    lock = Lock()
    active = 0
    maximum_active = 0
    extraction_thread_ids = set()
    checkpoint_thread_ids: dict[str, set[int]] = {}

    monkeypatch.setattr(
        orchestrator.concept_pipeline,
        "parse_document",
        lambda path: ["First page", "Second page"],
    )

    def extract(text):
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
            extraction_thread_ids.add(get_ident())
        try:
            barrier.wait(timeout=2)
            return ConceptExtraction(concepts=[])
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(
        orchestrator.concept_pipeline,
        "run_concept_extraction",
        extract,
    )
    checkpoint_store = orchestrator.concept_pipeline._checkpoint_store
    original_save = checkpoint_store.save

    def record_save(source, chunks):
        checkpoint_thread_ids.setdefault(source.source_id, set()).add(get_ident())
        original_save(source, chunks)

    monkeypatch.setattr(checkpoint_store, "save", record_save)

    results = orchestrator._extract_pending_concurrently(paths)

    assert maximum_active == 2
    assert all(result is not None for result in results.values())
    assert all(
        [chunk.page_numbers for chunk in result.concept_extraction_chunks] == [[1], [2]]
        for result in results.values()
        if result is not None
    )
    assert all(len(thread_ids) == 1 for thread_ids in checkpoint_thread_ids.values())
    assert not extraction_thread_ids.intersection(
        set().union(*checkpoint_thread_ids.values()),
    )


def test_completed_pdf_is_published_while_another_pdf_is_extracting(
    monkeypatch,
    tmp_path,
) -> None:
    orchestrator = _make_wiki_orchestrator(
        repository_path=tmp_path / "repository",
        extractor_model="test-model",
        critic_model="test-model",
        fixer_model="test-model",
        num_extraction_workers=2,
    )
    slow_path = orchestrator.sources_path / "slow.pdf"
    fast_path = orchestrator.sources_path / "fast.pdf"
    slow_started = Event()
    fast_published = Event()
    published = []

    def extract(path, *, executor):
        if path == slow_path:
            slow_started.set()
            assert fast_published.wait(timeout=2)
        else:
            assert slow_started.wait(timeout=2)
        return object()

    def publish(path, extracted):
        published.append(path)
        if path == fast_path:
            fast_published.set()

    monkeypatch.setattr(orchestrator.ingestion, "extract", extract)
    monkeypatch.setattr(orchestrator.ingestion, "publish", publish)

    orchestrator._extract_pending_concurrently([slow_path, fast_path])

    assert published == [fast_path, slow_path]


def test_extract_concepts_checks_docker_before_submitting_work(
    monkeypatch,
    tmp_path,
) -> None:
    orchestrator = _make_wiki_orchestrator(
        repository_path=tmp_path / "repository",
        extractor_model="test-model",
        critic_model="test-model",
        fixer_model="test-model",
    )
    (orchestrator.sources_path / "paper.pdf").write_bytes(b"PDF")
    extraction_started = False

    def reject_docker():
        raise RuntimeError("Docker is stopped")

    def extract(pending):
        nonlocal extraction_started
        extraction_started = True
        return {}

    monkeypatch.setattr(orchestrator, "_assert_docker_running", reject_docker)
    monkeypatch.setattr(
        orchestrator.ingestion, "requires_extraction", lambda path: True,
    )
    monkeypatch.setattr(orchestrator, "_extract_pending_concurrently", extract)

    with pytest.raises(RuntimeError, match="Docker is stopped"):
        orchestrator.extract_concepts()

    assert not extraction_started


def test_extract_concepts_skips_docker_when_all_outputs_are_current(
    monkeypatch,
    tmp_path,
) -> None:
    orchestrator = _make_wiki_orchestrator(
        repository_path=tmp_path / "repository",
        extractor_model="test-model",
        critic_model="test-model",
        fixer_model="test-model",
    )
    (orchestrator.sources_path / "paper.pdf").write_bytes(b"PDF")

    monkeypatch.setattr(
        orchestrator.ingestion,
        "requires_extraction",
        lambda path: False,
    )
    monkeypatch.setattr(
        orchestrator,
        "_assert_docker_running",
        lambda: pytest.fail("Docker must not be checked for current outputs"),
    )
    monkeypatch.setattr(orchestrator, "_extract_pending_concurrently", lambda paths: {})

    assert orchestrator.extract_concepts() == []


def test_docker_preflight_reports_an_unavailable_daemon(monkeypatch) -> None:
    def fail_connection():
        raise extraction_module.DockerException("socket unavailable")

    monkeypatch.setattr(extraction_module, "docker_from_env", fail_connection)

    with pytest.raises(
        RuntimeError,
        match="Start Docker and make its socket accessible",
    ):
        WikiOrchestrator._assert_docker_running()


def test_pending_pdfs_only_uses_drop_folder_and_canonical_artifacts(tmp_path) -> None:
    sources_path = tmp_path / "existing-papers"
    sources_path.mkdir()
    orchestrator = _make_wiki_orchestrator(
        repository_path=tmp_path / "repository",
        sources_path=sources_path,
        extractor_model="test-model",
        critic_model="test-model",
        fixer_model="test-model",
    )
    queued = sources_path / "queued.pdf"
    legacy = (
        orchestrator.repository_path / "wiki/references/papers/artifacts/legacy.PDF"
    )
    queued.write_bytes(b"queued")
    legacy.write_bytes(b"legacy")
    linked = (
        orchestrator.repository_path / "wiki/references/papers/artifacts/linked.pdf"
    )
    linked.symlink_to(queued)
    (sources_path / "ignored.txt").write_text("ignored")
    (orchestrator.repository_path / "sources/not-selected.pdf").write_bytes(b"ignored")

    assert orchestrator._pending_pdfs() == sorted(
        [queued.resolve(), legacy.resolve()],
        key=str,
    )


def test_pending_pdf_names_must_be_unique_across_source_folders(tmp_path) -> None:
    first = tmp_path / "first/A paper.pdf"
    second = tmp_path / "second/a_paper.pdf"

    with pytest.raises(
        ValueError,
        match="PDF filename normalization collision: paper_a_paper",
    ):
        WikiOrchestrator._validate_pending_names([first, second])


def test_corrupt_snapshot_is_reported_as_a_load_failure(
    monkeypatch,
    tmp_path,
    caplog,
) -> None:
    orchestrator = _make_wiki_orchestrator(
        repository_path=tmp_path / "repository",
        extractor_model="test-model",
        critic_model="test-model",
        fixer_model="test-model",
    )
    snapshot = orchestrator.repository_path / "extractions/corrupt.json"
    snapshot.write_text("{not valid JSON")
    monkeypatch.setattr(orchestrator.wiki_repository, "assert_clean", lambda: None)
    caplog.set_level(logging.ERROR, logger="wiki_agents.orchestrator")

    assert orchestrator.extract_concepts() == []
    assert f"Could not load extraction snapshot {snapshot}" in caplog.text
    assert "Could not synchronize Qdrant" not in caplog.text
