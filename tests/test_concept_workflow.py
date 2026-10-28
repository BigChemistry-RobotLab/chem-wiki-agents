from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event, Lock
from types import SimpleNamespace

import wiki_agents.concept.workflow as workflow_module
import wiki_agents.tracing as tracing_module
from wiki_agents.agents.concept.critic import ConceptCritique, CritiqueDecision
from wiki_agents.agents.concept.extractor import ExtractionAttempt
from wiki_agents.concept.models import ConceptExtraction
from wiki_agents.concept.workflow import ConceptExtractionWorkflow


def _critique(decision: CritiqueDecision) -> ConceptCritique:
    return ConceptCritique(
        decision=decision,
        feedback="Fix the extraction.",
        raw_output=f"<decision>{decision}</decision>",
    )


def test_concurrent_workflows_use_distinct_agent_instances(monkeypatch) -> None:
    workflow = ConceptExtractionWorkflow(
        extractor_model="extractor",
        critic_model="critic",
        fixer_model="fixer",
    )
    attempt = ExtractionAttempt(
        raw_output='{"concepts": []}',
        extraction=ConceptExtraction(concepts=[]),
    )
    fixer_barrier = Barrier(2)
    fixer_agents = []

    monkeypatch.setattr(workflow_module, "init_weave_tracing", lambda: None)
    monkeypatch.setattr(
        workflow_module,
        "run_concept_extractor_agent",
        lambda **kwargs: attempt,
    )
    monkeypatch.setattr(
        workflow_module,
        "run_concept_critic_agent",
        lambda **kwargs: _critique(
            CritiqueDecision.ACCEPT
            if kwargs["attempt"].raw_output == "fixed"
            else CritiqueDecision.REVISE,
        ),
    )

    def fix(**kwargs):
        fixer_agents.append(kwargs["agent"])
        fixer_barrier.wait(timeout=2)
        return ExtractionAttempt(
            raw_output="fixed",
            extraction=attempt.extraction,
        )

    monkeypatch.setattr(workflow_module, "run_concept_fixer_agent", fix)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(workflow.run, ["first", "second"]))

    assert results == [attempt.extraction, attempt.extraction]
    assert len(fixer_agents) == 2
    assert fixer_agents[0] is not fixer_agents[1]


def test_concurrent_tracing_initialization_runs_once(monkeypatch) -> None:
    initialization_started = Event()
    allow_initialization = Event()
    call_lock = Lock()
    initialization_calls = 0

    def initialize(project: str, **kwargs) -> None:
        nonlocal initialization_calls
        assert project
        assert kwargs["settings"]["print_call_link"] is False
        with call_lock:
            initialization_calls += 1
        initialization_started.set()
        assert allow_initialization.wait(timeout=2)

    monkeypatch.setattr(tracing_module, "_INITIALIZED", False)
    monkeypatch.setattr(tracing_module, "_TRACING_ACTIVE", False)
    monkeypatch.setattr(tracing_module, "set_trace_processors", lambda value: None)
    monkeypatch.setattr(tracing_module, "weave", SimpleNamespace(init=initialize))

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(tracing_module.init_weave_tracing) for _ in range(2)]
        assert initialization_started.wait(timeout=2)
        allow_initialization.set()
        for future in futures:
            future.result(timeout=2)

    assert initialization_calls == 1
