from wiki_agents.agents.concept.critic import (
    ConceptCritique,
    CritiqueDecision,
    create_concept_critic_agent,
    run_concept_critic_agent,
)
from wiki_agents.agents.concept.extractor import (
    ConceptExtraction,
    ExtractionAttempt,
    create_concept_extractor_agent,
    run_concept_extractor_agent,
)
from wiki_agents.agents.concept.fixer import (
    create_concept_fixer_agent,
    run_concept_fixer_agent,
)
from wiki_agents.concept.exceptions import ConceptExtractionExhausted
from wiki_agents.concept.models import ConceptExtractionRetryState
from wiki_agents.tracing import init_weave_tracing, record_trace_event


class ConceptExtractionWorkflow:
    """Coordinate extraction, critique, and revision for one source chunk."""

    def __init__(
        self,
        *,
        extractor_model: str,
        critic_model: str,
        fixer_model: str,
        max_revision_turns: int = 5,
    ) -> None:
        self.extractor_model = extractor_model
        self.critic_model = critic_model
        self.fixer_model = fixer_model
        self.max_revision_turns = max_revision_turns

    def run(
        self,
        anchored_text: str,
        *,
        retry_state: ConceptExtractionRetryState | None = None,
    ) -> ConceptExtraction:
        """Extract and review concepts, optionally resuming a previous failure."""
        init_weave_tracing()
        extractor_agent = create_concept_extractor_agent(model=self.extractor_model)
        critic_agent = create_concept_critic_agent(model=self.critic_model)
        fixer_agent = create_concept_fixer_agent(model=self.fixer_model)
        if retry_state is None:
            extraction_attempt = run_concept_extractor_agent(
                agent=extractor_agent,
                anchored_text=anchored_text,
            )
            critique_attempt = run_concept_critic_agent(
                agent=critic_agent,
                anchored_text=anchored_text,
                attempt=extraction_attempt,
            )
            self._record_critique(critique_attempt, revision=0)
        else:
            extraction_attempt = ExtractionAttempt(
                raw_output=retry_state.raw_output,
                extraction=retry_state.extraction,
            )
            critique_attempt = ConceptCritique(
                decision=CritiqueDecision.REVISE,
                feedback=retry_state.critic_feedback,
                raw_output=(
                    retry_state.critic_feedback + "\n\n<decision>REVISE</decision>"
                ),
            )

        revision_count = 0
        while critique_attempt.decision is CritiqueDecision.REVISE:
            if revision_count >= self.max_revision_turns:
                raise ConceptExtractionExhausted(
                    revision_count,
                    ConceptExtractionRetryState(
                        raw_output=extraction_attempt.raw_output,
                        extraction=extraction_attempt.extraction,
                        critic_feedback=critique_attempt.feedback,
                    ),
                )

            extraction_attempt = run_concept_fixer_agent(
                agent=fixer_agent,
                anchored_text=anchored_text,
                extraction_attempt=extraction_attempt,
                critic_feedback=critique_attempt.feedback,
            )
            critique_attempt = run_concept_critic_agent(
                agent=critic_agent,
                anchored_text=anchored_text,
                attempt=extraction_attempt,
            )
            revision_count += 1
            self._record_critique(critique_attempt, revision=revision_count)

        if not extraction_attempt.evidence_valid:
            raise RuntimeError(
                "Concept critic accepted an extraction with schema or evidence errors",
            )
        if extraction_attempt.extraction is None:
            raise RuntimeError("Concept extraction is unexpectedly missing")
        return extraction_attempt.extraction

    @staticmethod
    def _record_critique(critique: ConceptCritique, *, revision: int) -> None:
        """Add the parsed critic decision and feedback to the active trace."""
        record_trace_event(
            "Concept critic outcome",
            {
                "stage": "concept_critique",
                "revision": revision,
                "decision": critique.decision.value,
                "feedback": critique.feedback,
            },
        )
