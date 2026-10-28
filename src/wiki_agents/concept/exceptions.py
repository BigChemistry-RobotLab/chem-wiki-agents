"""Exceptions raised by concept-processing workflows."""

from wiki_agents.concept.models import ConceptExtractionRetryState


class ConceptExtractionExhausted(  # ruff: ignore[error-suffix-on-exception-name] - Preserve the public API.
    RuntimeError,
):
    """Raised when a chunk cannot pass review within its revision budget."""

    error_code = "revision_limit_exceeded"

    def __init__(
        self,
        turns: int,
        retry_state: ConceptExtractionRetryState | None = None,
    ) -> None:
        self.turns = turns
        self.retry_state = retry_state
        super().__init__(f"Concept extraction failed after {turns} turns")
