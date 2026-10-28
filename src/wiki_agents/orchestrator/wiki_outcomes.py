"""Define public outcomes from Wiki orchestration."""

from dataclasses import dataclass

from wiki_agents.agents.wiki.workflow import WikiPageOutcome


@dataclass(frozen=True, slots=True)
class WikiCorpusOutcome:
    """Summarize one atomic full-corpus Wiki build."""

    pages: tuple[WikiPageOutcome, ...]
    commit_sha: str | None
