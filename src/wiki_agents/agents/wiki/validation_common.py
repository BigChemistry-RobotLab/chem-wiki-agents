"""Provide shared OKF document and validation helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from posixpath import dirname, normpath
from typing import TYPE_CHECKING, Any

import yaml

from wiki_agents.agents.wiki.models import (
    WikiCreationResult,
    WikiFileChange,
    WikiReviewFindingType,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

WIKI_ACTOR = "wiki-agents/0.1.0"
WIKI_REVIEWER_ACTOR = "wiki-agents-reviewer/0.1.0"
_FRONTMATTER = re.compile(r"\A---\r?\n(.*?)\r?\n---(?:\r?\n|\Z)", re.DOTALL)
_LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^)\s]+)(?:\s+['\"][^'\"]*['\"])?\)")
_FOOTNOTE_USE = re.compile(r"\[\^([^\]]+)\]")
_FOOTNOTE_DEFINITION = re.compile(r"(?m)^\[\^([^\]]+)\]:")
_FOOTNOTE_DEFINITION_LINE = re.compile(r"(?m)^\[\^([^\]]+)\]:[ \t]*(.*)$")
_EVIDENCE_ANCHOR = re.compile(
    r"p\d{4,}-(?:l\d{4,}|(?:figure|table|region)-\d{4,})",
)
_KNOWN_OKF_FIELDS = {
    "type",
    "title",
    "description",
    "resource",
    "tags",
    "sources",
    "usage_window",
    "generated",
    "verified",
    "status",
    "stale_after",
    "computation",
    "runtime",
    "parameters",
    "executor",
    "attester",
}
_RANKING_FRONTMATTER_FIELDS = {
    "global_pagerank",
    "graph_centrality",
    "pagerank",
    "personalized_pagerank",
    "ranking_score",
}
_RELATED_CONCEPTS_HEADING = re.compile(
    r"(?im)^#{1,6}[ \t]+(?:related concepts|see also)[ \t]*$",
)


@dataclass(frozen=True, slots=True)
class WikiValidationIssue:
    """Describe one deterministic defect in a Wiki proposal."""

    finding_type: WikiReviewFindingType
    target: str
    problem: str
    required_change: str
    basis: str

    def as_payload(self) -> dict[str, str]:
        """Return the issue in the creator correction format."""
        return {
            "finding_type": self.finding_type.value,
            "target": self.target,
            "problem": self.problem,
            "required_change": self.required_change,
            "basis": self.basis,
        }


def utc_timestamp() -> str:
    """Return one second-resolution UTC timestamp."""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_okf_document(content: bytes) -> tuple[dict[str, Any], str]:
    """Parse a UTF-8 OKF concept document.

    Args:
        content: Complete Markdown document bytes.

    Returns:
        The YAML frontmatter mapping and Markdown body.

    Raises:
        ValueError: The document is not UTF-8 or has invalid frontmatter.

    """
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("The concept document must use UTF-8") from error
    match = _FRONTMATTER.match(text)
    if match is None:
        raise ValueError("The concept document must start with YAML frontmatter")
    try:
        loaded = yaml.safe_load(match.group(1))
    except yaml.YAMLError as error:
        raise ValueError("The concept document has invalid YAML frontmatter") from error
    if not isinstance(loaded, dict) or not all(isinstance(key, str) for key in loaded):
        raise ValueError("The concept frontmatter must be a string-keyed mapping")
    return loaded, text[match.end() :]


def _has_detached_concept_link_section(body: str) -> bool:
    """Return whether the body puts concept links in a detached section."""
    return _RELATED_CONCEPTS_HEADING.search(body) is not None


def render_okf_document(frontmatter: Mapping[str, Any], body: str) -> bytes:
    """Render an OKF document with stable YAML frontmatter."""
    yaml_text = yaml.safe_dump(
        dict(frontmatter),
        allow_unicode=True,
        sort_keys=False,
    ).rstrip()
    normalized_body = body.lstrip("\r\n")
    suffix = "" if not normalized_body else f"\n\n{normalized_body.rstrip()}\n"
    return f"---\n{yaml_text}\n---{suffix}".encode()


def normalize_managed_metadata(
    result: WikiCreationResult,
    *,
    generated_at: str,
) -> WikiCreationResult:
    """Set creator metadata and remove verification of prior content."""
    change = _target_change(result)
    if change.content is None:
        return result
    frontmatter, body = parse_okf_document(change.content)
    frontmatter["generated"] = {"by": WIKI_ACTOR, "at": generated_at}
    frontmatter.pop("verified", None)
    return _replace_target_content(
        result,
        render_okf_document(frontmatter, body),
    )


def add_machine_verification(
    result: WikiCreationResult,
    *,
    verified_at: str,
) -> WikiCreationResult:
    """Add the machine review record to accepted content."""
    change = _target_change(result)
    if change.content is None:
        return result
    frontmatter, body = parse_okf_document(change.content)
    frontmatter["verified"] = [
        {"by": WIKI_REVIEWER_ACTOR, "at": verified_at},
    ]
    return _replace_target_content(
        result,
        render_okf_document(frontmatter, body),
    )


def _validate_required_frontmatter(
    target: str,
    frontmatter: Mapping[str, Any],
) -> list[WikiValidationIssue]:
    """Require managed concept metadata."""
    issues = []
    for field in ("type", "title", "description"):
        value = frontmatter.get(field)
        if not isinstance(value, str) or not value.strip():
            issues.append(
                WikiValidationIssue(
                    WikiReviewFindingType.OKF,
                    f"{target} frontmatter.{field}",
                    f"The `{field}` value is missing or empty.",
                    f"Set a non-empty `{field}` string.",
                    "Deterministic OKF frontmatter validation",
                ),
            )
    description = frontmatter.get("description")
    if isinstance(description, str) and (
        _LINK.search(description) is not None
        or _FOOTNOTE_USE.search(description) is not None
    ):
        issues.append(
            WikiValidationIssue(
                WikiReviewFindingType.OKF,
                f"{target} frontmatter.description",
                "The description contains Markdown citation markup.",
                "Use plain summary text and keep provenance in sources and the body.",
                "Repository producer rule for plain descriptions",
            ),
        )
    return issues


def _validate_ranking_frontmatter(
    target: str,
    frontmatter: Mapping[str, Any],
) -> list[WikiValidationIssue]:
    """Reject pipeline-only graph ranking metadata in a Wiki document."""
    fields = sorted(
        key for key in frontmatter if key.casefold() in _RANKING_FRONTMATTER_FIELDS
    )
    if not fields:
        return []
    return [
        WikiValidationIssue(
            WikiReviewFindingType.OKF,
            f"{target} frontmatter",
            "Pipeline graph ranking fields are present: " + ", ".join(fields) + ".",
            "Remove graph ranking metadata from OKF frontmatter.",
            "Graph ranking metadata is pipeline state only",
        ),
    ]


def _source_entries(raw_sources: list[object]) -> tuple[dict[str, str], bool]:
    """Return valid unique source entries and an invalid-entry flag."""
    source_entries: dict[str, str] = {}
    invalid_entry = False
    for entry in raw_sources:
        if not isinstance(entry, dict):
            invalid_entry = True
            continue
        source_id = entry.get("id")
        resource = entry.get("resource")
        if (
            not isinstance(source_id, str)
            or not source_id.strip()
            or not isinstance(resource, str)
            or not resource.strip()
            or source_id in source_entries
        ):
            invalid_entry = True
            continue
        source_entries[source_id] = resource
    return source_entries, invalid_entry


def _concept_links(body: str, document_path: str) -> set[str]:
    """Return normalized internal concept links from Markdown."""
    links: set[str] = set()
    bundle_document = "/" + document_path.removeprefix("wiki/")
    for raw_target in _LINK.findall(body):
        link_target = raw_target.split("#", maxsplit=1)[0]
        if link_target.startswith("/concepts/"):
            links.add(normpath(link_target))
        elif ":" not in link_target and not link_target.startswith(("/", "#")):
            resolved = normpath(f"{dirname(bundle_document)}/{link_target}")
            if resolved.startswith("/concepts/") and resolved.endswith(".md"):
                links.add(resolved)
    return links


def _target_change(result: WikiCreationResult) -> WikiFileChange:
    """Return the single target change from an active proposal."""
    if len(result.changes) != 1:
        raise ValueError("An active Wiki proposal must contain one change")
    return result.changes[0]


def _replace_target_content(
    result: WikiCreationResult,
    content: bytes,
) -> WikiCreationResult:
    """Replace normalized content without changing proposal identity."""
    change = _target_change(result)
    return WikiCreationResult(
        plan=result.plan,
        changes=(
            WikiFileChange(
                path=change.path,
                status=change.status,
                content=content,
            ),
        ),
        raw_output=result.raw_output,
    )
