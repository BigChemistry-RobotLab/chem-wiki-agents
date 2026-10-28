"""Validate one proposed Wiki concept page."""

# ruff: file-ignore[import-private-name]

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from wiki_agents.agents.wiki.models import (
    WikiChangeStatus,
    WikiCreationResult,
    WikiFileChange,
    WikiPlan,
    WikiReviewFindingType,
)
from wiki_agents.agents.wiki.validation_common import (
    _EVIDENCE_ANCHOR,
    _FOOTNOTE_DEFINITION,
    _FOOTNOTE_DEFINITION_LINE,
    _FOOTNOTE_USE,
    _KNOWN_OKF_FIELDS,
    _LINK,
    WikiValidationIssue,
    _concept_links,
    _has_detached_concept_link_section,
    _source_entries,
    _target_change,
    _validate_ranking_frontmatter,
    _validate_required_frontmatter,
    parse_okf_document,
)
from wiki_agents.agents.wiki.workspace import apply_wiki_changes, evidence_target

if TYPE_CHECKING:
    from collections.abc import Mapping

    from wiki_agents.agents.wiki.models import WikiPageContext, WikiPageManifest
    from wiki_agents.concept.models import Concept
    from wiki_agents.document.models import SourceRef


@dataclass(frozen=True, slots=True)
class _EvidenceValidationContext:
    """Keep optional evidence data used during source validation."""

    concept: Concept | None
    wiki_files: Mapping[str, bytes]


def validate_wiki_proposal(
    result: WikiCreationResult,
    *,
    original_files: Mapping[str, bytes],
    source: SourceRef,
    concept: Concept | None = None,
    page_manifest: WikiPageManifest | None = None,
) -> tuple[WikiValidationIssue, ...]:
    """Validate one active proposal against its plan and accepted Wiki."""
    plan = result.plan
    issues = list(_validate_change_scope(plan, result.changes))
    if issues or plan.target_path is None:
        return tuple(issues)
    change = _target_change(result)
    if change.content is None:
        return tuple(issues)
    target = plan.target_path
    try:
        frontmatter, body = parse_okf_document(change.content)
    except ValueError as error:
        return (
            WikiValidationIssue(
                WikiReviewFindingType.OKF,
                target,
                str(error),
                "Write valid OKF v0.2 YAML frontmatter and UTF-8 Markdown.",
                "Deterministic OKF parser",
            ),
        )
    issues.extend(_validate_required_frontmatter(target, frontmatter))
    issues.extend(_validate_ranking_frontmatter(target, frontmatter))
    issues.extend(
        _validate_sources(
            target,
            frontmatter,
            body,
            source,
            _EvidenceValidationContext(concept, original_files),
        ),
    )
    proposed_files = apply_wiki_changes(original_files, result.changes)
    issues.extend(
        _validate_links(
            target,
            plan,
            body,
            original_files,
            proposed_files,
            manifest_paths=(
                {page.bundle_path for page in page_manifest.pages}
                if page_manifest is not None
                else None
            ),
        ),
    )
    if plan.decision.value == "UPDATE":
        relative_target = target.removeprefix("wiki/")
        original_content = original_files.get(relative_target)
        if original_content is not None:
            issues.extend(
                _validate_unknown_fields(target, original_content, frontmatter),
            )
    return tuple(issues)


def validate_wiki_page_proposal(
    result: WikiCreationResult,
    *,
    original_files: Mapping[str, bytes],
    page_context: WikiPageContext,
    page_manifest: WikiPageManifest,
) -> tuple[WikiValidationIssue, ...]:
    """Validate one page proposal against every original member candidate."""
    unique: dict[WikiValidationIssue, None] = {}
    for candidate in page_context.candidates:
        for issue in validate_wiki_proposal(
            result,
            original_files=original_files,
            source=candidate.source,
            concept=candidate.concept,
            page_manifest=page_manifest,
        ):
            unique[issue] = None
    return tuple(unique)


def _validate_change_scope(
    plan: WikiPlan,
    changes: tuple[WikiFileChange, ...],
) -> tuple[WikiValidationIssue, ...]:
    """Require one non-deleted change at the fixed target."""
    if (
        len(changes) == 1
        and changes[0].path == plan.target_path
        and changes[0].status is not WikiChangeStatus.DELETED
    ):
        return ()
    return (
        WikiValidationIssue(
            WikiReviewFindingType.SCOPE,
            plan.target_path or "wiki",
            "The proposal did not change exactly the planned target file.",
            "Change only the planned target file and do not delete a file.",
            "Deterministic changed-path validation",
        ),
    )


def _validate_sources(
    target: str,
    frontmatter: Mapping[str, Any],
    body: str,
    source: SourceRef,
    evidence_context: _EvidenceValidationContext,
) -> list[WikiValidationIssue]:
    """Validate source entries and keyed claim footnotes."""
    issues: list[WikiValidationIssue] = []
    raw_sources = frontmatter.get("sources")
    if not isinstance(raw_sources, list):
        return [
            WikiValidationIssue(
                WikiReviewFindingType.SOURCE,
                f"{target} frontmatter.sources",
                "The `sources` value is not a list.",
                "Add a `sources` list that contains the current paper.",
                "Deterministic provenance validation",
            ),
        ]
    source_entries, invalid_entry = _source_entries(raw_sources)
    if invalid_entry:
        issues.append(
            WikiValidationIssue(
                WikiReviewFindingType.SOURCE,
                f"{target} frontmatter.sources",
                "A source entry has no unique non-empty ID and resource.",
                "Give each source a unique `id` and a non-empty `resource`.",
                "Deterministic OKF source-entry validation",
            ),
        )
    if source_entries.get(source.source_id) != source.resource:
        issues.append(
            WikiValidationIssue(
                WikiReviewFindingType.SOURCE,
                f"{target} frontmatter.sources",
                "The current paper source ID and resource are missing or do not match.",
                f"Add source `{source.source_id}` with resource `{source.resource}`.",
                "Current concept source metadata",
            ),
        )
    definitions = set(_FOOTNOTE_DEFINITION.findall(body))
    body_without_definitions = _FOOTNOTE_DEFINITION.sub("", body)
    uses = _FOOTNOTE_USE.findall(body_without_definitions)
    unmatched = sorted((set(uses) | definitions) - source_entries.keys())
    if unmatched:
        issues.append(
            WikiValidationIssue(
                WikiReviewFindingType.SOURCE,
                target,
                f"Footnote labels have no matching source IDs: {', '.join(unmatched)}.",
                "Use only footnote labels that match `sources[].id` values.",
                "Deterministic source-footnote join validation",
            ),
        )
    if source.source_id not in uses or source.source_id not in definitions:
        issues.append(
            WikiValidationIssue(
                WikiReviewFindingType.SOURCE,
                target,
                "The current paper has no matching claim footnote and definition.",
                f"Cite a supported claim with `[^{source.source_id}]` and add its definition.",
                "Current concept source metadata and deterministic footnote validation",
            ),
        )
    definition_lines = dict(_FOOTNOTE_DEFINITION_LINE.findall(body))
    source_definition = definition_lines.get(source.source_id, "")
    definition_targets = _LINK.findall(source_definition)
    if source.resource not in definition_targets:
        issues.append(
            WikiValidationIssue(
                WikiReviewFindingType.SOURCE,
                target,
                "The current paper footnote definition does not link to its resource.",
                (
                    f"Use the supplied linked definition `[^{source.source_id}]: "
                    f"[{source.source_id}]({source.resource})`."
                ),
                "Deterministic source-footnote link validation",
            ),
        )
    if evidence_context.concept is not None:
        issues.extend(
            _validate_evidence_links(
                target,
                body,
                source,
                evidence_context.concept,
                evidence_context.wiki_files,
            ),
        )
    return issues


def _validate_evidence_links(
    target: str,
    body: str,
    source: SourceRef,
    concept: Concept,
    wiki_files: Mapping[str, bytes],
) -> list[WikiValidationIssue]:
    """Require a supplied evidence target and reject bare evidence anchors."""
    expected_targets = {
        evidence_target(evidence, source.resource)
        for claim in concept.claims
        for evidence in claim.evidence
    }
    body_targets = set(_LINK.findall(body))
    issues: list[WikiValidationIssue] = []
    if expected_targets and not expected_targets.intersection(body_targets):
        issues.append(
            WikiValidationIssue(
                WikiReviewFindingType.SOURCE,
                target,
                "The current paper has no supplied evidence link.",
                "Copy a supplied `citation.evidence_link` next to its supported claim.",
                "Current concept evidence and deterministic Markdown link validation",
            ),
        )
    reference_path = source.resource.removeprefix("/")
    reference_content = wiki_files.get(reference_path)
    if reference_content is None:
        issues.append(
            WikiValidationIssue(
                WikiReviewFindingType.SOURCE,
                target,
                f"The source resource does not resolve: {source.resource}.",
                "Use the supplied source resource that exists in the Wiki bundle.",
                "Deterministic bundle resource validation",
            ),
        )
    else:
        reference_text = reference_content.decode("utf-8", errors="replace")
        linked_expected_targets = expected_targets.intersection(body_targets)
        missing_anchors = sorted(
            link_target.rsplit("#", maxsplit=1)[1]
            for link_target in linked_expected_targets
            if (
                f'<a id="{link_target.rsplit("#", maxsplit=1)[1]}"></a>'
                not in reference_text
            )
        )
        if missing_anchors:
            issues.append(
                WikiValidationIssue(
                    WikiReviewFindingType.SOURCE,
                    target,
                    (
                        "Evidence links use anchors that do not exist in the source "
                        f"resource: {', '.join(missing_anchors)}."
                    ),
                    "Copy only supplied evidence links that resolve in the source.",
                    "Deterministic source-anchor resolution",
                ),
            )

    body_without_link_targets = re.sub(r"\]\([^)]+\)", "]", body)
    bare_anchors = sorted(set(_EVIDENCE_ANCHOR.findall(body_without_link_targets)))
    if bare_anchors:
        issues.append(
            WikiValidationIssue(
                WikiReviewFindingType.SOURCE,
                target,
                f"Bare evidence anchors are not linked: {', '.join(bare_anchors)}.",
                "Use the supplied human-readable evidence links instead of bare IDs.",
                "Deterministic bare-anchor validation",
            ),
        )
    return issues


def _validate_links(  # ruff: ignore[too-many-arguments] - Link validation compares plan, baseline, proposal, and manifest.
    target: str,
    plan: WikiPlan,
    body: str,
    original_files: Mapping[str, bytes],
    proposed_files: Mapping[str, bytes],
    *,
    manifest_paths: set[str] | None = None,
) -> list[WikiValidationIssue]:
    """Require planned links and reject unplanned new concept links."""
    issues: list[WikiValidationIssue] = []
    links = _concept_links(body, target)
    planned = {link.target_path for link in plan.links}
    missing = sorted(planned - links)
    if missing:
        issues.append(
            WikiValidationIssue(
                WikiReviewFindingType.LINK,
                target,
                f"Planned links are missing: {', '.join(missing)}.",
                "Add every planned outbound link with its exact bundle-root path.",
                "Fixed Wiki plan and deterministic Markdown link scan",
            ),
        )
    if planned and _has_detached_concept_link_section(body):
        issues.append(
            WikiValidationIssue(
                WikiReviewFindingType.LINK,
                target,
                "Selected concept links are in a detached related-concepts section.",
                (
                    "Remove the detached section. Put each selected link in the "
                    "explanatory body text and state its approved relationship."
                ),
                "OKF relationship semantics and deterministic heading validation",
            ),
        )
    original_links: set[str] = set()
    original_content = original_files.get(target.removeprefix("wiki/"))
    if original_content is not None:
        try:
            _, original_body = parse_okf_document(original_content)
        except ValueError:
            original_body = original_content.decode("utf-8", errors="replace")
        original_links = _concept_links(original_body, target)
    unplanned = sorted((links - original_links) - planned)
    if unplanned:
        issues.append(
            WikiValidationIssue(
                WikiReviewFindingType.LINK,
                target,
                f"New internal concept links were not planned: {', '.join(unplanned)}.",
                "Remove each unplanned internal concept link.",
                "Fixed Wiki plan and original/proposed Wiki comparison",
            ),
        )
    resolvable = (
        manifest_paths
        if manifest_paths is not None
        else {"/" + path for path in proposed_files}
    )
    unresolved = sorted(link for link in links if link not in resolvable)
    if unresolved:
        issues.append(
            WikiValidationIssue(
                WikiReviewFindingType.LINK,
                target,
                f"Internal concept links do not resolve: {', '.join(unresolved)}.",
                "Use only planned links that resolve in the accepted Wiki.",
                "Deterministic proposed Wiki path validation",
            ),
        )
    return issues


def _validate_unknown_fields(
    target: str,
    original_content: bytes,
    proposed_frontmatter: Mapping[str, Any],
) -> list[WikiValidationIssue]:
    """Require unknown producer fields to survive an update."""
    try:
        original_frontmatter, _ = parse_okf_document(original_content)
    except ValueError:
        return []
    missing = [
        key
        for key in original_frontmatter
        if key not in {"generated", "verified"} and key not in proposed_frontmatter
    ]
    changed = [
        key
        for key, value in original_frontmatter.items()
        if key not in _KNOWN_OKF_FIELDS and proposed_frontmatter.get(key) != value
    ]
    affected = sorted(set(missing) | set(changed))
    if not affected:
        return []
    return [
        WikiValidationIssue(
            WikiReviewFindingType.OKF,
            f"{target} frontmatter",
            f"Existing or unknown frontmatter fields changed or disappeared: {', '.join(affected)}.",
            "Restore each missing field and each unknown field with its original value.",
            f"Original Wiki path {target}",
        ),
    ]
