"""Validate the complete Wiki corpus against its fixed manifest."""

# ruff: file-ignore[import-private-name]

from __future__ import annotations

from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any

from wiki_agents.agents.wiki.graph_validation import (
    _bundle_issue,
    validate_graph_manifest,
)
from wiki_agents.agents.wiki.models import WikiReviewFindingType
from wiki_agents.agents.wiki.validation_common import (
    _EVIDENCE_ANCHOR,
    _FOOTNOTE_DEFINITION_LINE,
    _FOOTNOTE_USE,
    _LINK,
    WIKI_ACTOR,
    WIKI_REVIEWER_ACTOR,
    WikiValidationIssue,
    _concept_links,
    _has_detached_concept_link_section,
    _source_entries,
    _validate_ranking_frontmatter,
    _validate_required_frontmatter,
    parse_okf_document,
    render_okf_document,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from wiki_agents.agents.wiki.models import WikiManifestPage, WikiPageManifest


def validate_wiki_bundle(
    wiki_files: Mapping[str, bytes],
    page_manifest: WikiPageManifest,
    *,
    require_indexes: bool = False,
) -> tuple[WikiValidationIssue, ...]:
    """Validate the complete proposed Wiki against its fixed page manifest.

    Paths in ``wiki_files`` can be relative to the Wiki root or repository root.
    The function does not mutate or normalize the supplied corpus.
    """
    issues: list[WikiValidationIssue] = list(validate_graph_manifest(page_manifest))
    files = {
        path.removeprefix("wiki/"): content for path, content in wiki_files.items()
    }
    planned_paths = [
        page.target_path.removeprefix("wiki/") for page in page_manifest.pages
    ]
    if len(planned_paths) != len(set(planned_paths)):
        issues.append(
            _bundle_issue(
                "wiki/concepts",
                "The page manifest contains duplicate target paths.",
                "Assign one unique target path to each page.",
                WikiReviewFindingType.SCOPE,
            ),
        )
    planned = set(planned_paths)
    actual = {
        path
        for path in files
        if path.startswith("concepts/")
        and path.endswith(".md")
        and PurePosixPath(path).name.lower() not in {"index.md", "log.md"}
    }
    issues.extend(
        [
            _bundle_issue(
                f"wiki/{path}",
                "A planned concept page is missing.",
                "Create the page at its fixed manifest path.",
                WikiReviewFindingType.SCOPE,
            )
            for path in sorted(planned - actual)
        ],
    )
    issues.extend(
        [
            _bundle_issue(
                f"wiki/{path}",
                "An unexpected concept page exists.",
                "Remove concept pages that are not in the manifest.",
                WikiReviewFindingType.SCOPE,
            )
            for path in sorted(actual - planned)
        ],
    )

    pages_by_path = {
        page.target_path.removeprefix("wiki/"): page for page in page_manifest.pages
    }
    for path in sorted(actual):
        content = files[path]
        target = f"wiki/{path}"
        try:
            frontmatter, body = parse_okf_document(content)
        except ValueError as error:
            issues.append(
                _bundle_issue(
                    target,
                    str(error),
                    "Write UTF-8 Markdown with valid YAML frontmatter.",
                    WikiReviewFindingType.OKF,
                ),
            )
            continue
        issues.extend(_validate_required_frontmatter(target, frontmatter))
        issues.extend(_validate_ranking_frontmatter(target, frontmatter))
        issues.extend(_validate_bundle_metadata(target, frontmatter))
        sources, invalid_sources = _source_entries_for_bundle(
            frontmatter.get("sources"),
        )
        if invalid_sources:
            issues.append(
                _bundle_issue(
                    f"{target} frontmatter.sources",
                    "A source entry has no unique valid ID and resource.",
                    "Give every source a unique non-empty ID and bundle-root resource.",
                    WikiReviewFindingType.SOURCE,
                ),
            )
        issues.extend(_validate_bundle_footnotes(target, body, sources))
        issues.extend(_validate_bundle_evidence(target, body, files))
        page = pages_by_path.get(path)
        if page is not None:
            issues.extend(_validate_manifest_links(target, body, page, actual))

    if require_indexes:
        index_content = files.get("concepts/index.md")
        if index_content is None or not index_content.startswith(b"# Concepts\n"):
            issues.append(
                _bundle_issue(
                    "wiki/concepts/index.md",
                    "The deterministic concept index is missing or malformed.",
                    "Generate the concept index after finalization.",
                    WikiReviewFindingType.OKF,
                ),
            )
    return tuple(issues)


def normalize_wiki_bundle_metadata(
    wiki_files: Mapping[str, bytes],
    page_manifest: WikiPageManifest,
    *,
    generated_at: str,
    verified_at: str,
) -> dict[str, bytes]:
    """Set final managed metadata on every planned page in memory."""
    normalized = dict(wiki_files)
    for page in page_manifest.pages:
        path = page.target_path.removeprefix("wiki/")
        try:
            content = normalized[path]
        except KeyError as error:
            raise ValueError(
                f"planned page is missing during normalization: {path}",
            ) from error
        frontmatter, body = parse_okf_document(content)
        frontmatter["generated"] = {"by": WIKI_ACTOR, "at": generated_at}
        frontmatter["verified"] = [
            {"by": WIKI_REVIEWER_ACTOR, "at": verified_at},
        ]
        normalized[path] = render_okf_document(frontmatter, body)
    return normalized


def _validate_bundle_metadata(
    target: str,
    frontmatter: Mapping[str, Any],
) -> list[WikiValidationIssue]:
    """Validate generated and verified producer metadata."""
    issues = []
    generated = frontmatter.get("generated")
    if not isinstance(generated, dict) or not all(
        isinstance(generated.get(key), str) and generated[key].strip()
        for key in ("by", "at")
    ):
        issues.append(
            _bundle_issue(
                f"{target} frontmatter.generated",
                "Generated metadata is missing or invalid.",
                "Add valid producer and timestamp metadata.",
                WikiReviewFindingType.OKF,
            ),
        )
    verified = frontmatter.get("verified")
    if (
        not isinstance(verified, list)
        or not verified
        or not all(
            isinstance(item, dict)
            and isinstance(item.get("by"), str)
            and isinstance(item.get("at"), str)
            for item in verified
        )
    ):
        issues.append(
            _bundle_issue(
                f"{target} frontmatter.verified",
                "Verification metadata is missing or invalid.",
                "Add at least one valid verifier record.",
                WikiReviewFindingType.OKF,
            ),
        )
    return issues


def _source_entries_for_bundle(raw_sources: object) -> tuple[dict[str, str], bool]:
    """Return strict bundle-root source entries and an invalid flag."""
    if not isinstance(raw_sources, list):
        return {}, True
    entries, invalid = _source_entries(list(raw_sources))
    if any(not resource.startswith("/") for resource in entries.values()):
        invalid = True
    return entries, invalid


def _validate_bundle_footnotes(
    target: str,
    body: str,
    sources: Mapping[str, str],
) -> list[WikiValidationIssue]:
    """Validate every source-footnote join and definition target."""
    issues = []
    definitions = dict(_FOOTNOTE_DEFINITION_LINE.findall(body))
    body_without_definitions = _FOOTNOTE_DEFINITION_LINE.sub("", body)
    used = set(_FOOTNOTE_USE.findall(body_without_definitions))
    invalid_labels = sorted((used | definitions.keys()) - sources.keys())
    if invalid_labels:
        issues.append(
            _bundle_issue(
                target,
                f"Footnotes have no matching source ID: {', '.join(invalid_labels)}.",
                "Use only footnote keys declared in sources.",
                WikiReviewFindingType.SOURCE,
            ),
        )
    missing_definitions = sorted(used - definitions.keys())
    if missing_definitions:
        issues.append(
            _bundle_issue(
                target,
                f"Used source footnotes have no definition: {', '.join(missing_definitions)}.",
                "Define every used source footnote.",
                WikiReviewFindingType.SOURCE,
            ),
        )
    issues.extend(
        [
            _bundle_issue(
                target,
                f"Footnote {source_id!r} does not link to its declared resource.",
                "Link the footnote definition to the declared source resource.",
                WikiReviewFindingType.SOURCE,
            )
            for source_id in sorted(used & definitions.keys() & sources.keys())
            if sources[source_id] not in _LINK.findall(definitions[source_id])
        ],
    )
    return issues


def _validate_bundle_evidence(
    target: str,
    body: str,
    files: Mapping[str, bytes],
) -> list[WikiValidationIssue]:
    """Resolve every linked evidence resource and anchor in the bundle."""
    issues = []
    for raw_link in sorted(set(_LINK.findall(body))):
        if "#" not in raw_link:
            continue
        resource, anchor = raw_link.rsplit("#", maxsplit=1)
        if _EVIDENCE_ANCHOR.fullmatch(anchor) is None:
            continue
        resource_path = resource.removeprefix("/")
        resource_content = files.get(resource_path)
        if resource_content is None:
            issues.append(
                _bundle_issue(
                    target,
                    f"Evidence resource does not resolve: {resource}.",
                    "Use an evidence resource that exists in the final Wiki.",
                    WikiReviewFindingType.SOURCE,
                ),
            )
        elif f'<a id="{anchor}"></a>'.encode() not in resource_content:
            issues.append(
                _bundle_issue(
                    target,
                    f"Evidence anchor does not resolve: {raw_link}.",
                    "Use an anchor that exists in the evidence resource.",
                    WikiReviewFindingType.SOURCE,
                ),
            )
    return issues


def _validate_manifest_links(
    target: str,
    body: str,
    page: WikiManifestPage,
    actual_concepts: set[str],
) -> list[WikiValidationIssue]:
    """Require selected links and reject all other internal concept links."""
    issues = []
    links = _concept_links(body, target)
    selected = {link.target_path for link in page.links if link.selected_for_page}
    accepted_unselected = {
        link.target_path for link in page.links if not link.selected_for_page
    }
    accepted = selected | accepted_unselected
    missing = sorted(selected - links)
    if missing:
        issues.append(
            _bundle_issue(
                target,
                f"Selected links are missing: {', '.join(missing)}.",
                "Add every selected link with its exact target path.",
                WikiReviewFindingType.LINK,
            ),
        )
    if selected and _has_detached_concept_link_section(body):
        issues.append(
            _bundle_issue(
                target,
                "Selected concept links are in a detached related-concepts section.",
                (
                    "Remove the detached section. Put each selected link in the "
                    "explanatory body text and state its approved relationship."
                ),
                WikiReviewFindingType.LINK,
            ),
        )
    promoted = sorted(links & accepted_unselected)
    if promoted:
        issues.append(
            _bundle_issue(
                target,
                f"Unselected accepted links are present: {', '.join(promoted)}.",
                "Remove each valid relation that was not selected for this page.",
                WikiReviewFindingType.LINK,
            ),
        )
    unrelated = sorted(links - accepted)
    if unrelated:
        issues.append(
            _bundle_issue(
                target,
                f"Internal concept links have no accepted relation: {', '.join(unrelated)}.",
                "Remove every unrelated internal concept link.",
                WikiReviewFindingType.LINK,
            ),
        )
    self_path = "/" + target.removeprefix("wiki/")
    if self_path in links:
        issues.append(
            _bundle_issue(
                target,
                "The concept page links to itself.",
                "Remove the self-link.",
                WikiReviewFindingType.LINK,
            ),
        )
    unresolved = sorted(
        link for link in links if link.removeprefix("/") not in actual_concepts
    )
    if unresolved:
        issues.append(
            _bundle_issue(
                target,
                f"Internal concept links do not resolve: {', '.join(unresolved)}.",
                "Use only internal links that resolve in the final Wiki.",
                WikiReviewFindingType.LINK,
            ),
        )
    return issues
