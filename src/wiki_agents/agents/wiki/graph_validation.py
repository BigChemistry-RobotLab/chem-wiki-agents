"""Validate the Wiki graph manifest and ranking metadata."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from wiki_agents.agents.wiki.models import WikiReviewFindingType
from wiki_agents.agents.wiki.validation_common import WikiValidationIssue

if TYPE_CHECKING:
    from collections.abc import Mapping
    from uuid import UUID

    from wiki_agents.agents.wiki.models import (
        WikiManifestLink,
        WikiManifestPage,
        WikiPageManifest,
    )


def _bundle_issue(
    target: str,
    problem: str,
    required_change: str,
    finding_type: WikiReviewFindingType,
) -> WikiValidationIssue:
    """Create one issue backed by deterministic corpus validation."""
    return WikiValidationIssue(
        finding_type,
        target,
        problem,
        required_change,
        "Deterministic Wiki bundle validation",
    )


def validate_graph_manifest(
    page_manifest: WikiPageManifest,
) -> tuple[WikiValidationIssue, ...]:
    """Validate the complete accepted page graph and ranking metadata.

    PageRank values describe structural graph centrality. They do not describe
    scientific importance.

    Args:
        page_manifest: Complete immutable page and accepted-link manifest.

    Returns:
        Deterministic graph validation issues.

    """
    pages = page_manifest.pages
    issues: list[WikiValidationIssue] = []
    page_ids = [page.page_id for page in pages]
    target_paths = [page.target_path for page in pages]
    bundle_paths = [page.bundle_path for page in pages]
    issues.extend(_duplicate_manifest_value_issues(page_ids, "page IDs"))
    issues.extend(_duplicate_manifest_value_issues(target_paths, "target paths"))
    issues.extend(_duplicate_manifest_value_issues(bundle_paths, "bundle paths"))

    pages_by_id = {page.page_id: page for page in pages}
    final_page_ids = set(page_ids)
    graph_page_ids = final_page_ids | {
        link.target_page_id for page in pages for link in page.links
    }
    unknown_graph_nodes = sorted(graph_page_ids - final_page_ids)
    if unknown_graph_nodes:
        issues.append(
            _bundle_issue(
                "wiki/concepts",
                "Graph nodes have no final page: "
                + ", ".join(unknown_graph_nodes)
                + ".",
                "Remove each unknown graph node or add its final page definition.",
                WikiReviewFindingType.SCOPE,
            ),
        )

    candidate_pages: dict[UUID, set[str]] = {}
    for page in pages:
        for candidate_id in page.member_candidate_ids:
            candidate_pages.setdefault(candidate_id, set()).add(page.page_id)
    repeated_candidates = sorted(
        (
            str(candidate_id)
            for candidate_id, owners in candidate_pages.items()
            if len(owners) > 1
        ),
    )
    if repeated_candidates:
        issues.append(
            _bundle_issue(
                "wiki/concepts",
                "Candidates occur in more than one final page: "
                + ", ".join(repeated_candidates)
                + ".",
                "Assign each candidate to exactly one final page.",
                WikiReviewFindingType.SCOPE,
            ),
        )

    all_candidate_ids = set(candidate_pages)
    for page in pages:
        issues.extend(
            _validate_graph_manifest_page(
                page,
                pages_by_id,
                candidate_pages,
                all_candidate_ids,
            ),
        )

    if all(_is_unit_score(page.global_pagerank) for page in pages):
        expected_pages = sorted(
            pages,
            key=lambda page: (-float(page.global_pagerank), page.page_id),
        )
        if pages != expected_pages:
            issues.append(
                _bundle_issue(
                    "wiki/concepts",
                    "Manifest pages do not use stable PageRank and page-ID order.",
                    "Sort pages by descending global PageRank, then page ID.",
                    WikiReviewFindingType.SCOPE,
                ),
            )
    return tuple(issues)


def _duplicate_manifest_value_issues(
    values: list[str],
    label: str,
) -> list[WikiValidationIssue]:
    """Return one issue when a manifest identity collection has duplicates."""
    if len(values) == len(set(values)):
        return []
    return [
        _bundle_issue(
            "wiki/concepts",
            f"The page manifest contains duplicate {label}.",
            f"Use unique {label} for every final page.",
            WikiReviewFindingType.SCOPE,
        ),
    ]


def _validate_graph_manifest_page(
    page: WikiManifestPage,
    pages_by_id: Mapping[str, WikiManifestPage],
    candidate_pages: Mapping[UUID, set[str]],
    all_candidate_ids: set[UUID],
) -> list[WikiValidationIssue]:
    """Validate one source page and all accepted outgoing relations."""
    issues = _score_issues(
        page.target_path,
        {"global_pagerank": page.global_pagerank},
    )
    target_ids = [link.target_page_id for link in page.links]
    if len(target_ids) != len(set(target_ids)):
        issues.append(
            _bundle_issue(
                page.target_path,
                "The source page repeats an accepted target page.",
                "Keep at most one accepted relation for each target page.",
                WikiReviewFindingType.LINK,
            ),
        )
    for link in page.links:
        issues.extend(
            _validate_graph_manifest_link(
                page,
                link,
                pages_by_id.get(link.target_page_id),
                candidate_pages,
                all_candidate_ids,
            ),
        )
    if all(_is_unit_score(link.ranking_score) for link in page.links):
        expected_links = sorted(
            page.links,
            key=lambda link: (
                not link.selected_for_page,
                -float(link.ranking_score),
                link.target_page_id,
            ),
        )
        if page.links != expected_links:
            issues.append(
                _bundle_issue(
                    page.target_path,
                    "Accepted links do not use stable selected and ranking order.",
                    (
                        "Put selected links first and sort each group by descending "
                        "ranking score, then target page ID."
                    ),
                    WikiReviewFindingType.LINK,
                ),
            )
    return issues


def _validate_graph_manifest_link(
    source_page: WikiManifestPage,
    link: WikiManifestLink,
    target_page: WikiManifestPage | None,
    candidate_pages: Mapping[UUID, set[str]],
    all_candidate_ids: set[UUID],
) -> list[WikiValidationIssue]:
    """Validate one accepted relation and its candidate provenance."""
    issues: list[WikiValidationIssue] = []
    link_target_id = link.target_page_id
    if target_page is None:
        issues.append(
            _bundle_issue(
                source_page.target_path,
                f"Accepted target page does not exist: {link_target_id}.",
                "Keep accepted links only when their final target page exists.",
                WikiReviewFindingType.LINK,
            ),
        )
        if link.selected_for_page:
            issues.append(
                _bundle_issue(
                    source_page.target_path,
                    f"Selected link is not a valid accepted relation: {link_target_id}.",
                    "Select only an accepted outgoing relation for this source page.",
                    WikiReviewFindingType.LINK,
                ),
            )
    else:
        if link.target_path != target_page.bundle_path:
            issues.append(
                _bundle_issue(
                    source_page.target_path,
                    f"Accepted target path does not match page {link_target_id}.",
                    "Use the exact bundle path of the accepted target page.",
                    WikiReviewFindingType.LINK,
                ),
            )
        if (
            _is_unit_score(link.global_pagerank)
            and _is_unit_score(
                target_page.global_pagerank,
            )
            and not math.isclose(
                float(link.global_pagerank),
                float(target_page.global_pagerank),
                rel_tol=0.0,
                abs_tol=0.0,
            )
        ):
            issues.append(
                _bundle_issue(
                    source_page.target_path,
                    f"Stored global PageRank does not match target page {link_target_id}.",
                    "Copy the target page global PageRank into the accepted link.",
                    WikiReviewFindingType.LINK,
                ),
            )
    if (
        link_target_id == source_page.page_id
        or link.target_path == source_page.bundle_path
    ):
        issues.append(
            _bundle_issue(
                source_page.target_path,
                "The accepted page graph contains a self-link.",
                "Remove the accepted self-link.",
                WikiReviewFindingType.LINK,
            ),
        )

    issues.extend(
        _score_issues(
            source_page.target_path,
            {
                "semantic_score": link.semantic_score,
                "global_pagerank": link.global_pagerank,
                "personalized_pagerank": link.personalized_pagerank,
                "ranking_score": link.ranking_score,
            },
        ),
    )
    relationships = list(link.relationships)
    if relationships != sorted(set(relationships)):
        issues.append(
            _bundle_issue(
                source_page.target_path,
                f"Relationships for target {link_target_id} do not use stable unique order.",
                "Sort and deduplicate the accepted relationship text.",
                WikiReviewFindingType.LINK,
            ),
        )
    issues.extend(
        _validate_graph_contributions(
            source_page,
            target_page,
            link,
            candidate_pages,
            all_candidate_ids,
        ),
    )
    return issues


def _validate_graph_contributions(
    source_page: WikiManifestPage,
    target_page: WikiManifestPage | None,
    link: WikiManifestLink,
    candidate_pages: Mapping[UUID, set[str]],
    all_candidate_ids: set[UUID],
) -> list[WikiValidationIssue]:
    """Validate stable candidate provenance for one accepted page relation."""
    target_page_id = link.target_page_id
    contributions = link.contributing_candidate_pairs
    if not contributions:
        return [
            _bundle_issue(
                source_page.target_path,
                f"Accepted relation to {target_page_id} has no candidate contribution.",
                "Keep at least one directed candidate-pair contribution.",
                WikiReviewFindingType.LINK,
            ),
        ]
    issues: list[WikiValidationIssue] = []
    contribution_keys = [
        (str(item.source_candidate_id), str(item.target_candidate_id))
        for item in contributions
    ]
    if contribution_keys != sorted(contribution_keys) or len(contribution_keys) != len(
        set(contribution_keys),
    ):
        issues.append(
            _bundle_issue(
                source_page.target_path,
                f"Contributions for target {target_page_id} do not use stable unique order.",
                "Sort and deduplicate directed candidate-pair contributions.",
                WikiReviewFindingType.LINK,
            ),
        )
    source_members = set(source_page.member_candidate_ids)
    target_members = set(target_page.member_candidate_ids) if target_page else set()
    for contribution in contributions:
        source_candidate = contribution.source_candidate_id
        target_candidate = contribution.target_candidate_id
        missing_candidates = sorted(
            str(item)
            for item in (source_candidate, target_candidate)
            if item not in all_candidate_ids
        )
        if missing_candidates:
            issues.append(
                _bundle_issue(
                    source_page.target_path,
                    "Contribution references candidates absent from final pages: "
                    + ", ".join(missing_candidates)
                    + ".",
                    "Use only candidates assigned to final pages.",
                    WikiReviewFindingType.LINK,
                ),
            )
        if source_candidate not in source_members:
            issues.append(
                _bundle_issue(
                    source_page.target_path,
                    f"Contribution source candidate does not belong to page {source_page.page_id}.",
                    "Use a source candidate from the accepted source page.",
                    WikiReviewFindingType.LINK,
                ),
            )
        if target_candidate not in target_members:
            issues.append(
                _bundle_issue(
                    source_page.target_path,
                    f"Contribution target candidate does not belong to page {target_page_id}.",
                    "Use a target candidate from the accepted target page.",
                    WikiReviewFindingType.LINK,
                ),
            )
        if candidate_pages.get(source_candidate, set()) & candidate_pages.get(
            target_candidate,
            set(),
        ):
            issues.append(
                _bundle_issue(
                    source_page.target_path,
                    "A candidate contribution remains internal to one merged page.",
                    "Remove relations that became internal during MERGE collapse.",
                    WikiReviewFindingType.LINK,
                ),
            )
        issues.extend(
            _score_issues(
                source_page.target_path,
                {"contribution similarity": contribution.similarity},
            ),
        )
    return issues


def _is_unit_score(value: object) -> bool:
    """Return whether a value is a finite float in the closed unit interval."""
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0.0 <= value <= 1.0
    )


def _score_issues(
    target: str,
    scores: Mapping[str, object],
) -> list[WikiValidationIssue]:
    """Require normalized finite graph ranking and semantic scores."""
    return [
        _bundle_issue(
            target,
            f"Graph score `{name}` is not finite and within [0,1].",
            "Store a finite normalized pipeline score.",
            WikiReviewFindingType.LINK,
        )
        for name, value in scores.items()
        if not _is_unit_score(value)
    ]
