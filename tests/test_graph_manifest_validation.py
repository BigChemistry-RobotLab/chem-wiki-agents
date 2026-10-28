"""Test deterministic validation of the accepted Wiki page graph."""

from uuid import UUID

import pytest

from wiki_agents.agents.wiki.models import (
    WikiCandidatePairContribution,
    WikiManifestLink,
    WikiManifestPage,
    WikiPageManifest,
)
from wiki_agents.agents.wiki.validation import (
    validate_graph_manifest,
    validate_wiki_bundle,
)


def _contribution(
    source: int,
    target: int,
    *,
    similarity: float,
) -> WikiCandidatePairContribution:
    return WikiCandidatePairContribution(
        source_candidate_id=UUID(int=source),
        target_candidate_id=UUID(int=target),
        similarity=similarity,
    )


def _link(  # ruff: ignore[too-many-arguments] - The fixture exposes link ranking fields.
    target_page_id: str,
    target_path: str,
    *,
    target_candidate: int,
    global_pagerank: float,
    semantic_score: float,
    ranking_score: float,
    selected: bool,
) -> WikiManifestLink:
    return WikiManifestLink(
        target_page_id=target_page_id,
        target_path=target_path,
        relationships=[f"Relation to {target_page_id}."],
        semantic_score=semantic_score,
        global_pagerank=global_pagerank,
        personalized_pagerank=0.5,
        ranking_score=ranking_score,
        selected_for_page=selected,
        contributing_candidate_pairs=[
            _contribution(1, target_candidate, similarity=semantic_score),
        ],
    )


def _page(
    page_id: str,
    candidate: int,
    global_pagerank: float,
    *,
    links: list[WikiManifestLink] | None = None,
) -> WikiManifestPage:
    return WikiManifestPage(
        page_id=page_id,
        target_path=f"wiki/concepts/{page_id}.md",
        bundle_path=f"/concepts/{page_id}.md",
        title=page_id.upper(),
        scope=f"Scope for {page_id}.",
        work_summary=[f"Explain {page_id}."],
        member_candidate_ids=[UUID(int=candidate)],
        global_pagerank=global_pagerank,
        links=links or [],
    )


def _manifest() -> WikiPageManifest:
    selected = _link(
        "b",
        "/concepts/b.md",
        target_candidate=2,
        global_pagerank=0.6,
        semantic_score=0.8,
        ranking_score=0.75,
        selected=True,
    )
    unselected = _link(
        "c",
        "/concepts/c.md",
        target_candidate=3,
        global_pagerank=0.1,
        semantic_score=0.4,
        ranking_score=0.35,
        selected=False,
    )
    return WikiPageManifest(
        pages=[
            _page("b", 2, 0.6),
            _page("a", 1, 0.3, links=[selected, unselected]),
            _page("c", 3, 0.1),
        ],
    )


def _document(title: str, *, links: str = "", extra_frontmatter: str = "") -> bytes:
    return (
        "---\n"
        "type: Chemistry Concept\n"
        f"title: {title}\n"
        f"description: Supported description for {title}.\n"
        "sources: []\n"
        "generated:\n"
        "  by: wiki-agents/0.1.0\n"
        "  at: '2026-08-20T10:00:00Z'\n"
        "verified:\n"
        "  - by: wiki-agents-reviewer/0.1.0\n"
        "    at: '2026-08-20T10:01:00Z'\n"
        f"{extra_frontmatter}"
        "---\n\n"
        f"# {title}\n\n"
        f"Supported text.{links}\n"
    ).encode()


def _wiki_files(*, links_from_a: str = "", extra_a: str = "") -> dict[str, bytes]:
    return {
        "concepts/a.md": _document("A", links=links_from_a, extra_frontmatter=extra_a),
        "concepts/b.md": _document("B"),
        "concepts/c.md": _document("C"),
    }


def _replace_source_page(
    manifest: WikiPageManifest,
    source_page: WikiManifestPage,
) -> WikiPageManifest:
    return manifest.model_copy(
        update={
            "pages": [
                source_page if page.page_id == source_page.page_id else page
                for page in manifest.pages
            ],
        },
    )


def test_graph_manifest_and_selected_only_bundle_are_valid() -> None:
    manifest = _manifest()

    assert validate_graph_manifest(manifest) == ()
    assert (
        validate_wiki_bundle(
            _wiki_files(links_from_a=" See [B](/concepts/b.md)."),
            manifest,
        )
        == ()
    )


@pytest.mark.parametrize(
    ("field", "problem"),
    [
        ("page_id", "duplicate page IDs"),
        ("target_path", "duplicate target paths"),
        ("bundle_path", "duplicate bundle paths"),
    ],
)
def test_graph_manifest_rejects_duplicate_page_identities(
    field: str, problem: str,
) -> None:
    manifest = _manifest()
    first = manifest.pages[0]
    last = manifest.pages[-1].model_copy(update={field: getattr(first, field)})
    invalid = manifest.model_copy(update={"pages": [*manifest.pages[:-1], last]})

    issues = validate_graph_manifest(invalid)

    assert any(problem in issue.problem for issue in issues)


def test_graph_manifest_rejects_unknown_and_invalid_selected_target() -> None:
    manifest = _manifest()
    source = manifest.pages[1]
    unknown = source.links[0].model_copy(
        update={
            "target_page_id": "missing",
            "target_path": "/concepts/missing.md",
        },
    )
    invalid = _replace_source_page(
        manifest,
        source.model_copy(update={"links": [unknown, source.links[1]]}),
    )

    problems = [issue.problem for issue in validate_graph_manifest(invalid)]

    assert any("Graph nodes have no final page" in problem for problem in problems)
    assert any("Accepted target page does not exist" in problem for problem in problems)
    assert any(
        "Selected link is not a valid accepted relation" in problem
        for problem in problems
    )


def test_graph_manifest_rejects_self_and_duplicate_relations() -> None:
    manifest = _manifest()
    source = manifest.pages[1]
    self_link = source.links[0].model_copy(
        update={"target_page_id": "a", "target_path": "/concepts/a.md"},
    )
    duplicate = source.links[1].model_copy(
        update={
            "target_page_id": "b",
            "target_path": "/concepts/b.md",
            "global_pagerank": 0.6,
        },
    )
    invalid = _replace_source_page(
        manifest,
        source.model_copy(update={"links": [self_link, source.links[0], duplicate]}),
    )

    problems = [issue.problem for issue in validate_graph_manifest(invalid)]

    assert any("self-link" in problem for problem in problems)
    assert any("repeats an accepted target" in problem for problem in problems)


@pytest.mark.parametrize(
    ("update", "problem"),
    [
        ({"contributing_candidate_pairs": []}, "has no candidate contribution"),
        (
            {
                "contributing_candidate_pairs": [
                    _contribution(3, 2, similarity=0.8),
                ],
            },
            "source candidate does not belong",
        ),
        (
            {
                "contributing_candidate_pairs": [
                    _contribution(1, 3, similarity=0.8),
                ],
            },
            "target candidate does not belong",
        ),
        (
            {
                "contributing_candidate_pairs": [
                    _contribution(1, 9, similarity=0.8),
                ],
            },
            "absent from final pages",
        ),
    ],
)
def test_graph_manifest_rejects_bad_contribution_membership(
    update: dict[str, object],
    problem: str,
) -> None:
    manifest = _manifest()
    source = manifest.pages[1]
    link = source.links[0].model_copy(update=update)
    invalid = _replace_source_page(
        manifest,
        source.model_copy(update={"links": [link, source.links[1]]}),
    )

    assert any(problem in issue.problem for issue in validate_graph_manifest(invalid))


def test_graph_manifest_rejects_relation_internal_to_merged_page() -> None:
    manifest = _manifest()
    source = manifest.pages[1]
    target = manifest.pages[0]
    source_with_target_member = source.model_copy(
        update={"member_candidate_ids": [UUID(int=1), UUID(int=2)]},
    )
    invalid = manifest.model_copy(
        update={
            "pages": [
                target,
                source_with_target_member,
                manifest.pages[2],
            ],
        },
    )

    assert any(
        "internal to one merged page" in issue.problem
        for issue in validate_graph_manifest(invalid)
    )


def test_graph_manifest_rejects_unstable_relationship_contribution_and_link_order() -> (
    None
):
    manifest = _manifest()
    source = manifest.pages[1]
    contribution = source.links[0].contributing_candidate_pairs[0]
    unstable = source.links[0].model_copy(
        update={
            "relationships": ["Z relation.", "A relation."],
            "contributing_candidate_pairs": [contribution, contribution],
        },
    )
    invalid = _replace_source_page(
        manifest,
        source.model_copy(update={"links": [source.links[1], unstable]}),
    )

    problems = [issue.problem for issue in validate_graph_manifest(invalid)]

    assert any(
        "Relationships" in problem and "stable unique order" in problem
        for problem in problems
    )
    assert any(
        "Contributions" in problem and "stable unique order" in problem
        for problem in problems
    )
    assert any("stable selected and ranking order" in problem for problem in problems)


def test_graph_manifest_rejects_unstable_global_page_order() -> None:
    manifest = _manifest()
    invalid = manifest.model_copy(update={"pages": list(reversed(manifest.pages))})

    assert any(
        "stable PageRank and page-ID order" in issue.problem
        for issue in validate_graph_manifest(invalid)
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("semantic_score", float("nan")),
        ("global_pagerank", -0.1),
        ("personalized_pagerank", 1.1),
        ("ranking_score", float("inf")),
    ],
)
def test_graph_manifest_rejects_invalid_link_scores(field: str, value: float) -> None:
    manifest = _manifest()
    source = manifest.pages[1]
    invalid_link = source.links[0].model_copy(update={field: value})
    invalid = _replace_source_page(
        manifest,
        source.model_copy(update={"links": [invalid_link, source.links[1]]}),
    )

    assert any(
        f"`{field}`" in issue.problem for issue in validate_graph_manifest(invalid)
    )


def test_graph_manifest_rejects_invalid_page_and_contribution_scores() -> None:
    manifest = _manifest()
    source = manifest.pages[1]
    contribution = (
        source.links[0]
        .contributing_candidate_pairs[0]
        .model_copy(
            update={"similarity": 1.1},
        )
    )
    invalid_link = source.links[0].model_copy(
        update={"contributing_candidate_pairs": [contribution]},
    )
    invalid_source = source.model_copy(
        update={
            "global_pagerank": float("nan"),
            "links": [invalid_link, source.links[1]],
        },
    )

    problems = [
        issue.problem
        for issue in validate_graph_manifest(
            _replace_source_page(manifest, invalid_source),
        )
    ]

    assert any("`global_pagerank`" in problem for problem in problems)
    assert any("`contribution similarity`" in problem for problem in problems)


def test_graph_manifest_requires_link_global_pagerank_to_match_target() -> None:
    manifest = _manifest()
    source = manifest.pages[1]
    mismatched = source.links[0].model_copy(update={"global_pagerank": 0.2})
    invalid = _replace_source_page(
        manifest,
        source.model_copy(update={"links": [mismatched, source.links[1]]}),
    )

    assert any(
        "does not match target page" in issue.problem
        for issue in validate_graph_manifest(invalid)
    )


def test_graph_manifest_requires_exact_target_path() -> None:
    manifest = _manifest()
    source = manifest.pages[1]
    mismatched = source.links[0].model_copy(update={"target_path": "/concepts/c.md"})
    invalid = _replace_source_page(
        manifest,
        source.model_copy(update={"links": [mismatched, source.links[1]]}),
    )

    assert any(
        "target path does not match" in issue.problem
        for issue in validate_graph_manifest(invalid)
    )


def test_bundle_requires_selected_link() -> None:
    issues = validate_wiki_bundle(_wiki_files(), _manifest())

    assert any("Selected links are missing" in issue.problem for issue in issues)


def test_bundle_rejects_detached_related_concepts_section() -> None:
    issues = validate_wiki_bundle(
        _wiki_files(
            links_from_a=(
                "\n\n## Related concepts\n\n"
                "- [B](/concepts/b.md) — B supplies the comparison mechanism."
            ),
        ),
        _manifest(),
    )

    assert any("detached related-concepts section" in issue.problem for issue in issues)


def test_bundle_rejects_unselected_accepted_link() -> None:
    issues = validate_wiki_bundle(
        _wiki_files(
            links_from_a=(" See [B](/concepts/b.md) and [C](/concepts/c.md)."),
        ),
        _manifest(),
    )

    assert any(
        "Unselected accepted links are present" in issue.problem for issue in issues
    )


def test_bundle_rejects_unrelated_unresolved_and_self_links() -> None:
    issues = validate_wiki_bundle(
        _wiki_files(
            links_from_a=(
                " See [B](/concepts/b.md), [missing](/concepts/missing.md), "
                "and [this page](/concepts/a.md)."
            ),
        ),
        _manifest(),
    )
    problems = [issue.problem for issue in issues]

    assert any("have no accepted relation" in problem for problem in problems)
    assert any("do not resolve" in problem for problem in problems)
    assert any("links to itself" in problem for problem in problems)


@pytest.mark.parametrize(
    "field",
    [
        "pagerank",
        "global_pagerank",
        "personalized_pagerank",
        "ranking_score",
        "graph_centrality",
    ],
)
def test_bundle_rejects_ranking_frontmatter(field: str) -> None:
    issues = validate_wiki_bundle(
        _wiki_files(
            links_from_a=" See [B](/concepts/b.md).",
            extra_a=f"{field}: 0.5\n",
        ),
        _manifest(),
    )

    assert any(
        "Pipeline graph ranking fields are present" in issue.problem for issue in issues
    )
