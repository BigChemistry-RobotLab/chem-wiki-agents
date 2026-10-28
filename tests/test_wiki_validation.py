from pathlib import Path

import pytest
from pydantic import ValidationError

from wiki_agents.agents.wiki.indexes import render_concept_index
from wiki_agents.agents.wiki.models import (
    WikiChangeStatus,
    WikiCreationResult,
    WikiFileChange,
    WikiLinkPlan,
    WikiManifestPage,
    WikiPageManifest,
    WikiPlan,
    WikiPlanDecision,
)
from wiki_agents.agents.wiki.validation import (
    WIKI_ACTOR,
    WIKI_REVIEWER_ACTOR,
    add_machine_verification,
    normalize_managed_metadata,
    parse_okf_document,
    validate_wiki_proposal,
)
from wiki_agents.concept.models import Claim, Concept
from wiki_agents.document.models import EvidenceSpan, SourceRef


def _source() -> SourceRef:
    return SourceRef(
        source_id="paper-test",
        sha256="a" * 64,
        resource="/references/papers/paper-test.md",
    )


def _concept() -> Concept:
    return Concept(
        title="Catalysis",
        description="A supported chemistry concept.",
        claims=[
            Claim(
                statement="Catalysis changes a reaction.",
                evidence=[
                    EvidenceSpan(
                        start_line_id="p0001-l0001",
                        end_line_id="p0001-l0002",
                    ),
                ],
            ),
        ],
    )


def _document(*, extra: str = "", links: str = "") -> bytes:
    return (
        "---\n"
        "type: Chemistry Concept\n"
        "title: Catalysis\n"
        "description: A supported chemistry concept.\n"
        "sources:\n"
        "  - id: paper-test\n"
        "    resource: /references/papers/paper-test.md\n"
        f"{extra}"
        "---\n\n"
        f"Catalysis changes a reaction.[^paper-test]{links}\n\n"
        "[^paper-test]: [Test paper](/references/papers/paper-test.md)\n"
    ).encode()


def _plan(*, links: list[WikiLinkPlan] | None = None) -> WikiPlan:
    return WikiPlan(
        decision=WikiPlanDecision.CREATE,
        rationale="The source contains reusable knowledge.",
        target_path="wiki/concepts/catalysis.md",
        title="Catalysis",
        work_summary=["Add the supported mechanism."],
        links=links or [],
    )


def _result(plan: WikiPlan, content: bytes) -> WikiCreationResult:
    return WikiCreationResult(
        plan=plan,
        changes=(
            WikiFileChange(
                path="wiki/concepts/catalysis.md",
                status=WikiChangeStatus.ADDED,
                content=content,
            ),
        ),
        raw_output="created",
    )


@pytest.mark.parametrize(
    "target",
    [
        "wiki/index.md",
        "wiki/concepts/index.md",
        "wiki/concepts/log.md",
        "wiki/references/page.md",
        "wiki/concepts/page.txt",
    ],
)
def test_plan_rejects_unsafe_or_reserved_targets(target: str) -> None:
    with pytest.raises(ValidationError):
        WikiPlan(
            decision=WikiPlanDecision.CREATE,
            rationale="Create a concept.",
            target_path=target,
            title="Concept",
            work_summary=["Add content."],
        )


def test_plan_rejects_duplicate_link_targets() -> None:
    link = WikiLinkPlan(
        target_path="/concepts/existing.md",
        relationship="Related process.",
    )
    with pytest.raises(ValidationError, match="same link target"):
        _plan(links=[link, link])


@pytest.mark.parametrize(
    "target",
    [
        "/concepts/existing.md",
        "concepts/existing.md",
        "wiki/concepts/existing.md",
    ],
)
def test_plan_normalizes_safe_link_target_path_forms(target: str) -> None:
    link = WikiLinkPlan(
        target_path=target,
        relationship="Related process.",
    )

    assert link.target_path == "/concepts/existing.md"


@pytest.mark.parametrize(
    "target",
    [
        "/concepts/catalysis.md",
        "concepts/catalysis.md",
        "wiki/concepts/catalysis.md",
    ],
)
def test_plan_normalizes_safe_target_path_forms(target: str) -> None:
    plan = WikiPlan(
        decision=WikiPlanDecision.CREATE,
        rationale="Create a supported concept.",
        target_path=target,
        title="Catalysis",
        work_summary=["Add supported knowledge."],
    )

    assert plan.target_path == "wiki/concepts/catalysis.md"


def test_validation_accepts_okf_provenance_and_managed_trust() -> None:
    proposal = normalize_managed_metadata(
        _result(_plan(), _document()),
        generated_at="2026-08-19T10:00:00Z",
    )

    assert (
        validate_wiki_proposal(
            proposal,
            original_files={},
            source=_source(),
        )
        == ()
    )
    accepted = add_machine_verification(
        proposal,
        verified_at="2026-08-19T10:01:00Z",
    )
    frontmatter, _ = parse_okf_document(accepted.changes[0].content or b"")
    assert frontmatter["generated"]["by"] == WIKI_ACTOR
    assert frontmatter["verified"][0]["by"] == WIKI_REVIEWER_ACTOR


def test_validation_rejects_unlinked_source_footnote_definition() -> None:
    content = _document().replace(
        b"[Test paper](/references/papers/paper-test.md)",
        b"p0001-l0001",
    )

    issues = validate_wiki_proposal(
        _result(_plan(), content),
        original_files={},
        source=_source(),
    )

    assert any("does not link to its resource" in issue.problem for issue in issues)


def test_validation_requires_resolvable_evidence_link() -> None:
    evidence_link = (
        " ([page 1, lines 1-2](/references/papers/paper-test.md#p0001-l0001))."
    )
    source_document = b'<a id="p0001-l0001"></a> Source text\n'

    accepted = validate_wiki_proposal(
        _result(_plan(), _document(links=evidence_link)),
        original_files={
            "references/papers/paper-test.md": source_document,
        },
        source=_source(),
        concept=_concept(),
    )
    missing = validate_wiki_proposal(
        _result(_plan(), _document()),
        original_files={
            "references/papers/paper-test.md": source_document,
        },
        source=_source(),
        concept=_concept(),
    )

    assert accepted == ()
    assert any("no supplied evidence link" in issue.problem for issue in missing)


def test_validation_requires_planned_links_and_rejects_unplanned_links() -> None:
    planned = WikiLinkPlan(
        target_path="/concepts/neighbor.md",
        relationship="Related mechanism.",
    )
    plan = _plan(links=[planned])
    original = {"concepts/neighbor.md": _document()}

    missing = validate_wiki_proposal(
        _result(plan, _document()),
        original_files=original,
        source=_source(),
    )
    assert any("Planned links are missing" in issue.problem for issue in missing)

    unplanned_content = _document(links=" See [other](/concepts/other.md).")
    unplanned = validate_wiki_proposal(
        _result(_plan(), unplanned_content),
        original_files={"concepts/other.md": _document()},
        source=_source(),
    )
    assert any("were not planned" in issue.problem for issue in unplanned)


def test_page_validation_uses_manifest_paths_before_pages_exist() -> None:
    planned = WikiLinkPlan(
        target_path="/concepts/neighbor.md",
        relationship="The neighbor supplies the comparison mechanism.",
    )
    manifest_page = WikiManifestPage.model_construct(
        bundle_path="/concepts/neighbor.md",
    )
    manifest = WikiPageManifest.model_construct(pages=[manifest_page])
    content = _document(
        links=" The [neighbor](/concepts/neighbor.md) supplies the comparison mechanism.",
    )

    issues = validate_wiki_proposal(
        _result(_plan(links=[planned]), content),
        original_files={},
        source=_source(),
        page_manifest=manifest,
    )

    assert not any("do not resolve" in issue.problem for issue in issues)


def test_validation_rejects_detached_related_concepts_section() -> None:
    planned = WikiLinkPlan(
        target_path="/concepts/neighbor.md",
        relationship="The neighbor supplies the comparison mechanism.",
    )
    content = _document(
        links=(
            "\n\n## Related concepts\n\n"
            "- [Neighbor](/concepts/neighbor.md) — Comparison mechanism."
        ),
    )

    issues = validate_wiki_proposal(
        _result(_plan(links=[planned]), content),
        original_files={"concepts/neighbor.md": _document()},
        source=_source(),
    )

    assert any("detached related-concepts section" in issue.problem for issue in issues)


def test_update_validation_preserves_unknown_frontmatter_fields() -> None:
    original = _document(extra="producer_extension: keep-me\n")
    proposed = _document()
    plan = WikiPlan(
        decision=WikiPlanDecision.UPDATE,
        rationale="Add supported knowledge.",
        target_path="wiki/concepts/catalysis.md",
        title="Catalysis",
        work_summary=["Add one claim."],
    )
    result = WikiCreationResult(
        plan=plan,
        changes=(
            WikiFileChange(
                path="wiki/concepts/catalysis.md",
                status=WikiChangeStatus.MODIFIED,
                content=proposed,
            ),
        ),
        raw_output="updated",
    )

    issues = validate_wiki_proposal(
        result,
        original_files={"concepts/catalysis.md": original},
        source=_source(),
    )

    assert any("producer_extension" in issue.problem for issue in issues)


def test_concept_index_is_complete_and_deterministic(tmp_path: Path) -> None:
    directory = tmp_path / "concepts"
    directory.mkdir()
    (directory / "zeta.md").write_bytes(
        _document().replace(b"title: Catalysis", b"title: Zeta"),
    )
    (directory / "alpha.md").write_bytes(
        _document().replace(b"title: Catalysis", b"title: Alpha"),
    )
    (directory / "index.md").write_text("old index")

    rendered = render_concept_index(directory).decode()

    assert rendered.index("[Alpha](alpha.md)") < rendered.index("[Zeta](zeta.md)")
    assert "A supported chemistry concept." in rendered
