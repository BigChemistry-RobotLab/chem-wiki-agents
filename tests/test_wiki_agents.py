import pytest
from pydantic import ValidationError

from wiki_agents.agents.wiki.creator import (
    _validate_editor_changes,
    create_wiki_agent,
)
from wiki_agents.agents.wiki.models import (
    WikiChangeStatus,
    WikiFileChange,
    WikiPlan,
    WikiPlanDecision,
    WikiReview,
)
from wiki_agents.agents.wiki.resilience import WikiEditorOutputError
from wiki_agents.agents.wiki.reviewer import create_wiki_reviewer_agent
from wiki_agents.agents.wiki.workspace import (
    apply_wiki_changes,
    format_wiki_diff,
    wiki_changes,
)


def test_wiki_editor_and_reviewer_have_separate_roles() -> None:
    editor = create_wiki_agent(model="editor-model")
    reviewer = create_wiki_reviewer_agent(model="reviewer-model")

    assert editor.model == "editor-model"
    assert editor.output_type is str
    assert "approved plan" in str(editor.instructions)
    assert reviewer.model == "reviewer-model"
    assert reviewer.output_type is WikiReview
    assert "Do not edit any workspace file" in str(reviewer.instructions)
    assert "page_manifest.json` as the only authority" in str(reviewer.instructions)
    assert "Do not inspect the\n  filesystem" in str(reviewer.instructions)
    assert "creation or editing of another page" in str(reviewer.instructions)
    assert "similar_claims" not in str(reviewer.instructions)
    assert len(editor.capabilities) == 1
    assert len(reviewer.capabilities) == 1


def test_review_diff_contains_original_and_proposed_paths() -> None:
    diff = format_wiki_diff(
        {"concepts/current.md": b"old\n"},
        {"concepts/current.md": b"new\n"},
    )

    assert "--- original_wiki/concepts/current.md" in diff
    assert "+++ wiki/concepts/current.md" in diff


def test_plan_requires_fields_that_match_decision() -> None:
    skip = WikiPlan(
        decision=WikiPlanDecision.SKIP,
        rationale="No new knowledge.",
    )
    create = WikiPlan(
        decision=WikiPlanDecision.CREATE,
        rationale="The concept is relevant.",
        target_path="wiki/concepts/new-page.md",
        title="New page",
        work_summary=["Add the supported claim."],
    )

    assert skip.target_path is None
    assert create.target_path == "wiki/concepts/new-page.md"
    with pytest.raises(ValidationError):
        WikiPlan(
            decision=WikiPlanDecision.UPDATE,
            rationale="Update a page.",
            target_path="../outside.md",
            title="Outside",
            work_summary=["Change a file."],
        )


def test_editor_must_change_planned_target() -> None:
    plan = WikiPlan(
        decision=WikiPlanDecision.CREATE,
        rationale="Create it.",
        target_path="wiki/concepts/new.md",
        title="New",
        work_summary=["Add content."],
    )
    correct_change = WikiFileChange(
        path="wiki/concepts/new.md",
        status=WikiChangeStatus.ADDED,
        content=b"# New\n",
    )

    _validate_editor_changes(plan, (correct_change,))
    with pytest.raises(
        WikiEditorOutputError,
        match="did not change",
    ) as exc_info:
        _validate_editor_changes(plan, ())
    assert exc_info.value.code == "unchanged_target"


def test_wiki_changes_round_trip() -> None:
    original = {
        "concepts/keep.md": b"old",
        "concepts/delete.md": b"delete",
    }
    resulting = {
        "concepts/keep.md": b"new",
        "concepts/add.md": b"add",
    }

    changes = wiki_changes(original, resulting)

    assert [(change.path, change.status) for change in changes] == [
        ("wiki/concepts/add.md", WikiChangeStatus.ADDED),
        ("wiki/concepts/delete.md", WikiChangeStatus.DELETED),
        ("wiki/concepts/keep.md", WikiChangeStatus.MODIFIED),
    ]
    assert apply_wiki_changes(original, changes) == resulting
