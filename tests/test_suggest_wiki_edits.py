"""Tests for the report-to-Wiki edit suggestion script."""

from __future__ import annotations

import asyncio
import importlib.util
import io
import json
import logging
import sys
import tarfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Self, cast

import pytest
from agents.sandbox.entries import Dir, File
from pydantic import ValidationError

_SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "suggest_wiki_edits.py"
_SPEC = importlib.util.spec_from_file_location("suggest_wiki_edits", _SCRIPT_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Could not load {_SCRIPT_PATH}")
suggestions_module = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = suggestions_module
_SPEC.loader.exec_module(suggestions_module)

DEFAULT_MAX_TURNS = suggestions_module.DEFAULT_MAX_TURNS
DEFAULT_MODEL = suggestions_module.DEFAULT_MODEL
WikiEditSuggestions = suggestions_module.WikiEditSuggestions
build_manifest = suggestions_module.build_manifest
build_parser = suggestions_module.build_parser
serialize_suggestions = suggestions_module.serialize_suggestions
validate_input_paths = suggestions_module.validate_input_paths
validate_suggestions = suggestions_module.validate_suggestions
write_suggestions = suggestions_module.write_suggestions

if TYPE_CHECKING:
    from collections.abc import Iterator

    from agents import RunConfig
    from agents.sandbox import Manifest
    from agents.sandbox.sandboxes.docker import DockerSandboxClientOptions


def _suggestions(
    *,
    path: str = "wiki/results.md",
    action: str = "UPDATE",
) -> WikiEditSuggestions:
    """Build one suggestion payload through public structured validation."""
    return WikiEditSuggestions.model_validate(
        {
            "reasoning": "The report adds a bounded finding.",
            "edited_files": [
                {
                    "path": path,
                    "action": action,
                    "edits": ["Add the conditional result and its tested range."],
                    "reasoning": "This page already describes the tested system.",
                },
            ],
        },
    )


@pytest.fixture
def host_inputs(tmp_path: Path) -> tuple[Path, Path]:
    """Create a report and a repository with a nested Markdown Wiki."""
    report = tmp_path / "campaign-report.txt"
    report.write_text("# Result\n", encoding="utf-8")
    repository = tmp_path / "repository"
    wiki = repository / "wiki"
    (wiki / "nested").mkdir(parents=True)
    (wiki / "results.md").write_text("# Results\n", encoding="utf-8")
    (wiki / "nested" / "details.MD").write_text("# Details\n", encoding="utf-8")
    (wiki / "ignored.txt").write_text("not Markdown\n", encoding="utf-8")
    return report, repository


def test_parser_defaults_and_overrides() -> None:
    """Expose the planned positional arguments and stable defaults."""
    parser = build_parser()
    defaults = parser.parse_args(
        ["report.md", "repository", "--output", "suggestions.json"],
    )
    assert defaults.report_path == Path("report.md")
    assert defaults.wiki_repository == Path("repository")
    assert defaults.output == Path("suggestions.json")
    assert defaults.model == DEFAULT_MODEL
    assert defaults.max_turns == DEFAULT_MAX_TURNS

    overridden = parser.parse_args(
        [
            "report.md",
            "repository",
            "--output",
            "result.json",
            "--model",
            "test/model",
            "--max-turns",
            "7",
        ],
    )
    assert overridden.output == Path("result.json")
    assert overridden.model == "test/model"
    assert overridden.max_turns == 7


def test_parser_rejects_non_positive_turn_limit() -> None:
    """Reject a turn limit that cannot execute an agent."""
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "report.md",
                "repository",
                "--output",
                "suggestions.json",
                "--max-turns",
                "0",
            ],
        )


def test_parser_requires_output_path() -> None:
    """Require a destination for the JSON suggestion file."""
    with pytest.raises(SystemExit):
        build_parser().parse_args(["report.md", "repository"])


def test_input_validation_requires_report_repository_and_wiki(tmp_path: Path) -> None:
    """Reject each missing or incorrectly typed host input."""
    report = tmp_path / "report.md"
    repository = tmp_path / "repository"
    with pytest.raises(ValueError, match="Report path is not a file"):
        validate_input_paths(report, repository)

    report.write_text("report", encoding="utf-8")
    with pytest.raises(ValueError, match="Wiki repository is not a directory"):
        validate_input_paths(report, repository)

    repository.mkdir()
    with pytest.raises(ValueError, match="Wiki content path is not a directory"):
        validate_input_paths(report, repository)


def test_manifest_has_report_and_complete_read_only_markdown_tree(host_inputs) -> None:
    """Stage all Markdown files and no unrelated repository files."""
    report, repository = host_inputs
    manifest, paths = build_manifest(report, repository)

    assert paths == {"wiki/results.md", "wiki/nested/details.MD"}
    assert set(manifest.entries) == {
        "inputs",
        "wiki",
        "inputs/report.md",
        "wiki/results.md",
        "wiki/nested/details.MD",
    }
    assert [user.name for user in manifest.users] == ["planner"]
    assert cast("File", manifest.entries["inputs/report.md"]).content == b"# Result\n"
    for entry in manifest.entries.values():
        permissions = entry.permissions
        assert not permissions.owner & 2
        assert not permissions.group & 2
        assert not permissions.other & 2
        if isinstance(entry, Dir):
            assert permissions.directory


def test_structured_output_accepts_update_create_and_empty() -> None:
    """Accept matching actions and a justified result with no edits."""
    update = _suggestions()
    create = _suggestions(path="wiki/new.md", action="CREATE")
    empty = WikiEditSuggestions(
        reasoning="The report repeats existing knowledge and supports no change.",
        edited_files=[],
    )

    assert validate_suggestions(update, {"wiki/results.md"}) is update
    assert validate_suggestions(create, {"wiki/results.md"}) is create
    assert validate_suggestions(empty, {"wiki/results.md"}) is empty


def test_structured_output_accepts_multiple_distinct_files() -> None:
    """Accept one plan that updates and creates coherent concept pages."""
    payload = _suggestions().model_dump(mode="json")
    payload["edited_files"].append(
        {
            "path": "wiki/new-concept.md",
            "action": "CREATE",
            "edits": ["Add the distinct report-supported concept."],
            "reasoning": "No existing page has a compatible subject and scope.",
        },
    )
    suggestions = WikiEditSuggestions.model_validate(payload)

    assert validate_suggestions(suggestions, {"wiki/results.md"}) is suggestions


def test_agent_instructions_extract_lessons_and_update_concept_pages() -> None:
    """Require durable lessons, semantic search, and update-first selection."""
    instructions = suggestions_module._AGENT_INSTRUCTIONS

    assert "search the Wiki" in instructions
    assert "Do not\npropose a page that summarizes the report" in instructions
    assert "main durable lessons" in instructions
    assert "campaign bookkeeping" in instructions
    assert "Prefer an UPDATE" in instructions
    assert "Use CREATE only for a durable concept" in instructions
    assert "independently useful beyond this report" in instructions
    assert "one coherent concept page" in instructions
    assert "substantially\nduplicate an existing page" in instructions
    assert "zero, one, or several files" in instructions
    assert "not a report outline" in instructions


@pytest.mark.parametrize(
    "path",
    [
        "/wiki/page.md",
        "wiki/../secret.md",
        "../wiki/page.md",
        "other/page.md",
        "wiki/page.txt",
        "wiki\\page.md",
        "wiki//page.md",
    ],
)
def test_structured_output_rejects_unsafe_or_non_markdown_paths(path: str) -> None:
    """Reject paths that do not identify normalized Wiki Markdown files."""
    with pytest.raises(ValidationError):
        _suggestions(path=path)


def test_structured_output_rejects_duplicate_paths() -> None:
    """Reject multiple actions for one Wiki path."""
    payload = _suggestions().model_dump(mode="json")
    payload["edited_files"].append(payload["edited_files"][0])
    with pytest.raises(ValidationError, match="duplicate Wiki path"):
        WikiEditSuggestions.model_validate(payload)


@pytest.mark.parametrize(
    ("suggestion", "existing", "message"),
    [
        (_suggestions(), set(), "UPDATE target does not exist"),
        (
            _suggestions(path="wiki/new.md", action="CREATE"),
            {"wiki/new.md"},
            "CREATE target already exists",
        ),
    ],
)
def test_action_must_match_host_inventory(suggestion, existing, message: str) -> None:
    """Use the host inventory as the authority for create and update actions."""
    with pytest.raises(ValueError, match=message):
        validate_suggestions(suggestion, existing)


def test_json_serialization_is_formatted_and_contains_only_public_shape() -> None:
    """Serialize stable JSON without internal metadata."""
    serialized = serialize_suggestions(_suggestions())
    assert serialized.startswith('{\n  "reasoning"')
    assert json.loads(serialized) == _suggestions().model_dump(mode="json")


def test_write_suggestions_creates_json_file(tmp_path: Path) -> None:
    """Write the public suggestion payload to the selected destination."""
    output_path = tmp_path / "suggestions.json"

    write_suggestions(_suggestions(), output_path)

    assert output_path.read_text(encoding="utf-8").endswith("\n")
    assert json.loads(output_path.read_text(encoding="utf-8")) == (
        _suggestions().model_dump(mode="json")
    )


class _FakeExecResult:
    """Return a successful null-delimited file inventory."""

    def __init__(self, paths: list[str]) -> None:
        """Store the encoded inventory."""
        self.stdout = ("\0".join(paths) + "\0").encode()
        self.stderr = b""

    @staticmethod
    def ok() -> bool:
        """Report a successful command."""
        return True


class _FakeSession:
    """Provide the sandbox operations used by the suggestion script."""

    def __init__(self, manifest: Manifest) -> None:
        """Copy manifest files into an in-memory sandbox."""
        self.files = {
            str(path): entry.content
            for path, entry in manifest.entries.items()
            if isinstance(entry, File)
        }
        self.archives: dict[str, bytes] = {}
        self.archive_writes = 0
        self.state = SimpleNamespace(manifest=manifest)

    async def __aenter__(self) -> Self:
        """Enter the fake running session."""
        return self

    async def __aexit__(self, *_args: object) -> None:
        """Exit the fake running session."""

    async def exec(self, *args: object, **_kwargs: object) -> _FakeExecResult:
        """Extract one in-memory tar archive."""
        assert args[:2] == ("tar", "-xf")
        archive_path = cast("Path", args[2]).as_posix()
        with tarfile.open(fileobj=io.BytesIO(self.archives[archive_path]), mode="r:") as archive:
            for member in archive.getmembers():
                if member.isfile():
                    extracted = archive.extractfile(member)
                    assert extracted is not None
                    self.files[member.name] = extracted.read()
        return _FakeExecResult([])

    async def write(self, path: Path, data: io.BytesIO) -> None:
        """Store one archive uploaded to the fake session."""
        self.archives[path.as_posix()] = data.read()
        self.archive_writes += 1

    async def rm(self, path: Path) -> None:
        """Remove one uploaded archive from the fake session."""
        del self.archives[path.as_posix()]


class _FakeClient:
    """Track sandbox creation and cleanup."""

    def __init__(self) -> None:
        """Create empty sandbox client state."""
        self.options: DockerSandboxClientOptions | None = None
        self.manifest: Manifest | None = None
        self.session: _FakeSession | None = None
        self.deleted = False

    async def create(self, *, manifest, options) -> _FakeSession:
        """Create an in-memory session and record Docker options."""
        self.options = options
        self.manifest = manifest
        self.session = _FakeSession(manifest)
        return self.session

    async def delete(self, session) -> None:
        """Record unconditional session cleanup."""
        self.deleted = True


class _FakeRunResult:
    """Return a valid structured agent result."""

    def final_output_as(self, output_type, *, raise_if_incorrect_type):
        """Return a result with the requested output model."""
        assert output_type is WikiEditSuggestions
        assert raise_if_incorrect_type
        return _suggestions()


async def _apply_fake_manifest(
    session: _FakeSession,
    *,
    manifest: Manifest,
    provision_accounts: bool,
) -> None:
    """Materialize base manifest files in a fake session."""
    assert not provision_accounts
    for path, entry in manifest.entries.items():
        if isinstance(entry, File):
            session.files[str(path)] = entry.content


def test_workspace_materialization_batches_wiki_archives(monkeypatch) -> None:
    """Upload large Wiki inventories in the same archive batches as campaigns."""
    entries = {
        f"wiki/page-{index:03d}.md": File(content=f"page {index}\n".encode())
        for index in range(251)
    }
    entries["inputs/report.md"] = File(content=b"report\n")
    manifest = suggestions_module.Manifest(entries=entries)
    session = _FakeSession(suggestions_module._session_manifest())
    monkeypatch.setattr(
        suggestions_module.manifest_ops,
        "apply_manifest",
        _apply_fake_manifest,
    )

    asyncio.run(suggestions_module._materialize_workspace(session, manifest))

    assert session.archive_writes == 2
    assert len(session.files) == 252
    assert session.files["wiki/page-250.md"] == b"page 250\n"


def test_runner_forwards_model_turns_disables_network_and_cleans_up(
    host_inputs,
    monkeypatch,
    caplog,
) -> None:
    """Forward run settings and delete the isolated session after success."""
    report, repository = host_inputs
    client = _FakeClient()
    captured: dict[str, object] = {}

    @contextmanager
    def managed_client() -> Iterator[_FakeClient]:
        yield client

    def create_agent(*, model):
        captured["model"] = model
        return object()

    async def run(agent, run_input, *, max_turns, run_config):
        captured.update(
            agent=agent,
            run_input=run_input,
            max_turns=max_turns,
            run_config=run_config,
        )
        return _FakeRunResult()

    monkeypatch.setattr(suggestions_module, "managed_docker_sandbox_client", managed_client)
    monkeypatch.setattr(suggestions_module, "create_agent", create_agent)
    monkeypatch.setattr(suggestions_module.Runner, "run", run)
    monkeypatch.setattr(
        suggestions_module.manifest_ops,
        "apply_manifest",
        _apply_fake_manifest,
    )
    caplog.set_level(logging.INFO, logger="wiki_agents.suggest_wiki_edits")

    result = asyncio.run(
        suggestions_module.run_suggestion_agent(
            report,
            repository,
            model="test/model",
            max_turns=9,
        ),
    )

    assert result == _suggestions()
    assert captured["model"] == "test/model"
    assert captured["max_turns"] == 9
    run_config = cast("RunConfig", captured["run_config"])
    assert run_config.tracing_disabled is True
    assert client.options is not None
    assert client.options.network_mode == "none"
    assert client.manifest is not None
    assert client.manifest.entries == {}
    assert client.session is not None
    assert client.session.archive_writes == 1
    assert client.deleted
    assert "Starting Wiki edit suggestion agent" in caplog.text
    assert "Validated suggestion result: edited_files=1" in caplog.text
