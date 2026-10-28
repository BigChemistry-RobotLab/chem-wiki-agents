"""Manage the local Git repository that contains an OKF wiki bundle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Self

from git import Actor, Repo

if TYPE_CHECKING:
    from types import TracebackType

INITIAL_COMMIT_MESSAGE = "chore: initialize wiki repository"


@dataclass(frozen=True, slots=True)
class BotIdentity:
    """Identity used for commits created by the ingestion bot."""

    name: str = "Wiki Agent"
    email: str = "wiki-agent@localhost"


@dataclass(slots=True)
class WikiRepositoryBatch:
    """Track recoverable file changes for one PDF Wiki batch."""

    repository: WikiRepository
    backups: dict[str, bytes | None]
    paths: set[str]
    committed: bool = False

    def apply(self, files: dict[str, bytes]) -> None:
        """Apply and stage exact repository-relative file contents."""
        for relative, content in files.items():
            path = PurePosixPath(relative)
            if path.is_absolute() or ".." in path.parts or not path.parts:
                raise ValueError(f"Repository path is not safe: {relative}")
            target = self.repository.path / relative
            if relative not in self.backups:
                self.backups[relative] = (
                    target.read_bytes() if target.is_file() else None
                )
            if target.is_file() and target.read_bytes() == content:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            self.paths.add(relative)
        self.repository.stage_paths(sorted(self.paths))

    def commit(self, message: str) -> str:
        """Commit the exact staged batch and return its SHA."""
        commit_sha = self.repository.commit_staged(sorted(self.paths), message)
        self.committed = True
        return commit_sha

    def rollback(self) -> None:
        """Restore and unstage all active batch paths."""
        if self.committed:
            raise RuntimeError("A committed Wiki batch cannot be rolled back")
        self.repository.unstage(sorted(self.paths))
        for relative, content in self.backups.items():
            target = self.repository.path / relative
            if content is None:
                if target.exists():
                    target.unlink()
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
        self.paths.clear()

    def __enter__(self) -> Self:
        """Return the active batch."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Roll back an uncommitted batch when its scope exits."""
        if not self.committed and (exc_type is not None or self.paths):
            self.rollback()


class WikiRepository:
    """Own repository initialization and narrowly scoped Git operations."""

    def __init__(
        self,
        path: str | Path,
        *,
        identity: BotIdentity | None = None,
    ) -> None:
        self.path = Path(path).resolve()
        self.identity = identity or BotIdentity()
        self.path.mkdir(parents=True, exist_ok=True)
        if (self.path / ".git").is_dir():
            self.repo = Repo(self.path)
            return
        self.repo = Repo.init(self.path, initial_branch="main")
        self._initialize()

    def _initialize(self) -> None:
        """Create and commit the deterministic repository skeleton."""
        files = {
            ".gitignore": (
                "/.checkpoints/\n"
                "/.qdrant/\n"
                "/.staging/\n"
                "/sources/*\n"
                "!/sources/.gitkeep\n"
                "/wiki/references/papers/artifacts/*.pdf\n"
            ),
            "README.md": "# Wiki repository\n\nManaged by wiki-agents.\n",
            "sources/.gitkeep": "",
            "extractions/.gitkeep": "",
            "wiki/index.md": (
                "---\n"
                'okf_version: "0.2"\n'
                "---\n\n"
                "# Wiki\n\n"
                "- [Concepts](concepts/)\n"
                "- [References](references/)\n"
            ),
            "wiki/concepts/index.md": "# Concepts\n",
            "wiki/references/index.md": ("# References\n\n- [Papers](papers/)\n"),
            "wiki/references/papers/index.md": "# Papers\n",
            "wiki/references/papers/artifacts/.gitkeep": "",
        }
        for relative, content in files.items():
            target = self.path / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_text(content)
        paths = list(files)
        self.repo.git.add("--", *paths)
        self._commit(INITIAL_COMMIT_MESSAGE)

    def assert_clean(self) -> None:
        """Reject changes outside the orchestrator's ignored input/state areas."""
        if self.repo.is_dirty(index=True, working_tree=True, untracked_files=False):
            raise RuntimeError("Wiki repository has tracked or staged changes")
        allowed_prefixes = (
            ".checkpoints/",
            ".staging/",
        )
        artifact_prefix = "wiki/references/papers/artifacts/"
        unexpected = [
            path
            for path in self.repo.untracked_files
            if not (
                path.startswith(allowed_prefixes)
                or (path.startswith(artifact_prefix) and path.lower().endswith(".pdf"))
            )
        ]
        if unexpected:
            joined = ", ".join(sorted(unexpected))
            raise RuntimeError(
                f"Wiki repository has unexpected untracked files: {joined}",
            )

    def stage_and_commit(self, paths: list[str], message: str) -> str:
        """Stage exactly ``paths`` and create one commit."""
        self.stage_paths(paths)
        return self.commit_staged(paths, message)

    def stage_paths(self, paths: list[str]) -> None:
        """Stage exact paths and reject any path outside the supplied union."""
        if not paths:
            raise ValueError("At least one path is required for staging")
        self.repo.git.add("--", *paths)
        staged = {
            line.strip()
            for line in self.repo.git.diff("--cached", "--name-only").splitlines()
            if line.strip()
        }
        requested = set(paths)
        if not staged or not staged.issubset(requested):
            self.unstage(paths)
            raise RuntimeError(
                f"Unexpected staged paths: requested {sorted(paths)}, got {sorted(staged)}",
            )

    def commit_staged(self, paths: list[str], message: str) -> str:
        """Commit only when the index contains the exact requested paths."""
        staged = {
            line.strip()
            for line in self.repo.git.diff("--cached", "--name-only").splitlines()
            if line.strip()
        }
        if staged != set(paths):
            raise RuntimeError(
                f"Unexpected staged paths: requested {sorted(paths)}, got {sorted(staged)}",
            )
        return self._commit(message)

    def begin_batch(self) -> WikiRepositoryBatch:
        """Start one recoverable PDF Wiki publication batch."""
        staged = self.repo.git.diff("--cached", "--name-only").strip()
        if staged:
            raise RuntimeError("Cannot start a Wiki batch with staged changes")
        return WikiRepositoryBatch(self, {}, set())

    def unstage(self, paths: list[str]) -> None:
        """Remove only the supplied paths from the index."""
        if not paths:
            return
        self.repo.git.restore("--staged", "--", *paths)

    def _commit(self, message: str) -> str:
        actor = Actor(self.identity.name, self.identity.email)
        commit = self.repo.index.commit(message, author=actor, committer=actor)
        return commit.hexsha
