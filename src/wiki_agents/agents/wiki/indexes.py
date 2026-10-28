"""Generate deterministic OKF concept-directory indexes."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from wiki_agents.agents.wiki.validation import parse_okf_document

_RESERVED_NAMES = {"index.md", "log.md"}

if TYPE_CHECKING:
    from collections.abc import Mapping


def render_concept_index(directory: Path) -> bytes:
    """Render one index from immediate child concept frontmatter.

    Args:
        directory: Host directory that contains concept Markdown files.

    Returns:
        Deterministic UTF-8 index content.

    """
    files = {
        path.name: path.read_bytes()
        for path in sorted(directory.glob("*.md"))
        if path.name.lower() not in _RESERVED_NAMES
    }
    return render_concept_index_from_files(files, directory_name=directory.name)


def render_concept_index_from_files(
    files: Mapping[str, bytes],
    *,
    directory_name: str = "concepts",
) -> bytes:
    """Render an index from an in-memory final Wiki file mapping."""
    entries: list[tuple[str, str, str]] = []
    for raw_path, content in sorted(files.items()):
        path = PurePosixPath(raw_path)
        if path.name.lower() in _RESERVED_NAMES or path.suffix.lower() != ".md":
            continue
        frontmatter, _ = parse_okf_document(content)
        title = frontmatter.get("title")
        description = frontmatter.get("description")
        if not isinstance(title, str) or not title.strip():
            raise ValueError(f"Concept has no non-empty title: {raw_path}")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"Concept has no non-empty description: {raw_path}")
        entries.append((title.strip(), path.name, description.strip()))
    heading = (
        "Concepts"
        if directory_name == "concepts"
        else directory_name.replace(
            "-",
            " ",
        ).title()
    )
    lines = [f"# {heading}"]
    if entries:
        lines.extend(
            [
                "",
                *(
                    f"- [{title}]({path}) - {description}"
                    for title, path, description in sorted(
                        entries,
                        key=lambda item: (item[0].casefold(), item[1]),
                    )
                ),
            ],
        )
    return ("\n".join(lines) + "\n").encode()


def affected_index_path(target_path: str) -> str:
    """Return the repository-relative index for one concept target."""
    target = Path(target_path)
    return (target.parent / "index.md").as_posix()
