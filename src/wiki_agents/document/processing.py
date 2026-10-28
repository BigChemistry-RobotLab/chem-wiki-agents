"""Convert source documents into representations used by wiki-agents."""

from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import cast

import pymupdf


def pdf_to_text(path: str | Path) -> list[str]:
    """Extract plain text from each PDF page in approximate reading order.

    Image-only pages are not OCRed.

    Args:
        path: Path to the PDF file to extract.

    Returns:
        A list of strings with extracted text, including empty strings for pages
        without extractable text.

    """
    with pymupdf.open(path) as document:
        return [
            cast("str", page.get_text("text", sort=False)).strip() for page in document
        ]


def assign_provenance(text: list[str] | str) -> list[str]:
    """Add stable page/line anchors to extracted text.

    A string represents the first page; a list contains one string per page.

    Args:
        text: Extracted text for one page, or a list containing the extracted
            text of each page in page order. Empty list entries represent empty
            pages and preserve page numbering.

    Returns:
        Extracted text with a stable HTML anchor for each line.

    """
    pages_to_process = [text] if isinstance(text, str) else text

    pages: list[str] = []
    for page_number, page in enumerate(pages_to_process, start=1):
        lines = page.splitlines()
        pages.append(
            "\n".join(
                f'<a id="p{page_number:04d}-l{line_number:04d}"></a> {line.strip()}'
                for line_number, line in enumerate(lines, start=1)
            ),
        )

    return pages


def source_id_from_sha(source_sha256: str) -> str:
    """Create a stable source ID.

    Args:
        source_sha256: Hexadecimal SHA-256 value for the source.

    Returns:
        A stable source identifier.

    """
    return sha256(f"wiki-source:{source_sha256}".encode()).hexdigest()


def pdf_to_okf_source(path: str | Path) -> tuple[str, list[str]]:
    """Convert a PDF into provenance-annotated Markdown with source metadata.

    Args:
        path: Path to the PDF file to convert.

    Returns:
        A tuple containing the YAML frontmatter and a list of provenance-annotated
        Markdown strings, one per page.

    """
    source_path = Path(path)
    timestamp = (
        datetime.now(UTC)
        .isoformat(timespec="microseconds")
        .replace(
            "+00:00",
            "Z",
        )
    )

    def quote(value: str) -> str:
        return "'" + value.replace("'", "''") + "'"

    frontmatter = (
        "---\n"
        "type: Source\n"
        f"created_at: {quote(timestamp)}\n"
        f"source_sha256: {sha256(source_path.read_bytes()).hexdigest()}\n"
        f"source: {quote(str(source_path))}\n"
        f"extraction_method: {quote(f'pymupdf:{pymupdf.VersionBind}')}\n"
        "---"
    )
    anchored_pages = assign_provenance(pdf_to_text(source_path))

    return frontmatter, anchored_pages
