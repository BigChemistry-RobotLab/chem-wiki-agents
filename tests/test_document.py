from hashlib import sha256

import pymupdf

from wiki_agents.document import (
    assign_provenance,
    pdf_to_okf_source,
    pdf_to_text,
    source_id_from_sha,
)


def test_pdf_to_text_extracts_and_strips_each_page(tmp_path) -> None:
    pdf_path = tmp_path / "document.pdf"

    with pymupdf.open() as document:
        first_page = document.new_page()
        first_page.insert_text((72, 72), "First page")
        document.new_page()
        third_page = document.new_page()
        third_page.insert_text((72, 72), "Third page")
        document.save(pdf_path)

    assert pdf_to_text(pdf_path) == ["First page", "", "Third page"]


def test_pdf_to_okf_source_adds_metadata_and_linkable_page_line_ids(tmp_path) -> None:
    source_path = tmp_path / "source's document.pdf"

    with pymupdf.open() as document:
        first_page = document.new_page()
        first_page.insert_text((72, 72), "First line\nSecond line")
        second_page = document.new_page()
        second_page.insert_text((72, 72), "First line on page two")
        document.save(source_path)

    source_bytes = source_path.read_bytes()
    frontmatter, anchored_text = pdf_to_okf_source(source_path)

    assert frontmatter.startswith("---\ntype: Source\ncreated_at: '")
    assert f"source_sha256: {sha256(source_bytes).hexdigest()}" in frontmatter
    assert f"source: '{str(source_path).replace("'", "''")}'" in frontmatter
    assert f"extraction_method: 'pymupdf:{pymupdf.VersionBind}'" in frontmatter
    assert anchored_text == [
        '<a id="p0001-l0001"></a> First line\n<a id="p0001-l0002"></a> Second line',
        '<a id="p0002-l0001"></a> First line on page two',
    ]


def test_assign_provenance_preserves_page_numbers_across_empty_pages() -> None:
    result = assign_provenance(["Page one", "", "Page three"])

    assert result == [
        '<a id="p0001-l0001"></a> Page one',
        "",
        '<a id="p0003-l0001"></a> Page three',
    ]


def test_assign_provenance_treats_a_string_as_the_first_page_only() -> None:
    result = assign_provenance("First line\nSecond line")

    assert '<a id="p0001-l0001"></a> First line' in result[0]
    assert '<a id="p0001-l0002"></a> Second line' in result[0]
    assert 'id="p0002-' not in result[0]


def test_assign_provenance_strips_lines_but_preserves_blank_line_anchors() -> None:
    assert assign_provenance(["  First  \n   \n\tSecond\t", ""])[0] == (
        '<a id="p0001-l0001"></a> First\n'
        '<a id="p0001-l0002"></a> \n'
        '<a id="p0001-l0003"></a> Second'
    )
    assert assign_provenance("") == [""]
    assert assign_provenance([]) == []


def test_source_id_from_sha_is_stable_and_namespaced() -> None:
    digest = "a" * 64

    assert (
        source_id_from_sha(digest)
        == sha256(
            f"wiki-source:{digest}".encode(),
        ).hexdigest()
    )
    assert source_id_from_sha(digest) == source_id_from_sha(digest)
    assert source_id_from_sha("b" * 64) != source_id_from_sha(digest)
