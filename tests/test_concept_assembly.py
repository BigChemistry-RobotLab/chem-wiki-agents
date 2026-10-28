from wiki_agents.concept.assembly import build_concept_chunk_work
from wiki_agents.document.models import SourceRef


def test_build_concept_chunk_work_skips_empty_pages() -> None:
    source = SourceRef(
        source_id="paper_test",
        sha256="a" * 64,
        resource="/references/papers/paper_test.md",
    )
    pages = ["Page one", "", "  \n", "Page four", ""]

    work_items = build_concept_chunk_work(source, pages, pages_per_chunk=3)

    assert [work.page_numbers for work in work_items] == [[1], [4]]
    assert [work.anchored_text for work in work_items] == ["Page one", "Page four"]


def test_build_concept_chunk_work_keeps_original_numbers_around_empty_page() -> None:
    source = SourceRef(
        source_id="paper_test",
        sha256="a" * 64,
        resource="/references/papers/paper_test.md",
    )
    pages = ["Page one", "", "Page three"]

    work_items = build_concept_chunk_work(source, pages, pages_per_chunk=3)

    assert len(work_items) == 1
    assert work_items[0].page_numbers == [1, 3]
    assert work_items[0].anchored_text == "Page one\n\nPage three"
