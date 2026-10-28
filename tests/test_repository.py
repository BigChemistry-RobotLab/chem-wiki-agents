from hashlib import sha256

import pymupdf
import pytest

from wiki_agents.document.models import AnchoredPage, SourceRef
from wiki_agents.orchestrator import (
    ExtractedSourceConcepts,
    WikiOrchestrator,
    WikiOrchestratorConfig,
    normalize_filename_stem,
)
from wiki_agents.repository import BotIdentity, WikiRepository


def test_repository_initialization_is_idempotent_and_uses_bot_identity(
    tmp_path,
) -> None:
    path = tmp_path / "repository"
    repository = WikiRepository(
        path,
        identity=BotIdentity("Test Bot", "bot@example.test"),
    )

    first_commit = repository.repo.head.commit
    reopened = WikiRepository(path)

    assert reopened.repo.head.commit == first_commit
    assert first_commit.message == "chore: initialize wiki repository"
    assert first_commit.author.name == "Test Bot"
    assert first_commit.author.email == "bot@example.test"
    assert repository.repo.active_branch.name == "main"
    assert not (path / ".gitattributes").exists()
    assert (
        "/wiki/references/papers/artifacts/*.pdf\n" in (path / ".gitignore").read_text()
    )
    assert "/.checkpoints/\n" in (path / ".gitignore").read_text()
    assert "/.staging/\n" in (path / ".gitignore").read_text()
    assert (
        (path / "wiki/index.md")
        .read_text()
        .startswith(
            '---\nokf_version: "0.2"\n---\n',
        )
    )
    assert (
        not (path / "wiki/references/papers/index.md")
        .read_text()
        .startswith(
            "---",
        )
    )


def test_repository_allows_internal_state_but_rejects_other_untracked_files(
    tmp_path,
) -> None:
    path = tmp_path / "repository"
    artifact = path / "wiki/references/papers/artifacts/legacy.pdf"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"legacy")
    repository = WikiRepository(path)
    gitignore = path / ".gitignore"
    gitignore.write_text(
        gitignore.read_text()
        .replace("/.checkpoints/\n", "")
        .replace("/.staging/\n", ""),
    )
    repository.stage_and_commit([".gitignore"], "test: emulate old repository")
    (path / ".checkpoints/paper/checkpoint.json").parent.mkdir(parents=True)
    (path / ".checkpoints/paper/checkpoint.json").write_text("{}")
    (path / ".staging/paper/result.json").parent.mkdir(parents=True)
    (path / ".staging/paper/result.json").write_text("{}")
    assert ".checkpoints/paper/checkpoint.json" in repository.repo.untracked_files
    assert ".staging/paper/result.json" in repository.repo.untracked_files

    repository.assert_clean()
    (path / "unexpected.txt").write_text("unexpected")

    with pytest.raises(RuntimeError, match="unexpected untracked files"):
        repository.assert_clean()


def test_repository_batch_stages_exact_paths_and_rolls_back(tmp_path) -> None:
    repository = WikiRepository(tmp_path / "repository")
    existing_path = repository.path / "wiki/concepts/index.md"
    original = existing_path.read_bytes()
    batch = repository.begin_batch()

    batch.apply(
        {
            "wiki/concepts/index.md": b"# Changed\n",
            "wiki/concepts/new.md": b"new concept\n",
        },
    )

    assert set(
        repository.repo.git.diff("--cached", "--name-only").splitlines(),
    ) == {"wiki/concepts/index.md", "wiki/concepts/new.md"}
    batch.rollback()
    assert existing_path.read_bytes() == original
    assert not (repository.path / "wiki/concepts/new.md").exists()
    assert not repository.repo.git.diff("--cached", "--name-only")


def test_repository_batch_commits_exact_union(tmp_path) -> None:
    repository = WikiRepository(tmp_path / "repository")
    batch = repository.begin_batch()
    batch.apply({"wiki/concepts/one.md": b"one\n"})
    batch.apply({"wiki/concepts/two.md": b"two\n"})

    commit_sha = batch.commit("wiki: update paper_test")

    assert commit_sha == repository.repo.head.commit.hexsha
    assert set(repository.repo.head.commit.stats.files) == {
        "wiki/concepts/one.md",
        "wiki/concepts/two.md",
    }


@pytest.mark.parametrize(
    ("stem", "normalized"),
    [
        ("10.1002_Ange.202011977", "10.1002_ange.202011977"),
        ("  A paper (draft)  ", "a_paper_draft"),
        ("a___b", "a_b"),
    ],
)
def test_normalize_filename_stem(stem: str, normalized: str) -> None:
    assert normalize_filename_stem(stem) == normalized


def test_ingestion_commits_a_link_and_portable_snapshot(
    monkeypatch,
    tmp_path,
) -> None:
    repository_path = tmp_path / "repository"
    sources_path = tmp_path / "existing-papers"
    sources_path.mkdir()
    orchestrator = WikiOrchestrator(
        WikiOrchestratorConfig(
            repository_path=repository_path,
            sources_path=sources_path,
            extractor_model="extractor",
            critic_model="critic",
            fixer_model="fixer",
        ),
    )
    pdf_path = sources_path / "10.1002_ange.202011977.pdf"
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 72), "Paper text")
        document.save(pdf_path)
    pdf_bytes = pdf_path.read_bytes()
    digest = sha256(pdf_bytes).hexdigest()
    extracted = ExtractedSourceConcepts(
        source=SourceRef(
            source_id="paper_10.1002_ange.202011977",
            sha256=digest,
            resource="/references/papers/paper_10.1002_ange.202011977.md",
        ),
        pages=[
            AnchoredPage(
                page_number=1,
                text='<a id="p0001-l0001"></a> Paper text',
            ),
        ],
        candidates=[],
        concept_extraction_status="succeeded",
        concept_extraction_chunks=[],
    )
    monkeypatch.setattr(
        orchestrator.concept_pipeline,
        "extract_concepts",
        lambda path: extracted,
    )

    orchestrator.ingestion.ingest(pdf_path)

    repo = orchestrator.wiki_repository.repo
    artifact = "wiki/references/papers/artifacts/10.1002_ange.202011977.pdf"
    artifact_path = repository_path / artifact
    assert repo.head.commit.message == "ingest: add paper_10.1002_ange.202011977"
    assert set(repo.head.commit.stats.files) == {
        "extractions/paper_10.1002_ange.202011977.json",
        "wiki/references/papers/paper_10.1002_ange.202011977.md",
    }
    assert (repository_path / "wiki/references/papers/index.md").read_text() == (
        "# Papers\n"
    )
    assert artifact_path.is_symlink()
    assert artifact_path.resolve() == pdf_path.resolve()
    assert str(artifact_path.readlink()) == (
        "../../../../../existing-papers/10.1002_ange.202011977.pdf"
    )
    assert not repo.git.ls_files("--", artifact)
    assert artifact not in repo.untracked_files
    assert pdf_path.read_bytes() == pdf_bytes
    assert orchestrator._snapshot_pdf_paths() == {
        "paper_10.1002_ange.202011977": artifact_path,
    }
    snapshot = (
        repository_path / "extractions/paper_10.1002_ange.202011977.json"
    ).read_text()
    assert str(repository_path) not in snapshot
    assert '"reference_resource": "/references/papers/' in snapshot

    committed = repo.head.commit
    monkeypatch.setattr(
        orchestrator.concept_pipeline,
        "extract_concepts",
        lambda path: pytest.fail("unchanged source must not be extracted again"),
    )

    orchestrator.ingestion.ingest(pdf_path)

    assert pdf_path.read_bytes() == pdf_bytes
    assert repo.head.commit == committed


def test_incomplete_external_source_is_not_moved(monkeypatch, tmp_path) -> None:
    sources_path = tmp_path / "existing-papers"
    sources_path.mkdir()
    orchestrator = WikiOrchestrator(
        WikiOrchestratorConfig(
            repository_path=tmp_path / "repository",
            sources_path=sources_path,
            extractor_model="extractor",
            critic_model="critic",
            fixer_model="fixer",
        ),
    )
    pdf_path = sources_path / "incomplete.pdf"
    pdf_path.write_bytes(b"incomplete PDF")
    digest = sha256(pdf_path.read_bytes()).hexdigest()
    incomplete = ExtractedSourceConcepts(
        source=SourceRef(
            source_id="paper_incomplete",
            sha256=digest,
            resource="/references/papers/paper_incomplete.md",
        ),
        pages=[],
        candidates=[],
        concept_extraction_status="failed",
        concept_extraction_chunks=[],
    )
    monkeypatch.setattr(
        orchestrator.concept_pipeline,
        "extract_concepts",
        lambda path: incomplete,
    )

    assert orchestrator.ingestion.ingest(pdf_path) is None
    assert pdf_path.exists()
    assert orchestrator.wiki_repository.repo.head.commit.message == (
        "chore: initialize wiki repository"
    )


def test_corrupt_snapshot_requires_extraction(tmp_path) -> None:
    orchestrator = WikiOrchestrator(
        WikiOrchestratorConfig(
            repository_path=tmp_path / "repository",
            extractor_model="extractor",
            critic_model="critic",
            fixer_model="fixer",
        ),
    )
    source_path = orchestrator.sources_path / "paper.pdf"
    source_path.write_bytes(b"PDF")
    snapshot_path = orchestrator.repository_path / "extractions/paper_paper.json"
    snapshot_path.write_text("{not valid JSON")

    assert orchestrator.ingestion.requires_extraction(source_path)


def test_explicit_sources_path_must_be_an_existing_directory(tmp_path) -> None:
    with pytest.raises(ValueError, match=r"Sources path .* is not a directory"):
        WikiOrchestrator(
            WikiOrchestratorConfig(
                repository_path=tmp_path / "repository",
                sources_path=tmp_path / "missing",
                extractor_model="extractor",
                critic_model="critic",
                fixer_model="fixer",
            ),
        )
