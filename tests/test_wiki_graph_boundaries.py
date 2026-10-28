"""Test graph metadata boundaries for Wiki page workers."""

import json
from typing import cast
from uuid import UUID

from wiki_agents.agents.wiki.aggregator import build_page_contexts, build_page_manifest
from wiki_agents.agents.wiki.models import (
    RankedWikiLink,
    WikiCandidatePairContribution,
    WikiPageDefinition,
    WikiSelectedLinkContext,
)
from wiki_agents.agents.wiki.workspace import writer_manifest_projection
from wiki_agents.concept.models import Claim, Concept, ConceptCandidate
from wiki_agents.document.models import EvidenceSpan, SourceRef


def _definition(page_id: str, number: int) -> WikiPageDefinition:
    return WikiPageDefinition(
        page_id=page_id,
        title=f"Title {page_id}",
        scope=f"Scope {page_id}",
        work_summary=[f"Work {page_id}"],
        member_candidate_ids=[UUID(int=number)],
    )


def _candidate(number: int) -> ConceptCandidate:
    return ConceptCandidate(
        candidate_id=UUID(int=number),
        source=SourceRef(
            source_id=f"paper-{number}",
            sha256=f"{number:064x}",
            resource=f"/references/papers/paper-{number}.md",
        ),
        concept=Concept(
            title=f"Concept {number}",
            description=f"Description {number}",
            claims=[
                Claim(
                    statement=f"Claim {number}",
                    evidence=[
                        EvidenceSpan(
                            start_line_id="p0001-l0001",
                            end_line_id="p0001-l0001",
                        ),
                    ],
                ),
            ],
        ),
    )


def _ranked_link(
    target_page_id: str,
    target_candidate: int,
    *,
    selected: bool,
) -> RankedWikiLink:
    return RankedWikiLink(
        target_page_id=target_page_id,
        relationships=(f"Relation to {target_page_id}",),
        semantic_score=0.8 if selected else 0.4,
        global_pagerank=0.3 if selected else 0.2,
        personalized_pagerank=0.4 if selected else 0.1,
        ranking_score=0.75 if selected else 0.30,
        selected_for_page=selected,
        contributing_candidate_pairs=(
            WikiCandidatePairContribution(
                source_candidate_id=UUID(int=1),
                target_candidate_id=UUID(int=target_candidate),
                similarity=0.8 if selected else 0.4,
            ),
        ),
    )


def _key_names(value: object) -> set[str]:
    if isinstance(value, dict):
        return {key for key in value if isinstance(key, str)} | {
            key for item in value.values() for key in _key_names(item)
        }
    if isinstance(value, list):
        return {key for item in value for key in _key_names(item)}
    return set()


def test_page_workers_receive_only_selected_links_without_ranking_metadata() -> None:
    definitions = [_definition("A", 1), _definition("B", 2), _definition("C", 3)]
    manifest = build_page_manifest(
        definitions,
        {
            "A": [
                _ranked_link("B", 2, selected=True),
                _ranked_link("C", 3, selected=False),
            ],
        },
        {"A": 0.5, "B": 0.3, "C": 0.2},
    )
    contexts = build_page_contexts(
        manifest,
        [_candidate(1), _candidate(2), _candidate(3)],
    )
    active = next(item for item in contexts if item.page_job.page_id == "A")

    assert set(WikiSelectedLinkContext.model_fields) == {
        "target_page_id",
        "target_path",
        "target_title",
        "relationship",
    }
    assert [item.target_page_id for item in active.selected_links] == ["B"]
    assert [item.target_path for item in active.page_job.plan.links] == [
        "/concepts/title-b.md",
    ]
    assert len(next(page for page in manifest.pages if page.page_id == "A").links) == 2

    projection = writer_manifest_projection(active, manifest)
    pages = cast("list[dict[str, object]]", projection["pages"])
    selected_outgoing = cast(
        "list[dict[str, object]]",
        projection["selected_outgoing_links"],
    )
    assert isinstance(pages, list)
    assert isinstance(selected_outgoing, list)
    assert {page["page_id"] for page in pages} == {"A", "B", "C"}
    assert [item["target_page_id"] for item in selected_outgoing] == ["B"]
    assert _key_names(active.model_dump(mode="json")).isdisjoint(
        {
            "semantic_score",
            "global_pagerank",
            "personalized_pagerank",
            "ranking_score",
            "selected_for_page",
            "contributing_candidate_pairs",
        },
    )
    serialized_projection = json.dumps(projection, sort_keys=True)
    assert "Relation to C" not in serialized_projection
    assert _key_names(projection).isdisjoint(
        {
            "semantic_score",
            "global_pagerank",
            "personalized_pagerank",
            "ranking_score",
            "selected_for_page",
            "contributing_candidate_pairs",
        },
    )
