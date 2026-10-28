from __future__ import annotations

import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import httpx2
import pytest
from openai import InternalServerError
from qdrant_client import QdrantClient, models

from wiki_agents.agents.concept.extractor import Claim, Concept, EvidenceSpan
from wiki_agents.document.models import SourceRef
from wiki_agents.embedding import (
    ConceptVectorStore,
    OpenAICompatibleEmbeddingProvider,
    format_concept_document,
    format_query,
    make_concept_candidate,
)
from wiki_agents.embedding import formats as embedding_formats


def _claim(statement: str) -> Claim:
    return Claim(
        statement=statement,
        evidence=[
            EvidenceSpan(
                start_line_id="p0001-l0001",
                end_line_id="p0001-l0002",
            ),
        ],
    )


def _concept(title: str, description: str, *claims: str) -> Concept:
    return Concept(
        title=title,
        description=description,
        claims=[_claim(statement) for statement in claims],
    )


def _source(source_id: str = "source-a", digest: str = "a") -> SourceRef:
    return SourceRef(
        source_id=source_id,
        sha256=digest * 64,
        resource=f"/references/{source_id}.md",
    )


class FixedProvider:
    def __init__(self, dimension: int = 3, model: str = "test-model") -> None:
        self.dimension = dimension
        self.model = model
        self.calls: list[list[str]] = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return [
            [float(len(text)), 1.0, *([0.0] * (self.dimension - 2))] for text in texts
        ]


class KeywordProvider:
    model = "keyword-model"

    def embed(self, texts):
        vectors = []
        for text in texts:
            lower = text.lower()
            if "alpha" in lower:
                vectors.append([1.0, 0.0, 0.0])
            elif "beta" in lower:
                vectors.append([0.9, 0.1, 0.0])
            else:
                vectors.append([0.0, 1.0, 0.0])
        return vectors


class FakeEmbeddings:
    def __init__(self, responses) -> None:
        self.responses = iter(responses)
        self.inputs: list[list[str]] = []

    def create(self, *, model, input):
        self.inputs.append(input)
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        return SimpleNamespace(data=response)


@dataclass
class FakeEmbeddingClient:
    embeddings: Any


def _item(index: int, vector: list[float]):
    return SimpleNamespace(index=index, embedding=vector)


def _oversized_input_error(tokens: int = 528) -> InternalServerError:
    message = (
        f"input ({tokens} tokens) is too large to process. "
        "increase the physical batch size (current batch size: 512)"
    )
    return InternalServerError(
        message,
        response=httpx2.Response(
            500,
            request=httpx2.Request("POST", "https://embedding.example.test/v1"),
        ),
        body={"error": {"message": message}},
    )


def _payload(point) -> dict[str, Any]:
    assert point.payload is not None
    return point.payload


def test_embedding_formats_do_not_include_evidence() -> None:
    concept = _concept("Alpha", "A description", "A supported claim")

    assert format_concept_document(concept) == "title: Alpha | text: A description"
    assert format_query("Alpha") == "task: search result | query: Alpha"


def test_openai_provider_batches_and_restores_response_order() -> None:
    endpoint = FakeEmbeddings(
        [
            [_item(1, [2.0, 0.0]), _item(0, [1.0, 0.0])],
            [_item(0, [3.0, 0.0])],
        ],
    )
    provider = OpenAICompatibleEmbeddingProvider(
        model="test",
        batch_size=2,
        client=FakeEmbeddingClient(endpoint),
    )

    assert provider.embed(["one", "two", "three"]) == [
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0],
    ]
    assert endpoint.inputs == [["one", "two"], ["three"]]


def test_openai_provider_chunks_and_pools_long_inputs() -> None:
    endpoint = FakeEmbeddings(
        [
            [_item(0, [1.0, 0.0]), _item(1, [0.0, 1.0])],
            [_item(0, [3.0, 0.0])],
        ],
    )
    provider = OpenAICompatibleEmbeddingProvider(
        model="test",
        batch_size=2,
        max_input_chars=5,
        client=FakeEmbeddingClient(endpoint),
    )

    assert provider.embed(["alpha beta", "short"]) == [
        [5 / 9, 4 / 9],
        [3.0, 0.0],
    ]
    assert endpoint.inputs == [["alpha", "beta"], ["short"]]


def test_openai_provider_splits_oversized_request_batch() -> None:
    endpoint = FakeEmbeddings(
        [
            _oversized_input_error(),
            [_item(0, [1.0, 0.0])],
            [_item(0, [0.0, 1.0])],
        ],
    )
    provider = OpenAICompatibleEmbeddingProvider(
        model="test",
        batch_size=2,
        client=FakeEmbeddingClient(endpoint),
    )

    assert provider.embed(["one", "two"]) == [[1.0, 0.0], [0.0, 1.0]]
    assert endpoint.inputs == [["one", "two"], ["one"], ["two"]]


def test_openai_provider_splits_one_oversized_input() -> None:
    endpoint = FakeEmbeddings(
        [
            _oversized_input_error(),
            [_item(0, [1.0, 0.0]), _item(1, [0.0, 1.0])],
        ],
    )
    provider = OpenAICompatibleEmbeddingProvider(
        model="test",
        client=FakeEmbeddingClient(endpoint),
    )

    assert provider.embed(["alpha beta"]) == [[5 / 9, 4 / 9]]
    assert endpoint.inputs == [["alpha beta"], ["alpha", "beta"]]


@pytest.mark.parametrize("batch_size", [0, -1])
def test_openai_provider_rejects_nonpositive_batch_size(batch_size: int) -> None:
    with pytest.raises(ValueError, match="batch_size must be positive"):
        OpenAICompatibleEmbeddingProvider(model="test", batch_size=batch_size)


@pytest.mark.parametrize("max_input_chars", [0, -1])
def test_openai_provider_rejects_nonpositive_input_limit(
    max_input_chars: int,
) -> None:
    with pytest.raises(ValueError, match="max_input_chars must be positive"):
        OpenAICompatibleEmbeddingProvider(
            model="test",
            max_input_chars=max_input_chars,
        )


def test_openai_provider_empty_input_does_not_call_endpoint() -> None:
    endpoint = FakeEmbeddings([])
    provider = OpenAICompatibleEmbeddingProvider(
        model="test",
        client=FakeEmbeddingClient(endpoint),
    )

    assert provider.embed([]) == []
    assert endpoint.inputs == []


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ([_item(1, [1.0])], "indexes"),
        ([_item(0, [])], "invalid vector"),
        ([_item(0, [float("nan")])], "invalid vector"),
        (
            [_item(0, cast("list[float]", ["not-a-number"]))],
            "could not convert string to float",
        ),
        ([_item(0, [1.0]), _item(1, [1.0, 2.0])], "inconsistent dimensions"),
    ],
)
def test_openai_provider_rejects_malformed_responses(data, message) -> None:
    endpoint = FakeEmbeddings([data])
    provider = OpenAICompatibleEmbeddingProvider(
        model="test",
        client=FakeEmbeddingClient(endpoint),
    )

    with pytest.raises(ValueError, match=message):
        provider.embed(["one"] if len(data) == 1 else ["one", "two"])


def test_store_persists_provenance_and_linkage(tmp_path) -> None:
    path = tmp_path / "qdrant"
    source = _source()
    concept = _concept("Alpha", "Description", "First claim", "Second claim")
    candidate = make_concept_candidate(
        source=source,
        concept=concept,
    )
    client = QdrantClient(path=str(path))
    store = ConceptVectorStore(client=client, provider=FixedProvider())
    store.replace_source_candidates(source, [candidate])
    collection_name = store.collection_name
    client.close()

    reopened = QdrantClient(path=str(path))
    points, _ = reopened.scroll(collection_name, limit=10, with_payload=True)
    collection = reopened.get_collection(collection_name)
    assert len(points) == 1
    vectors = collection.config.params.vectors
    metadata = collection.config.metadata
    assert isinstance(vectors, models.VectorParams)
    assert vectors.size == 3
    assert metadata is not None
    assert metadata["embedding_model"] == "test-model"
    assert {_payload(point)["record_type"] for point in points} == {"concept"}
    assert {_payload(point)["candidate_id"] for point in points} == {
        str(candidate.candidate_id),
    }
    assert all("concept_ordinal" not in _payload(point) for point in points)
    assert all("page_number" not in _payload(point) for point in points)
    assert all("claim_json" not in _payload(point) for point in points)
    concept_payload = _payload(points[0])
    assert concept_payload["source_sha256"] == source.sha256
    assert concept_payload["source_resource"] == source.resource
    assert concept_payload["json"]["concept"] == concept.model_dump(
        mode="json",
    )
    assert "claim_index" not in concept_payload
    reopened.close()


def test_replace_is_idempotent_and_removes_stale_and_empty_records() -> None:
    client = QdrantClient(":memory:")
    source = _source()
    first = make_concept_candidate(
        source=source,
        concept=_concept("Alpha", "Old", "Old claim"),
    )
    second = make_concept_candidate(
        source=source,
        concept=_concept("Alpha", "New", "New claim"),
    )
    provider = FixedProvider()
    store = ConceptVectorStore(client=client, provider=provider)

    store.replace_source_candidates(source, [first])
    store.replace_source_candidates(source, [first])
    assert client.count(store.collection_name).count == 1

    store.replace_source_candidates(source, [second])
    points, _ = client.scroll(store.collection_name, limit=10, with_payload=True)
    assert len(points) == 1
    concept_point = points[0]
    assert _payload(concept_point)["json"]["concept"]["description"] == "New"

    call_count = len(provider.calls)
    store.replace_source_candidates(source, [])
    assert client.count(store.collection_name).count == 0
    assert len(provider.calls) == call_count


def test_replace_rejects_candidate_owned_by_another_source_before_embedding() -> None:
    provider = FixedProvider()
    store = ConceptVectorStore(client=QdrantClient(":memory:"), provider=provider)
    source = _source()
    foreign_candidate = make_concept_candidate(
        source=_source("source-b", "b"),
        concept=_concept("Alpha", "Description", "Claim"),
    )

    with pytest.raises(ValueError, match="source does not match"):
        store.replace_source_candidates(source, [foreign_candidate])

    assert provider.calls == []


def test_replacing_one_source_does_not_change_another_source() -> None:
    client = QdrantClient(":memory:")
    store = ConceptVectorStore(client=client, provider=FixedProvider())
    source_a = _source()
    source_b = _source("source-b", "b")
    original_a = make_concept_candidate(
        source=source_a,
        concept=_concept("Alpha", "Old", "Claim"),
    )
    replacement_a = make_concept_candidate(
        source=source_a,
        concept=_concept("Alpha", "New", "Claim"),
    )
    candidate_b = make_concept_candidate(
        source=source_b,
        concept=_concept("Beta", "Unchanged", "Claim"),
    )
    store.replace_source_candidates(source_a, [original_a])
    store.replace_source_candidates(source_b, [candidate_b])

    store.replace_source_candidates(source_a, [replacement_a])

    points, _ = client.scroll(store.collection_name, limit=10, with_payload=True)
    candidates = {
        _payload(point)["source_id"]: _payload(point)["json"] for point in points
    }
    assert set(candidates) == {source_a.source_id, source_b.source_id}
    assert candidates[source_a.source_id]["concept"]["description"] == "New"
    assert candidates[source_b.source_id]["concept"]["description"] == "Unchanged"


def test_store_rejects_dimension_and_model_mismatch() -> None:
    client = QdrantClient(":memory:")
    source = _source()
    candidate = make_concept_candidate(
        source=source,
        concept=_concept("Alpha", "Description", "Claim"),
    )
    store = ConceptVectorStore(client=client, provider=FixedProvider(3))
    store.replace_source_candidates(source, [candidate])

    wrong_dimension = ConceptVectorStore(client=client, provider=FixedProvider(4))
    with pytest.raises(ValueError, match="dimension"):
        wrong_dimension.replace_source_candidates(source, [candidate])

    wrong_model = ConceptVectorStore(
        client=client,
        provider=FixedProvider(3, model="another-model"),
    )
    with pytest.raises(ValueError, match="model"):
        wrong_model.replace_source_candidates(source, [candidate])


def test_store_rejects_embedding_format_mismatch(monkeypatch) -> None:
    client = QdrantClient(":memory:")
    source = _source()
    candidate = make_concept_candidate(
        source=source,
        concept=_concept("Alpha", "Description", "Claim"),
    )
    store = ConceptVectorStore(client=client, provider=FixedProvider())
    store.replace_source_candidates(source, [candidate])
    monkeypatch.setattr(
        embedding_formats,
        "QUERY_FORMAT",
        "changed query format: {text}",
    )

    with pytest.raises(ValueError, match="formats"):
        store.top_k_similar_concepts(candidate)


def test_candidate_id_is_deterministic_and_source_or_content_sensitive() -> None:
    concept = _concept("Alpha", "Description", "Claim")
    first = make_concept_candidate(source=_source(), concept=concept)

    assert (
        make_concept_candidate(source=_source(), concept=concept).candidate_id
        == first.candidate_id
    )
    assert (
        make_concept_candidate(
            source=_source("source-b", "b"), concept=concept,
        ).candidate_id
        != first.candidate_id
    )
    assert (
        make_concept_candidate(
            source=_source(),
            concept=_concept("Alpha", "Changed", "Claim"),
        ).candidate_id
        != first.candidate_id
    )


@pytest.mark.parametrize("top_k", [0, -1])
def test_retrieval_nonpositive_top_k_does_not_embed(top_k: int) -> None:
    provider = FixedProvider()
    store = ConceptVectorStore(client=QdrantClient(":memory:"), provider=provider)
    candidate = make_concept_candidate(
        source=_source(),
        concept=_concept("Alpha", "Description", "Claim"),
    )

    assert store.top_k_similar_concepts(candidate, top_k=top_k) == []
    assert provider.calls == []


def test_retrieval_absent_collection_does_not_embed() -> None:
    provider = FixedProvider()
    store = ConceptVectorStore(client=QdrantClient(":memory:"), provider=provider)
    candidate = make_concept_candidate(
        source=_source(),
        concept=_concept("Alpha", "Description", "Claim"),
    )

    assert store.top_k_similar_concepts(candidate) == []
    assert provider.calls == []


def test_retrieval_excludes_self_and_includes_same_source_sibling() -> None:
    client = QdrantClient(":memory:")
    source = _source()
    alpha = make_concept_candidate(
        source=source,
        concept=_concept("Alpha", "Alpha process", "Alpha claim"),
    )
    beta = make_concept_candidate(
        source=source,
        concept=_concept("Beta", "Beta process", "Beta claim"),
    )
    other_source = _source("source-b", "b")
    gamma = make_concept_candidate(
        source=other_source,
        concept=_concept("Gamma", "Gamma process", "Gamma claim"),
    )
    store = ConceptVectorStore(client=client, provider=KeywordProvider())
    store.replace_source_candidates(source, [alpha, beta])
    store.replace_source_candidates(other_source, [gamma])

    matches = store.top_k_similar_concepts(alpha, top_k=5)

    assert [match.candidate.candidate_id for match in matches] == [
        beta.candidate_id,
        gamma.candidate_id,
    ]
    assert matches[0].candidate.source.source_id == source.source_id
    assert matches[0].matched_claim is None
    assert matches[0].concept_score is not None
    assert matches[0].claim_score is None
    assert len({match.candidate.candidate_id for match in matches}) == len(matches)

    limited = store.top_k_similar_concepts(alpha, top_k=1)
    assert [match.candidate.candidate_id for match in limited] == [beta.candidate_id]


@pytest.mark.skipif(
    os.getenv("RUN_EMBEDDING_SMOKE_TEST") != "1",
    reason="Set RUN_EMBEDDING_SMOKE_TEST=1 to use the real embedding endpoint",
)
def test_real_embedding_endpoint_smoke() -> None:
    provider = OpenAICompatibleEmbeddingProvider(
        model=os.getenv("EMBEDDING_DEFAULT_MODEL", "embeddinggemma-300m"),
        base_url=os.environ["EMBEDDING_BASE_URL"],
        api_key=os.getenv("EMBEDDING_API_KEY"),
    )

    vectors = provider.embed(["task: search result | query: chemistry"])

    assert len(vectors) == 1
    assert vectors[0]
