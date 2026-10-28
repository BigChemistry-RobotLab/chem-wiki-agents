"""Store concept embeddings and find related concepts."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from qdrant_client import QdrantClient, models

from wiki_agents.concept.models import (
    ConceptCandidate,
    ConceptMatch,
)
from wiki_agents.embedding import formats as embedding_formats

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    from wiki_agents.document.models import SourceRef
    from wiki_agents.embedding.provider import EmbeddingProvider


def _required_payload(point: models.ScoredPoint) -> dict[str, Any]:
    """Return a point payload that was requested from Qdrant."""
    payload = point.payload
    if payload is None:
        raise ValueError("Qdrant point is missing its requested payload")
    return payload


class ConceptVectorStore:
    """Store and search concept vectors in a Qdrant collection."""

    def __init__(
        self,
        *,
        client: QdrantClient,
        provider: EmbeddingProvider,
        collection_name: str = "wiki-concepts",
    ) -> None:
        """Create the store interface.

        Args:
            client: Qdrant client used for storage and search.
            provider: Provider used to create embeddings.
            collection_name: Name of the Qdrant collection.

        """
        self.client = client
        self.provider = provider
        self.collection_name = collection_name

    def replace_source_candidates(
        self,
        source: SourceRef,
        candidates: Sequence[ConceptCandidate],
    ) -> None:
        """Replace all stored candidates for one source.

        Args:
            source: Source document to replace.
            candidates: Complete new candidate set for the source.

        """
        for candidate in candidates:
            if candidate.source.source_id != source.source_id:
                raise ValueError("Candidate source does not match the replaced source")

        existing_ids = self._source_point_ids(source.source_id)
        if not candidates:
            if existing_ids:
                self.client.delete(self.collection_name, list(existing_ids), wait=True)
            return

        texts = [
            embedding_formats.format_concept_document(candidate.concept)
            for candidate in candidates
        ]
        vectors = self._checked_vectors(self.provider.embed(texts), len(texts))
        self._ensure_collection(len(vectors[0]))

        points: list[models.PointStruct] = []
        new_ids: set[str] = set()
        for vector, candidate in zip(vectors, candidates, strict=True):
            point_id = candidate.candidate_id
            new_ids.add(str(point_id))
            payload: dict[str, Any] = {
                "record_type": "concept",
                "candidate_id": str(candidate.candidate_id),
                "source_id": source.source_id,
                "source_sha256": source.sha256,
                "source_resource": source.resource,
                "json": candidate.model_dump(mode="json"),
                "embedding_model": self.provider.model,
                "embedding_formats": self._embedding_formats(),
            }
            points.append(
                models.PointStruct(id=point_id, vector=vector, payload=payload),
            )

        self.client.upsert(self.collection_name, points, wait=True)
        stale_ids = existing_ids - new_ids
        if stale_ids:
            self.client.delete(self.collection_name, list(stale_ids), wait=True)

    def top_k_similar_concepts(
        self,
        candidate: ConceptCandidate,
        *,
        top_k: int = 5,
    ) -> list[ConceptMatch]:
        """Find related concepts using concept-level cosine similarity.

        Args:
            candidate: Current concept to use as the search query.
            top_k: Maximum number of matches to return.

        Returns:
            Concept matches in descending similarity order.

        """
        if top_k < 1:
            return []
        if not self.client.collection_exists(self.collection_name):
            return []

        query_text = embedding_formats.format_query(
            embedding_formats.format_concept_document(candidate.concept),
        )
        vector = self._checked_vectors(self.provider.embed([query_text]), 1)[0]
        self._verify_collection(len(vector))
        hits = self.client.query_points(
            self.collection_name,
            query=vector,
            query_filter=self._search_filter(candidate.candidate_id),
            limit=top_k,
            with_payload=True,
        ).points
        return [
            ConceptMatch(
                candidate=ConceptCandidate.model_validate(
                    _required_payload(hit)["json"],
                ),
                score=float(hit.score),
                concept_score=float(hit.score),
            )
            for hit in hits
        ]

    def _ensure_collection(self, dimension: int) -> None:
        if self.client.collection_exists(self.collection_name):
            self._verify_collection(dimension)
            return
        self.client.create_collection(
            self.collection_name,
            vectors_config=models.VectorParams(
                size=dimension,
                distance=models.Distance.COSINE,
            ),
            metadata={
                "embedding_model": self.provider.model,
                "embedding_formats": self._embedding_formats(),
            },
        )
        for field in ("record_type", "candidate_id", "source_id"):
            self.client.create_payload_index(
                self.collection_name,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
                wait=True,
            )

    def _verify_collection(self, dimension: int) -> None:
        info = self.client.get_collection(self.collection_name)
        vector_config = info.config.params.vectors
        stored_dimension = getattr(vector_config, "size", None)
        if stored_dimension != dimension:
            raise ValueError(
                f"Embedding dimension {dimension} does not match collection dimension "
                f"{stored_dimension}",
            )
        metadata = info.config.metadata or {}
        stored_model = metadata.get("embedding_model")
        if stored_model is not None and stored_model != self.provider.model:
            raise ValueError("Embedding model does not match the collection model")
        stored_formats = metadata.get("embedding_formats")
        if stored_formats is not None and stored_formats != self._embedding_formats():
            raise ValueError("Embedding formats do not match the collection formats")

    @staticmethod
    def _embedding_formats() -> dict[str, str]:
        return {
            "concept": embedding_formats.CONCEPT_DOCUMENT_FORMAT,
            "query": embedding_formats.QUERY_FORMAT,
        }

    def _source_point_ids(self, source_id: str) -> set[str]:
        if not self.client.collection_exists(self.collection_name):
            return set()
        points, offset = self.client.scroll(
            self.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="source_id",
                        match=models.MatchValue(value=source_id),
                    ),
                ],
            ),
            limit=256,
            with_payload=False,
            with_vectors=False,
        )
        ids = {str(point.id) for point in points}
        while offset is not None:
            points, offset = self.client.scroll(
                self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source_id",
                            match=models.MatchValue(value=source_id),
                        ),
                    ],
                ),
                limit=256,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            ids.update(str(point.id) for point in points)
        return ids

    @staticmethod
    def _checked_vectors(
        vectors: Sequence[Sequence[float]],
        expected_count: int,
    ) -> list[list[float]]:
        if len(vectors) != expected_count:
            raise ValueError("Embedding response count does not match the request")
        checked = [[float(value) for value in vector] for vector in vectors]
        if not checked or not checked[0]:
            raise ValueError("Embedding response contains an empty vector")
        dimension = len(checked[0])
        if any(
            len(vector) != dimension
            or not vector
            or not all(math.isfinite(value) for value in vector)
            for vector in checked
        ):
            raise ValueError("Embedding response contains invalid dimensions or values")
        return checked

    @staticmethod
    def _search_filter(candidate_id: UUID) -> models.Filter:
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="record_type",
                    match=models.MatchValue(value="concept"),
                ),
            ],
            must_not=[
                models.FieldCondition(
                    key="candidate_id",
                    match=models.MatchValue(value=str(candidate_id)),
                ),
            ],
        )
