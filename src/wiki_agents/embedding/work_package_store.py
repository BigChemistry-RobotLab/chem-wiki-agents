"""Store and retrieve normalized Wiki work-package embeddings."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from qdrant_client import QdrantClient, models

from wiki_agents.agents.wiki.models import WikiCandidateDecision, WikiCandidatePlan
from wiki_agents.embedding.formats import (
    QUERY_FORMAT,
    WORK_PACKAGE_FORMAT,
    format_query,
    format_work_package,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from wiki_agents.embedding.provider import EmbeddingProvider


@dataclass(frozen=True, slots=True)
class WorkPackageMatch:
    """Keep one similar work package and its cosine score."""

    plan: WikiCandidatePlan
    score: float


class WorkPackageVectorStore:
    """Store CREATE candidate plans in a separate Qdrant collection."""

    def __init__(
        self,
        *,
        client: QdrantClient,
        provider: EmbeddingProvider,
        collection_name: str = "wiki-work-packages",
    ) -> None:
        """Create a store that shares its client and provider with concept storage."""
        self.client = client
        self.provider = provider
        self.collection_name = collection_name

    def replace_work_packages(
        self,
        plans: Sequence[WikiCandidatePlan],
        *,
        mapping_fingerprint: str = "",
    ) -> None:
        """Replace the complete work-package collection with CREATE plans."""
        create_plans = [
            plan for plan in plans if plan.decision is WikiCandidateDecision.CREATE
        ]
        if any(plan.page_intent is None for plan in create_plans):
            raise ValueError("each CREATE work package requires a page intent")
        if not create_plans:
            self._delete_all_points()
            return
        intents = []
        for plan in create_plans:
            if plan.page_intent is None:
                raise ValueError("each CREATE work package requires a page intent")
            intents.append(plan.page_intent)
        texts = [format_work_package(intent) for intent in intents]
        vectors = self._checked_vectors(self.provider.embed(texts), len(texts))
        self._ensure_collection(len(vectors[0]))
        points = [
            models.PointStruct(
                id=plan.candidate_id,
                vector=vector,
                payload={
                    "record_type": "work_package",
                    "candidate_id": str(plan.candidate_id),
                    "json": plan.model_dump(mode="json"),
                    "mapping_fingerprint": mapping_fingerprint,
                    "embedding_model": self.provider.model,
                    "embedding_format": WORK_PACKAGE_FORMAT,
                },
            )
            for plan, vector in zip(create_plans, vectors, strict=True)
        ]
        self.client.upsert(self.collection_name, points, wait=True)
        new_ids = {str(plan.candidate_id) for plan in create_plans}
        stale_ids = self._point_ids() - new_ids
        if stale_ids:
            self.client.delete(self.collection_name, list(stale_ids), wait=True)

    def top_k_similar_work_packages(
        self,
        plan: WikiCandidatePlan,
        *,
        top_k: int = 20,
    ) -> list[WorkPackageMatch]:
        """Return bounded semantic neighbors for one CREATE work package."""
        if top_k < 1 or not self.client.collection_exists(self.collection_name):
            return []
        if plan.page_intent is None:
            raise ValueError("work-package search requires a page intent")
        vector = self._checked_vectors(
            self.provider.embed([format_query(format_work_package(plan.page_intent))]),
            1,
        )[0]
        self._verify_collection(len(vector))
        hits = self.client.query_points(
            self.collection_name,
            query=vector,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="record_type",
                        match=models.MatchValue(value="work_package"),
                    ),
                ],
                must_not=[
                    models.FieldCondition(
                        key="candidate_id",
                        match=models.MatchValue(value=str(plan.candidate_id)),
                    ),
                ],
            ),
            limit=top_k,
            with_payload=True,
        ).points
        return [
            WorkPackageMatch(
                plan=WikiCandidatePlan.model_validate(self._payload(hit)["json"]),
                score=float(hit.score),
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
                "embedding_format": WORK_PACKAGE_FORMAT,
                "query_format": QUERY_FORMAT,
            },
        )
        for field in ("record_type", "candidate_id", "mapping_fingerprint"):
            self.client.create_payload_index(
                self.collection_name,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
                wait=True,
            )

    def _verify_collection(self, dimension: int) -> None:
        info = self.client.get_collection(self.collection_name)
        vectors = info.config.params.vectors
        if getattr(vectors, "size", None) != dimension:
            raise ValueError(
                "embedding dimension does not match work-package collection",
            )
        metadata = info.config.metadata or {}
        if metadata.get("embedding_model", self.provider.model) != self.provider.model:
            raise ValueError("embedding model does not match work-package collection")
        if metadata.get("embedding_format", WORK_PACKAGE_FORMAT) != WORK_PACKAGE_FORMAT:
            raise ValueError("embedding format does not match work-package collection")

    def _point_ids(self) -> set[str]:
        if not self.client.collection_exists(self.collection_name):
            return set()
        ids: set[str] = set()
        offset: Any = None
        while True:
            points, offset = self.client.scroll(
                self.collection_name,
                limit=256,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            ids.update(str(point.id) for point in points)
            if offset is None:
                return ids

    def _delete_all_points(self) -> None:
        ids = self._point_ids()
        if ids:
            self.client.delete(self.collection_name, list(ids), wait=True)

    @staticmethod
    def _payload(point: models.ScoredPoint) -> dict[str, Any]:
        payload = point.payload
        if payload is None:
            raise ValueError("Qdrant work-package point has no payload")
        return payload

    @staticmethod
    def _checked_vectors(
        vectors: Sequence[Sequence[float]],
        expected_count: int,
    ) -> list[list[float]]:
        if len(vectors) != expected_count:
            raise ValueError("embedding response count does not match the request")
        checked = [[float(value) for value in vector] for vector in vectors]
        if not checked or not checked[0]:
            raise ValueError("embedding response contains an empty vector")
        dimension = len(checked[0])
        if any(
            len(vector) != dimension
            or not all(math.isfinite(value) for value in vector)
            for vector in checked
        ):
            raise ValueError("embedding response contains invalid dimensions or values")
        return checked
