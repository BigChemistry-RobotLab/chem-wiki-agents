"""Build and manage the concept store used by an orchestration run."""

from dataclasses import dataclass, field
from pathlib import Path

from qdrant_client import QdrantClient

from wiki_agents.concept.store import ConceptStore
from wiki_agents.embedding import (
    ConceptVectorStore,
    OpenAICompatibleEmbeddingProvider,
    WorkPackageVectorStore,
)
from wiki_agents.embedding.provider import EmbeddingProvider


@dataclass(frozen=True, slots=True)
class ConceptStoreSettings:
    """Configure the default vector-backed concept store."""

    repository_path: Path
    qdrant_url: str | None = None
    qdrant_path: str | Path | None = None
    qdrant_api_key: str | None = field(default=None, repr=False)
    qdrant_collection: str = "wiki-concepts"
    qdrant_work_package_collection: str = "wiki-work-packages"
    embedding_model: str = "embeddinggemma-300m"
    embedding_base_url: str | None = None
    embedding_api_key: str | None = field(default=None, repr=False)
    embedding_batch_size: int = 64
    embedding_max_input_chars: int = 1200

    def __post_init__(self) -> None:
        """Validate settings that apply to every store instance."""
        if self.qdrant_url and self.qdrant_path:
            raise ValueError("QDRANT_URL and QDRANT_PATH are mutually exclusive")
        if self.embedding_batch_size < 1:
            raise ValueError("embedding_batch_size must be positive")
        if self.embedding_max_input_chars < 1:
            raise ValueError("embedding_max_input_chars must be positive")


class ConceptStoreProvider:
    """Create a concept store lazily and close only owned resources."""

    def __init__(
        self,
        settings: ConceptStoreSettings,
        *,
        store: ConceptStore | None = None,
    ) -> None:
        """Create a provider.

        Args:
            settings: Settings for a store created by this provider.
            store: Optional caller-owned store.

        """
        self.settings = settings
        self._store = store
        self._work_package_store: WorkPackageVectorStore | None = None
        self._owned_client: QdrantClient | None = None
        self._owned_provider: EmbeddingProvider | None = None

    def get(self) -> ConceptStore:
        """Return the supplied store or lazily create the default store."""
        if self._store is not None:
            return self._store

        client, embedding_provider = self._owned_resources()
        store = ConceptVectorStore(
            client=client,
            provider=embedding_provider,
            collection_name=self.settings.qdrant_collection,
        )
        self._store = store
        return store

    def get_work_package_store(self) -> WorkPackageVectorStore:
        """Return the work-package store on the shared client and provider."""
        if self._work_package_store is not None:
            return self._work_package_store
        client, embedding_provider = self._owned_resources()
        self._work_package_store = WorkPackageVectorStore(
            client=client,
            provider=embedding_provider,
            collection_name=self.settings.qdrant_work_package_collection,
        )
        return self._work_package_store

    def close(self) -> None:
        """Close the owned Qdrant client, if one exists."""
        client = self._owned_client
        if client is None:
            return

        self._owned_client = None
        self._owned_provider = None
        self._store = None
        self._work_package_store = None
        client.close()

    def _owned_resources(self) -> tuple[QdrantClient, EmbeddingProvider]:
        """Create or return the one owned client and embedding provider."""
        if self._owned_client is not None and self._owned_provider is not None:
            return self._owned_client, self._owned_provider
        client = self._create_qdrant_client()
        try:
            provider = OpenAICompatibleEmbeddingProvider(
                model=self.settings.embedding_model,
                base_url=self.settings.embedding_base_url,
                api_key=self.settings.embedding_api_key,
                batch_size=self.settings.embedding_batch_size,
                max_input_chars=self.settings.embedding_max_input_chars,
            )
        except Exception:
            client.close()
            raise
        self._owned_client = client
        self._owned_provider = provider
        return client, provider

    def _create_qdrant_client(self) -> QdrantClient:
        """Create the Qdrant client from the configured location."""
        if self.settings.qdrant_url:
            return QdrantClient(
                url=self.settings.qdrant_url,
                api_key=self.settings.qdrant_api_key,
            )

        local_path = (
            str(self.settings.qdrant_path)
            if self.settings.qdrant_path is not None
            else str(self.settings.repository_path / ".qdrant")
        )
        return QdrantClient(path=local_path)
