"""Embedding provider interfaces and OpenAI-compatible implementation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

from openai import InternalServerError, OpenAI

if TYPE_CHECKING:
    from collections.abc import Sequence


class _EmbeddingItem(Protocol):
    index: int
    embedding: Sequence[float]


class _EmbeddingResponse(Protocol):
    data: Sequence[_EmbeddingItem]


class _EmbeddingsResource(Protocol):
    def create(
        self,
        *,
        model: str,
        input: list[str],  # ruff: ignore[builtin-argument-shadowing] - API parity.
    ) -> _EmbeddingResponse: ...


class _EmbeddingClient(Protocol):
    embeddings: _EmbeddingsResource


@dataclass(frozen=True, slots=True)
class _EmbeddingChunk:
    text_index: int
    text: str
    weight: int


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Create embeddings for text."""

    model: str

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Create embeddings.

        Args:
            texts: Text items in the required order.

        Returns:
            One finite vector for each text item in the same order.

        """
        ...


class OpenAICompatibleEmbeddingProvider:
    """Use an OpenAI-compatible embeddings endpoint."""

    def __init__(  # ruff: ignore[too-many-arguments] - The endpoint requires separate settings.
        self,
        *,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        batch_size: int = 64,
        max_input_chars: int = 1200,
        client: _EmbeddingClient | None = None,
    ) -> None:
        """Create the provider.

        Args:
            model: Embedding model name.
            base_url: Optional endpoint base URL.
            api_key: Optional endpoint API key.
            batch_size: Maximum number of text items in one request.
            max_input_chars: Maximum characters in one embedding input.
            client: Optional OpenAI-compatible client for dependency injection.

        """
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if max_input_chars < 1:
            raise ValueError("max_input_chars must be positive")
        self.model = model
        self.batch_size = batch_size
        self.max_input_chars = max_input_chars
        self.client = cast(
            "_EmbeddingClient",
            client or OpenAI(base_url=base_url, api_key=api_key),
        )

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Create checked embeddings in batches.

        Args:
            texts: Text items in the required order.

        Returns:
            One finite vector for each text item in the same order.

        """
        if not texts:
            return []

        initial_chunks = [
            _EmbeddingChunk(text_index, chunk, max(len(chunk), 1))
            for text_index, text in enumerate(texts)
            for chunk in self._split_text(text)
        ]
        chunks: list[_EmbeddingChunk] = []
        chunk_vectors: list[list[float]] = []
        for start in range(0, len(initial_chunks), self.batch_size):
            batch_chunks = initial_chunks[start : start + self.batch_size]
            resolved_chunks, resolved_vectors = self._embed_chunk_batch(batch_chunks)
            chunks.extend(resolved_chunks)
            chunk_vectors.extend(resolved_vectors)

        if len(chunk_vectors) != len(chunks):
            raise ValueError("Embedding response count does not match the request")
        dimensions = {len(vector) for vector in chunk_vectors}
        if len(dimensions) != 1:
            raise ValueError("Embedding response has inconsistent dimensions")
        return self._pool_vectors(chunks, chunk_vectors, len(texts))

    def _embed_chunk_batch(
        self,
        chunks: Sequence[_EmbeddingChunk],
    ) -> tuple[list[_EmbeddingChunk], list[list[float]]]:
        """Embed chunks and reduce a request that exceeds the server batch limit."""
        batch = [chunk.text for chunk in chunks]
        try:
            response = self.client.embeddings.create(model=self.model, input=batch)
        except InternalServerError as error:
            if not self._is_oversized_input_error(error):
                raise
            if len(chunks) > 1:
                midpoint = len(chunks) // 2
                left_chunks, left_vectors = self._embed_chunk_batch(chunks[:midpoint])
                right_chunks, right_vectors = self._embed_chunk_batch(chunks[midpoint:])
                return left_chunks + right_chunks, left_vectors + right_vectors
            chunk = chunks[0]
            if len(chunk.text) <= 1:
                raise
            left_text, right_text = self._split_once(chunk.text, len(chunk.text) // 2)
            split_chunks = [
                _EmbeddingChunk(chunk.text_index, left_text, max(len(left_text), 1)),
                _EmbeddingChunk(chunk.text_index, right_text, max(len(right_text), 1)),
            ]
            return self._embed_chunk_batch(split_chunks)
        return list(chunks), self._checked_response_vectors(response, len(batch))

    @staticmethod
    def _is_oversized_input_error(error: InternalServerError) -> bool:
        """Return whether the server reports an insufficient physical batch size."""
        message = str(error).lower()
        return "too large to process" in message and "physical batch size" in message

    @staticmethod
    def _checked_response_vectors(
        response: _EmbeddingResponse,
        expected_count: int,
    ) -> list[list[float]]:
        """Validate one endpoint response and restore request order."""
        data = list(response.data)
        indexes = [item.index for item in data]
        if sorted(indexes) != list(range(expected_count)) or len(set(indexes)) != len(
            data,
        ):
            raise ValueError("Embedding response indexes do not match the request")
        data.sort(key=lambda item: item.index)
        vectors = [[float(value) for value in item.embedding] for item in data]
        if any(
            not vector or not all(math.isfinite(value) for value in vector)
            for vector in vectors
        ):
            raise ValueError("Embedding response contains an invalid vector")
        dimensions = {len(vector) for vector in vectors}
        if len(dimensions) > 1:
            raise ValueError("Embedding response has inconsistent dimensions")
        return vectors

    def _split_text(self, text: str) -> list[str]:
        """Split text at nearby whitespace before it reaches the input limit."""
        if len(text) <= self.max_input_chars:
            return [text]

        chunks: list[str] = []
        remaining = text
        minimum_split = self.max_input_chars // 2
        while len(remaining) > self.max_input_chars:
            split_at = max(
                remaining.rfind(" ", minimum_split, self.max_input_chars + 1),
                remaining.rfind("\n", minimum_split, self.max_input_chars + 1),
                remaining.rfind("\t", minimum_split, self.max_input_chars + 1),
            )
            if split_at < minimum_split:
                split_at = self.max_input_chars
            chunk = remaining[:split_at].strip()
            if chunk:
                chunks.append(chunk)
            remaining = remaining[split_at:].lstrip()
        if remaining or not chunks:
            chunks.append(remaining)
        return chunks

    @staticmethod
    def _split_once(text: str, target: int) -> tuple[str, str]:
        """Split text near the target and return two nonempty parts."""
        minimum_split = max(target // 2, 1)
        split_at = max(
            text.rfind(" ", minimum_split, target + 1),
            text.rfind("\n", minimum_split, target + 1),
            text.rfind("\t", minimum_split, target + 1),
        )
        if split_at < minimum_split:
            split_at = target
        left = text[:split_at].strip()
        right = text[split_at:].lstrip()
        if not left or not right:
            split_at = max(min(target, len(text) - 1), 1)
            left = text[:split_at]
            right = text[split_at:]
        return left, right

    @staticmethod
    def _pool_vectors(
        chunks: Sequence[_EmbeddingChunk],
        vectors: Sequence[Sequence[float]],
        text_count: int,
    ) -> list[list[float]]:
        """Create one length-weighted vector for each input text."""
        pooled: list[list[float]] = []
        for text_index in range(text_count):
            members = [
                (chunk.weight, vector)
                for chunk, vector in zip(chunks, vectors, strict=True)
                if chunk.text_index == text_index
            ]
            total_weight = sum(weight for weight, _ in members)
            combined = [
                sum(weight * vector[index] for weight, vector in members) / total_weight
                for index in range(len(members[0][1]))
            ]
            pooled.append(combined)
        return pooled
