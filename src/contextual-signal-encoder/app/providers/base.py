"""Base interface for embedding providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class EmbeddingResult:
    """Result from an embedding provider."""

    vector: list[float]
    model: str
    dimension: int
    # Optional IAB taxonomy enrichment (provider-dependent)
    iab_categories: list[dict] | None = None


class EmbeddingProvider(ABC):
    """Interface for pluggable embedding backends."""

    @abstractmethod
    async def encode_text(self, text: str) -> EmbeddingResult:
        """Encode text content into a contextual embedding vector."""

    async def encode_image(self, image_url: str) -> EmbeddingResult:
        """Encode image content. Override in multimodal providers."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support image encoding. "
            "Use the Mixpeek provider for multimodal content."
        )

    async def encode_video(self, video_url: str) -> EmbeddingResult:
        """Encode video content. Override in multimodal providers."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support video encoding. "
            "Use the Mixpeek provider for multimodal content."
        )

    @abstractmethod
    async def close(self) -> None:
        """Release resources."""
