"""Embedding provider interface.

The public lane ships with sentence-transformers (open-source, standardized).
The private lane is any custom provider implementing this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class EmbeddingResult:
    vector: list[float]
    model: str
    version: str
    dimension: int
    embedding_space_id: str
    metric: str = "cosine"


class EmbeddingProvider(ABC):
    """Interface for pluggable embedding backends (public or private lane)."""

    @abstractmethod
    async def encode_text(self, text: str) -> EmbeddingResult: ...

    @abstractmethod
    async def close(self) -> None: ...
