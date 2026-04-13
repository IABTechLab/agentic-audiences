"""Base interface for embedding providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class EmbeddingResult:
    vector: list[float]
    model: str
    dimension: int


class EmbeddingProvider(ABC):
    @abstractmethod
    async def encode_text(self, text: str) -> EmbeddingResult: ...

    @abstractmethod
    async def close(self) -> None: ...
