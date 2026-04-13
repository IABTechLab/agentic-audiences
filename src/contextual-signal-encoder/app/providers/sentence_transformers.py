"""Sentence Transformers provider — open-source, runs locally."""

from __future__ import annotations

import numpy as np

from app.config import EncoderConfig
from app.providers.base import EmbeddingProvider, EmbeddingResult


class SentenceTransformersProvider(EmbeddingProvider):
    """Generate contextual embeddings using sentence-transformers models.

    Default model: all-MiniLM-L6-v2 (384 dimensions, cosine similarity).
    Any sentence-transformers model can be swapped in via config.
    """

    def __init__(self, config: EncoderConfig) -> None:
        self._config = config
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._config.model_name)
        return self._model

    async def encode_text(self, text: str) -> EmbeddingResult:
        model = self._load_model()
        embedding = model.encode(text, normalize_embeddings=True)

        vector = np.asarray(embedding, dtype=np.float32).tolist()

        return EmbeddingResult(
            vector=vector,
            model=self._config.model_name,
            dimension=len(vector),
        )

    async def close(self) -> None:
        self._model = None
