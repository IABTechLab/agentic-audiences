"""Sentence Transformers provider — open-source, runs locally."""

from __future__ import annotations

import os

import numpy as np

from app.providers.base import EmbeddingProvider, EmbeddingResult

MODEL_NAME = os.getenv("ENCODER_MODEL", "all-MiniLM-L6-v2")


class SentenceTransformersProvider(EmbeddingProvider):

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self._model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        return self._model

    async def encode_text(self, text: str) -> EmbeddingResult:
        model = self._load_model()
        embedding = model.encode(text, normalize_embeddings=True)
        vector = np.asarray(embedding, dtype=np.float32).tolist()
        return EmbeddingResult(vector=vector, model=self._model_name, dimension=len(vector))

    async def close(self) -> None:
        self._model = None
