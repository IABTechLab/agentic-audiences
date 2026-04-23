"""Public lane provider: sentence-transformers.

This is the reference "public lane" encoder — a standardized, open-source
model that any party can load to produce or read embeddings in a shared
embedding space. As discussed in the working group, this provides a baseline
for interoperability without requiring proprietary model access.
"""

from __future__ import annotations

import numpy as np

from app.providers.base import EmbeddingProvider, EmbeddingResult

# Default public lane model and its embedding space identifier
DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_VERSION = "2.0.0"
DEFAULT_SPACE = "aa://spaces/contextual/sentence-transformers/minilm-l6-v2"


class SentenceTransformersProvider(EmbeddingProvider):

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        version: str = DEFAULT_VERSION,
        embedding_space_id: str = DEFAULT_SPACE,
    ) -> None:
        self._model_name = model_name
        self._version = version
        self._embedding_space_id = embedding_space_id
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
        return EmbeddingResult(
            vector=vector,
            model=self._model_name,
            version=self._version,
            dimension=len(vector),
            embedding_space_id=self._embedding_space_id,
            metric="cosine",
        )

    async def close(self) -> None:
        self._model = None
