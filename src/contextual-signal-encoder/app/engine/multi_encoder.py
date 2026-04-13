"""Multi-model encoder — one content item, multiple named vectors."""

from __future__ import annotations

import uuid

from app.engine.extractor import extract_text_from_url
from app.models.encode import (
    ContextualData,
    EmbeddingExt,
    ModelMetadata,
    ModelSpec,
    MultiEncodeRequest,
    MultiEncodeResponse,
    NamedVector,
    Segment,
)
from app.providers.base import EmbeddingProvider

MODEL_INFO: dict[str, dict] = {
    "all-MiniLM-L6-v2": {"architecture": "transformer", "pooling": "mean", "training_domain": ["general"], "version": "2.0"},
    "all-mpnet-base-v2": {"architecture": "transformer", "pooling": "mean", "training_domain": ["general"], "version": "2.0"},
    "mixpeek-multimodal": {"architecture": "multimodal-fusion", "pooling": "cls", "training_domain": ["general", "multimodal", "adtech"], "version": "1.0"},
}


class MultiModelEncoder:

    def __init__(self) -> None:
        self._providers: dict[str, EmbeddingProvider] = {}

    def _get_provider(self, spec: ModelSpec) -> EmbeddingProvider:
        key = f"{spec.provider}:{spec.model or 'default'}"
        if key not in self._providers:
            if spec.provider == "mixpeek":
                from app.providers.mixpeek import MixpeekProvider
                self._providers[key] = MixpeekProvider(model_name=spec.model or "mixpeek-multimodal")
            else:
                from app.providers.sentence_transformers import SentenceTransformersProvider
                self._providers[key] = SentenceTransformersProvider(model_name=spec.model or "all-MiniLM-L6-v2")
        return self._providers[key]

    async def encode(self, request: MultiEncodeRequest) -> MultiEncodeResponse:
        text = request.text
        if request.url and not text:
            text = await extract_text_from_url(request.url)

        named_vectors: list[NamedVector] = []
        segments: list[Segment] = []

        for spec in request.models:
            provider = self._get_provider(spec)

            if spec.modality == "video" and request.video_url:
                result = await provider.encode_video(request.video_url)
            elif spec.modality == "image" and request.image_url:
                result = await provider.encode_image(request.image_url)
            elif text:
                result = await provider.encode_text(text)
            else:
                continue

            model_name = spec.model or result.model
            info = MODEL_INFO.get(model_name, {})

            named_vectors.append(NamedVector(
                name=spec.name,
                vector=result.vector,
                model=model_name,
                dimension=result.dimension,
                modality=spec.modality,
                provider=spec.provider,
                metadata=ModelMetadata(
                    architecture=info.get("architecture"),
                    pooling=info.get("pooling"),
                    training_domain=info.get("training_domain", []),
                    version=info.get("version"),
                ),
            ))

            segments.append(Segment(
                id=f"ctx-{spec.name}-{uuid.uuid4().hex[:6]}",
                name=f"contextual-{spec.name}",
                ext=EmbeddingExt(vector=result.vector, model=model_name, dimension=result.dimension, type="context"),
            ))

        storage_example = {
            "point_id": f"content-{uuid.uuid4().hex[:8]}",
            "named_vectors": {nv.name: {"dimension": nv.dimension} for nv in named_vectors},
            "payload": {
                "models": {
                    nv.name: {"model": nv.model, "dimension": nv.dimension, "modality": nv.modality,
                              "architecture": nv.metadata.architecture, "pooling": nv.metadata.pooling,
                              "version": nv.metadata.version}
                    for nv in named_vectors
                }
            },
        }

        return MultiEncodeResponse(named_vectors=named_vectors, ortb_data=ContextualData(segment=segments), storage_example=storage_example)

    async def close(self) -> None:
        for p in self._providers.values():
            await p.close()
        self._providers.clear()
