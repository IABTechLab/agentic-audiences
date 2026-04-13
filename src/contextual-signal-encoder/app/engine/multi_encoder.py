"""Multi-model encoder — produces named vectors from content.

Runs content through multiple embedding models simultaneously, producing
one named vector per model. Each vector carries full model metadata for
routing, versioning, and compatibility checks.

This addresses the named vector / model permutation problem in the spec:
instead of separate ORTB segments with string-based model:type partitioning,
content is encoded once and stored as a single point with multiple named
vectors and indexed payload metadata.
"""

from __future__ import annotations

import uuid

from app.config import AppConfig, EncoderConfig, MixpeekConfig
from app.engine.extractor import extract_text_from_url, truncate_text
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

# Known model metadata (extend as needed)
MODEL_INFO: dict[str, dict] = {
    "all-MiniLM-L6-v2": {
        "architecture": "transformer",
        "pooling": "mean",
        "normalization": "l2_unit",
        "training_domain": ["general", "semantic-similarity"],
        "version": "2.0",
    },
    "all-mpnet-base-v2": {
        "architecture": "transformer",
        "pooling": "mean",
        "normalization": "l2_unit",
        "training_domain": ["general", "semantic-similarity"],
        "version": "2.0",
    },
    "mixpeek-multimodal": {
        "architecture": "multimodal-fusion",
        "pooling": "cls",
        "normalization": "l2_unit",
        "training_domain": ["general", "multimodal", "adtech"],
        "version": "1.0",
    },
}


class MultiModelEncoder:
    """Encodes content through multiple models, producing named vectors."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._providers: dict[str, EmbeddingProvider] = {}

    def _get_provider(self, spec: ModelSpec) -> EmbeddingProvider:
        """Get or create a provider for the given model spec."""
        cache_key = f"{spec.provider}:{spec.model or 'default'}"
        if cache_key not in self._providers:
            if spec.provider == "mixpeek":
                from app.providers.mixpeek import MixpeekProvider
                self._providers[cache_key] = MixpeekProvider(
                    self._config.mixpeek,
                    model_name=spec.model or "mixpeek-multimodal",
                )
            else:
                from app.providers.sentence_transformers import SentenceTransformersProvider
                encoder_cfg = EncoderConfig(
                    model_name=spec.model or self._config.encoder.model_name,
                )
                self._providers[cache_key] = SentenceTransformersProvider(encoder_cfg)
        return self._providers[cache_key]

    async def encode(self, request: MultiEncodeRequest) -> MultiEncodeResponse:
        """Encode content through all requested models."""
        # Extract text if URL provided
        text = request.text
        if request.url and not text:
            text = await extract_text_from_url(request.url, self._config.extraction)
            text = truncate_text(text, self._config.encoder.max_tokens)

        named_vectors: list[NamedVector] = []
        segments: list[Segment] = []

        for spec in request.models:
            provider = self._get_provider(spec)

            # Route to the right encoding method based on modality
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

            meta = ModelMetadata(
                embedding_type="context",
                architecture=info.get("architecture"),
                pooling=info.get("pooling"),
                normalization=info.get("normalization", "l2_unit"),
                training_domain=info.get("training_domain", []),
                version=info.get("version"),
            )

            named_vectors.append(NamedVector(
                name=spec.name,
                vector=result.vector,
                model=model_name,
                dimension=result.dimension,
                modality=spec.modality,
                provider=spec.provider,
                metadata=meta,
            ))

            # Also build ORTB segment for backwards compatibility
            segments.append(Segment(
                id=f"ctx-{spec.name}-{uuid.uuid4().hex[:6]}",
                name=f"contextual-{spec.name}",
                ext=EmbeddingExt(
                    vector=result.vector,
                    model=model_name,
                    dimension=result.dimension,
                    type="context",
                ),
            ))

        # Build the storage example showing how this maps to a vector DB point
        point_id = f"content-{uuid.uuid4().hex[:8]}"
        storage_example = {
            "_description": (
                "Named vector storage format. Each model produces a named vector "
                "stored under the same point_id. Payload metadata enables filtering "
                "by model characteristics at query time."
            ),
            "point_id": point_id,
            "named_vectors": {
                nv.name: {
                    "values": f"[{nv.dimension}d float32 vector]",
                    "dimension": nv.dimension,
                }
                for nv in named_vectors
            },
            "payload": {
                "models": {
                    nv.name: {
                        "model": nv.model,
                        "dimension": nv.dimension,
                        "modality": nv.modality,
                        "provider": nv.provider,
                        "architecture": nv.metadata.architecture,
                        "pooling": nv.metadata.pooling,
                        "normalization": nv.metadata.normalization,
                        "training_domain": nv.metadata.training_domain,
                        "version": nv.metadata.version,
                        "embedding_type": nv.metadata.embedding_type,
                    }
                    for nv in named_vectors
                },
            },
        }

        return MultiEncodeResponse(
            named_vectors=named_vectors,
            ortb_data=ContextualData(segment=segments),
            storage_example=storage_example,
        )

    async def close(self) -> None:
        for provider in self._providers.values():
            await provider.close()
        self._providers.clear()
