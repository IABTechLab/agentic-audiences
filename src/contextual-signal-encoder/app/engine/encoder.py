"""Core encoding engine — orchestrates content extraction and embedding."""

from __future__ import annotations

import uuid

from app.config import AppConfig
from app.engine.extractor import extract_text_from_url, truncate_text
from app.models.encode import (
    ContextualData,
    EncodeMetadata,
    EncodeRequest,
    EncodeResponse,
    IABCategory,
    Segment,
    EmbeddingExt,
)
from app.providers.base import EmbeddingProvider


class ContextualEncoder:
    """Encodes content into ORTB-compatible contextual embeddings."""

    def __init__(self, config: AppConfig, provider: EmbeddingProvider) -> None:
        self._config = config
        self._provider = provider

    async def encode(self, request: EncodeRequest) -> EncodeResponse:
        """Encode content from any supported input source."""

        # Determine input source and get content
        if request.video_url:
            result = await self._provider.encode_video(request.video_url)
            source = "video_url"
            content_length = len(request.video_url)
        elif request.image_url:
            result = await self._provider.encode_image(request.image_url)
            source = "image_url"
            content_length = len(request.image_url)
        elif request.url:
            text = await extract_text_from_url(request.url, self._config.extraction)
            text = truncate_text(text, self._config.encoder.max_tokens)
            result = await self._provider.encode_text(text)
            source = "url"
            content_length = len(text)
        elif request.text:
            text = truncate_text(request.text, self._config.encoder.max_tokens)
            result = await self._provider.encode_text(text)
            source = "text"
            content_length = len(text)
        else:
            raise ValueError("At least one of text, url, image_url, or video_url is required.")

        # Build ORTB-compatible segment
        segment = Segment(
            id=f"ctx-{uuid.uuid4().hex[:8]}",
            name=f"contextual-{source}",
            ext=EmbeddingExt(
                vector=result.vector,
                model=result.model,
                dimension=result.dimension,
                type=self._config.encoder.embedding_type,
            ),
        )

        # Build metadata
        iab_categories = None
        if result.iab_categories:
            iab_categories = [
                IABCategory(
                    name=c.get("name", ""),
                    tier1=c.get("tier1", ""),
                    path=c.get("path", ""),
                    score=c.get("score", 0.0),
                )
                for c in result.iab_categories
            ]

        provider_name = self._config.encoder.provider
        if request.provider:
            provider_name = request.provider

        metadata = EncodeMetadata(
            source=source,
            provider=provider_name,
            content_length=content_length,
            iab_categories=iab_categories,
        )

        return EncodeResponse(
            data=ContextualData(segment=[segment]),
            metadata=metadata,
        )
