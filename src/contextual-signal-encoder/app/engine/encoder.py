"""Core encoding engine — content in, ORTB segment out."""

from __future__ import annotations

import uuid

from app.engine.extractor import extract_text_from_url
from app.models.encode import (
    ContextualData,
    EmbeddingExt,
    EncodeRequest,
    EncodeResponse,
    Segment,
)
from app.providers.base import EmbeddingProvider

MAX_WORDS = 512


class ContextualEncoder:

    def __init__(self, provider: EmbeddingProvider) -> None:
        self._provider = provider

    async def encode(self, request: EncodeRequest) -> EncodeResponse:
        if request.url:
            text = await extract_text_from_url(request.url)
            source = "url"
        elif request.text:
            text = request.text
            source = "text"
        else:
            raise ValueError("Either text or url is required.")

        # Truncate to approximate token limit
        words = text.split()
        if len(words) > MAX_WORDS:
            text = " ".join(words[:MAX_WORDS])

        result = await self._provider.encode_text(text)

        segment = Segment(
            id=f"ctx-{uuid.uuid4().hex[:8]}",
            name=f"contextual-{source}",
            ext=EmbeddingExt(
                vector=result.vector,
                model=result.model,
                dimension=result.dimension,
                type="context",
            ),
        )

        return EncodeResponse(
            data=ContextualData(segment=[segment]),
            source=source,
            content_length=len(text),
        )
