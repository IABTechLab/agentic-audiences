"""Tests for the contextual signal encoder."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from app.engine.encoder import ContextualEncoder
from app.models.encode import EncodeRequest
from app.providers.base import EmbeddingResult


def _mock_provider(dim: int = 384):
    provider = AsyncMock()
    provider.encode_text = AsyncMock(
        return_value=EmbeddingResult(
            vector=np.random.randn(dim).tolist(),
            model="all-MiniLM-L6-v2",
            dimension=dim,
        )
    )
    return provider


class TestEncodeText:
    @pytest.mark.asyncio
    async def test_returns_ortb_segment(self):
        encoder = ContextualEncoder(_mock_provider())
        resp = await encoder.encode(EncodeRequest(text="NFL playoff highlights"))

        assert resp.data.id == "contextual-signal-encoder"
        assert len(resp.data.segment) == 1
        ext = resp.data.segment[0].ext
        assert ext.ver == "1.0"
        assert ext.type == "context"
        assert ext.model == "all-MiniLM-L6-v2"
        assert ext.dimension == 384
        assert len(ext.vector) == 384

    @pytest.mark.asyncio
    async def test_metadata(self):
        encoder = ContextualEncoder(_mock_provider())
        resp = await encoder.encode(EncodeRequest(text="Some content"))
        assert resp.source == "text"
        assert resp.content_length == len("Some content")

    @pytest.mark.asyncio
    async def test_segment_id_is_unique(self):
        encoder = ContextualEncoder(_mock_provider())
        r1 = await encoder.encode(EncodeRequest(text="A"))
        r2 = await encoder.encode(EncodeRequest(text="B"))
        assert r1.data.segment[0].id != r2.data.segment[0].id

    @pytest.mark.asyncio
    async def test_empty_request_raises(self):
        encoder = ContextualEncoder(_mock_provider())
        with pytest.raises(ValueError):
            await encoder.encode(EncodeRequest())


class TestORTBCompatibility:
    @pytest.mark.asyncio
    async def test_builds_valid_score_request(self):
        """Encoder output plugs directly into the scoring service."""
        encoder = ContextualEncoder(_mock_provider())
        resp = await encoder.encode(EncodeRequest(text="Test content"))

        score_request = {
            "id": "bid-001",
            "user": {"id": "page", "data": [resp.data.model_dump()]},
            "top_k": 5,
        }

        data = score_request["user"]["data"][0]
        seg = data["segment"][0]
        assert seg["ext"]["ver"] == "1.0"
        assert seg["ext"]["type"] == "context"
        assert seg["ext"]["dimension"] == len(seg["ext"]["vector"])

        # Serializes cleanly (no numpy types)
        json.loads(json.dumps(score_request))

    @pytest.mark.asyncio
    async def test_url_extraction(self):
        encoder = ContextualEncoder(_mock_provider())
        with patch("app.engine.encoder.extract_text_from_url", new_callable=AsyncMock) as mock:
            mock.return_value = "Extracted page content"
            resp = await encoder.encode(EncodeRequest(url="https://example.com"))
            assert resp.source == "url"
            mock.assert_called_once()
