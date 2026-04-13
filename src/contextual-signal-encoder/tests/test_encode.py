"""Tests for the contextual signal encoder.

Verifies that output conforms to:
1. Scoring service EmbeddingSegmentExt wire format
2. specs/v1.0/embedding_format.schema.json model metadata fields
3. Embedding space identification for interoperability
"""

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
            version="2.0.0",
            dimension=dim,
            embedding_space_id="aa://spaces/contextual/sentence-transformers/minilm-l6-v2",
            metric="cosine",
        )
    )
    return provider


class TestORTBWireFormat:
    """Output must match the scoring service's EmbeddingSegmentExt schema."""

    @pytest.mark.asyncio
    async def test_segment_ext_fields(self):
        encoder = ContextualEncoder(_mock_provider())
        resp = await encoder.encode(EncodeRequest(text="NFL playoff highlights"))

        ext = resp.data.segment[0].ext
        assert ext.ver == "1.0"
        assert ext.type == "context"
        assert ext.model == "all-MiniLM-L6-v2"
        assert ext.dimension == 384
        assert len(ext.vector) == 384

    @pytest.mark.asyncio
    async def test_plugs_into_score_request(self):
        encoder = ContextualEncoder(_mock_provider())
        resp = await encoder.encode(EncodeRequest(text="Test"))

        score_request = {
            "id": "bid-001",
            "user": {"id": "page", "data": [resp.data.model_dump()]},
            "top_k": 5,
        }
        parsed = json.loads(json.dumps(score_request))
        seg = parsed["user"]["data"][0]["segment"][0]
        assert seg["ext"]["dimension"] == len(seg["ext"]["vector"])


class TestSpecMetadata:
    """Model metadata fields per embedding_format.schema.json."""

    @pytest.mark.asyncio
    async def test_version_present(self):
        encoder = ContextualEncoder(_mock_provider())
        resp = await encoder.encode(EncodeRequest(text="Test"))
        assert resp.data.segment[0].ext.version == "2.0.0"

    @pytest.mark.asyncio
    async def test_embedding_space_id(self):
        encoder = ContextualEncoder(_mock_provider())
        resp = await encoder.encode(EncodeRequest(text="Test"))
        assert resp.data.segment[0].ext.embedding_space_id == \
            "aa://spaces/contextual/sentence-transformers/minilm-l6-v2"

    @pytest.mark.asyncio
    async def test_metric(self):
        encoder = ContextualEncoder(_mock_provider())
        resp = await encoder.encode(EncodeRequest(text="Test"))
        assert resp.data.segment[0].ext.metric == "cosine"

    @pytest.mark.asyncio
    async def test_private_lane_different_space(self):
        """A private-lane provider produces different metadata."""
        provider = AsyncMock()
        provider.encode_text = AsyncMock(
            return_value=EmbeddingResult(
                vector=np.random.randn(768).tolist(),
                model="proprietary-model-v3",
                version="3.1.0",
                dimension=768,
                embedding_space_id="aa://spaces/private/acme-corp/v3",
                metric="dot",
            )
        )
        encoder = ContextualEncoder(provider)
        resp = await encoder.encode(EncodeRequest(text="Test"))
        ext = resp.data.segment[0].ext

        assert ext.model == "proprietary-model-v3"
        assert ext.embedding_space_id == "aa://spaces/private/acme-corp/v3"
        assert ext.metric == "dot"


class TestBasicBehavior:

    @pytest.mark.asyncio
    async def test_unique_segment_ids(self):
        encoder = ContextualEncoder(_mock_provider())
        r1 = await encoder.encode(EncodeRequest(text="A"))
        r2 = await encoder.encode(EncodeRequest(text="B"))
        assert r1.data.segment[0].id != r2.data.segment[0].id

    @pytest.mark.asyncio
    async def test_empty_request_raises(self):
        encoder = ContextualEncoder(_mock_provider())
        with pytest.raises(ValueError):
            await encoder.encode(EncodeRequest())

    @pytest.mark.asyncio
    async def test_url_extraction(self):
        encoder = ContextualEncoder(_mock_provider())
        with patch("app.engine.encoder.extract_text_from_url", new_callable=AsyncMock) as mock:
            mock.return_value = "Extracted page content"
            resp = await encoder.encode(EncodeRequest(url="https://example.com"))
            assert resp.source == "url"
