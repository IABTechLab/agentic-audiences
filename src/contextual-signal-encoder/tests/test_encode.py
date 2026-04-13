"""Tests for the contextual signal encoder."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from app.config import AppConfig
from app.engine.encoder import ContextualEncoder
from app.models.encode import EncodeRequest
from app.providers.base import EmbeddingResult


# ---------------------------------------------------------------------------
# Unit tests (mocked provider — no model download required)
# ---------------------------------------------------------------------------

def _mock_provider(dim: int = 384):
    provider = AsyncMock()
    provider.encode_text = AsyncMock(
        return_value=EmbeddingResult(
            vector=np.random.randn(dim).tolist(),
            model="test-model",
            dimension=dim,
        )
    )
    provider.encode_image = AsyncMock(
        return_value=EmbeddingResult(
            vector=np.random.randn(dim).tolist(),
            model="test-model",
            dimension=dim,
        )
    )
    provider.encode_video = AsyncMock(
        return_value=EmbeddingResult(
            vector=np.random.randn(dim).tolist(),
            model="test-model",
            dimension=dim,
        )
    )
    return provider


@pytest.fixture
def config():
    return AppConfig()


@pytest.fixture
def provider():
    return _mock_provider()


@pytest.fixture
def encoder(config, provider):
    return ContextualEncoder(config, provider)


class TestEncodeText:
    @pytest.mark.asyncio
    async def test_returns_ortb_segment(self, encoder):
        req = EncodeRequest(text="NFL playoff game highlights")
        resp = await encoder.encode(req)

        assert resp.data.id == "contextual-signal-encoder"
        assert len(resp.data.segment) == 1

        ext = resp.data.segment[0].ext
        assert ext.ver == "1.0"
        assert ext.type == "context"
        assert ext.model == "test-model"
        assert ext.dimension == 384
        assert len(ext.vector) == 384

    @pytest.mark.asyncio
    async def test_metadata_source_is_text(self, encoder):
        req = EncodeRequest(text="Some content")
        resp = await encoder.encode(req)

        assert resp.metadata.source == "text"
        assert resp.metadata.provider == "sentence_transformers"
        assert resp.metadata.content_length == len("Some content")

    @pytest.mark.asyncio
    async def test_segment_id_is_unique(self, encoder):
        req = EncodeRequest(text="Test")
        resp1 = await encoder.encode(req)
        resp2 = await encoder.encode(req)

        assert resp1.data.segment[0].id != resp2.data.segment[0].id


class TestEncodeURL:
    @pytest.mark.asyncio
    async def test_url_extraction(self, encoder, provider):
        with patch("app.engine.encoder.extract_text_from_url", new_callable=AsyncMock) as mock_extract:
            mock_extract.return_value = "Extracted page content about basketball"
            req = EncodeRequest(url="https://example.com/sports")
            resp = await encoder.encode(req)

            mock_extract.assert_called_once()
            assert resp.metadata.source == "url"
            provider.encode_text.assert_called_once()


class TestEncodeMultimodal:
    @pytest.mark.asyncio
    async def test_image_encoding(self, encoder, provider):
        req = EncodeRequest(image_url="https://example.com/image.jpg")
        resp = await encoder.encode(req)

        assert resp.metadata.source == "image_url"
        provider.encode_image.assert_called_once_with("https://example.com/image.jpg")

    @pytest.mark.asyncio
    async def test_video_encoding(self, encoder, provider):
        req = EncodeRequest(video_url="https://example.com/video.mp4")
        resp = await encoder.encode(req)

        assert resp.metadata.source == "video_url"
        provider.encode_video.assert_called_once_with("https://example.com/video.mp4")


class TestIABEnrichment:
    @pytest.mark.asyncio
    async def test_iab_categories_from_provider(self, config):
        provider = _mock_provider()
        provider.encode_text.return_value = EmbeddingResult(
            vector=np.random.randn(384).tolist(),
            model="mixpeek-multimodal",
            dimension=384,
            iab_categories=[
                {"name": "Basketball", "tier1": "Sports", "path": "Sports > Basketball", "score": 0.92},
                {"name": "NBA", "tier1": "Sports", "path": "Sports > Basketball > NBA", "score": 0.88},
            ],
        )
        encoder = ContextualEncoder(config, provider)

        req = EncodeRequest(text="NBA Finals game 7")
        resp = await encoder.encode(req)

        assert resp.metadata.iab_categories is not None
        assert len(resp.metadata.iab_categories) == 2
        assert resp.metadata.iab_categories[0].name == "Basketball"
        assert resp.metadata.iab_categories[0].tier1 == "Sports"

    @pytest.mark.asyncio
    async def test_no_iab_when_not_available(self, encoder):
        req = EncodeRequest(text="Some content")
        resp = await encoder.encode(req)

        assert resp.metadata.iab_categories is None


class TestValidation:
    @pytest.mark.asyncio
    async def test_empty_request_raises(self, encoder):
        req = EncodeRequest()
        with pytest.raises(ValueError, match="At least one"):
            await encoder.encode(req)


class TestORTBCompatibility:
    """Verify that encoder output matches the scoring service's expected input."""

    @pytest.mark.asyncio
    async def test_segment_ext_matches_scoring_schema(self, encoder):
        """The ext object must have: ver, vector, model, dimension, type."""
        req = EncodeRequest(text="Test content")
        resp = await encoder.encode(req)
        ext = resp.data.segment[0].ext

        # These are the exact fields the scoring service's EmbeddingSegmentExt expects
        assert hasattr(ext, "ver")
        assert hasattr(ext, "vector")
        assert hasattr(ext, "model")
        assert hasattr(ext, "dimension")
        assert hasattr(ext, "type")
        assert isinstance(ext.vector, list)
        assert all(isinstance(v, float) for v in ext.vector)
        assert ext.dimension == len(ext.vector)

    @pytest.mark.asyncio
    async def test_type_is_wire_format(self, encoder):
        """The type should be 'context' (wire format), resolved to 'contextual' by scoring service."""
        req = EncodeRequest(text="Test content")
        resp = await encoder.encode(req)
        assert resp.data.segment[0].ext.type == "context"

    @pytest.mark.asyncio
    async def test_data_structure_matches_ortb_user_data(self, encoder):
        """The data object should be placeable into user.data[] in a score request."""
        req = EncodeRequest(text="Test content")
        resp = await encoder.encode(req)
        data = resp.data

        # Must have id, name, segment[] to match ORTB Data model
        assert data.id is not None
        assert data.name is not None
        assert isinstance(data.segment, list)
        assert len(data.segment) > 0
        assert data.segment[0].id is not None
        assert data.segment[0].ext is not None
