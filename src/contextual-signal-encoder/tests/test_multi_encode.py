"""Tests for multi-model encoding with named vectors."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from app.config import AppConfig
from app.engine.multi_encoder import MultiModelEncoder
from app.models.encode import ModelSpec, MultiEncodeRequest
from app.providers.base import EmbeddingResult


def _mock_result(dim: int, model: str) -> EmbeddingResult:
    return EmbeddingResult(
        vector=np.random.randn(dim).tolist(),
        model=model,
        dimension=dim,
    )


class TestMultiModelEncode:
    @pytest.fixture
    def config(self):
        return AppConfig()

    @pytest.fixture
    def encoder(self, config):
        return MultiModelEncoder(config)

    @pytest.mark.asyncio
    async def test_produces_named_vectors(self, encoder):
        """Each model spec produces a named vector."""
        with patch.object(encoder, "_get_provider") as mock_get:
            provider = AsyncMock()
            provider.encode_text = AsyncMock(side_effect=[
                _mock_result(384, "all-MiniLM-L6-v2"),
                _mock_result(768, "all-mpnet-base-v2"),
            ])
            mock_get.return_value = provider

            req = MultiEncodeRequest(
                text="NFL playoff highlights and predictions",
                models=[
                    ModelSpec(name="text-minilm", model="all-MiniLM-L6-v2"),
                    ModelSpec(name="text-mpnet", model="all-mpnet-base-v2"),
                ],
            )
            resp = await encoder.encode(req)

            assert len(resp.named_vectors) == 2
            assert resp.named_vectors[0].name == "text-minilm"
            assert resp.named_vectors[0].dimension == 384
            assert resp.named_vectors[1].name == "text-mpnet"
            assert resp.named_vectors[1].dimension == 768

    @pytest.mark.asyncio
    async def test_named_vectors_carry_metadata(self, encoder):
        """Each named vector has full model metadata."""
        with patch.object(encoder, "_get_provider") as mock_get:
            provider = AsyncMock()
            provider.encode_text = AsyncMock(return_value=_mock_result(384, "all-MiniLM-L6-v2"))
            mock_get.return_value = provider

            req = MultiEncodeRequest(
                text="Test content",
                models=[ModelSpec(name="text-v1", model="all-MiniLM-L6-v2")],
            )
            resp = await encoder.encode(req)

            nv = resp.named_vectors[0]
            assert nv.metadata.embedding_type == "context"
            assert nv.metadata.architecture == "transformer"
            assert nv.metadata.pooling == "mean"
            assert nv.metadata.normalization == "l2_unit"
            assert "general" in nv.metadata.training_domain

    @pytest.mark.asyncio
    async def test_ortb_segments_generated(self, encoder):
        """ORTB-compatible segments are also generated for backwards compat."""
        with patch.object(encoder, "_get_provider") as mock_get:
            provider = AsyncMock()
            provider.encode_text = AsyncMock(side_effect=[
                _mock_result(384, "all-MiniLM-L6-v2"),
                _mock_result(768, "all-mpnet-base-v2"),
            ])
            mock_get.return_value = provider

            req = MultiEncodeRequest(
                text="Test content",
                models=[
                    ModelSpec(name="text-minilm", model="all-MiniLM-L6-v2"),
                    ModelSpec(name="text-mpnet", model="all-mpnet-base-v2"),
                ],
            )
            resp = await encoder.encode(req)

            assert len(resp.ortb_data.segment) == 2
            assert "text-minilm" in resp.ortb_data.segment[0].id
            assert resp.ortb_data.segment[0].ext.type == "context"
            assert resp.ortb_data.segment[1].ext.dimension == 768

    @pytest.mark.asyncio
    async def test_storage_example_has_all_vectors(self, encoder):
        """Storage example shows named vectors and payload metadata."""
        with patch.object(encoder, "_get_provider") as mock_get:
            provider = AsyncMock()
            provider.encode_text = AsyncMock(side_effect=[
                _mock_result(384, "all-MiniLM-L6-v2"),
                _mock_result(768, "all-mpnet-base-v2"),
            ])
            mock_get.return_value = provider

            req = MultiEncodeRequest(
                text="Test content",
                models=[
                    ModelSpec(name="text-minilm", model="all-MiniLM-L6-v2"),
                    ModelSpec(name="text-mpnet", model="all-mpnet-base-v2"),
                ],
            )
            resp = await encoder.encode(req)

            store = resp.storage_example
            assert "point_id" in store
            assert "text-minilm" in store["named_vectors"]
            assert "text-mpnet" in store["named_vectors"]
            assert store["named_vectors"]["text-minilm"]["dimension"] == 384
            assert store["named_vectors"]["text-mpnet"]["dimension"] == 768

            # Payload metadata for filtering
            models_meta = store["payload"]["models"]
            assert models_meta["text-minilm"]["architecture"] == "transformer"
            assert models_meta["text-mpnet"]["model"] == "all-mpnet-base-v2"

    @pytest.mark.asyncio
    async def test_multimodal_named_vectors(self, encoder):
        """Different modalities produce named vectors on the same point."""
        with patch.object(encoder, "_get_provider") as mock_get:
            text_provider = AsyncMock()
            text_provider.encode_text = AsyncMock(return_value=_mock_result(384, "all-MiniLM-L6-v2"))

            image_provider = AsyncMock()
            image_provider.encode_image = AsyncMock(return_value=_mock_result(1152, "mixpeek-multimodal"))

            mock_get.side_effect = [text_provider, image_provider]

            req = MultiEncodeRequest(
                text="Sports article about basketball",
                image_url="https://example.com/basketball.jpg",
                models=[
                    ModelSpec(name="text-minilm", provider="sentence_transformers", model="all-MiniLM-L6-v2", modality="text"),
                    ModelSpec(name="image-siglip", provider="mixpeek", modality="image"),
                ],
            )
            resp = await encoder.encode(req)

            assert len(resp.named_vectors) == 2
            assert resp.named_vectors[0].name == "text-minilm"
            assert resp.named_vectors[0].modality == "text"
            assert resp.named_vectors[0].dimension == 384
            assert resp.named_vectors[1].name == "image-siglip"
            assert resp.named_vectors[1].modality == "image"
            assert resp.named_vectors[1].dimension == 1152

    @pytest.mark.asyncio
    async def test_response_serializes_to_json(self, encoder):
        """Full response serializes cleanly (no numpy types)."""
        with patch.object(encoder, "_get_provider") as mock_get:
            provider = AsyncMock()
            provider.encode_text = AsyncMock(return_value=_mock_result(384, "all-MiniLM-L6-v2"))
            mock_get.return_value = provider

            req = MultiEncodeRequest(
                text="Test",
                models=[ModelSpec(name="test-vec", model="all-MiniLM-L6-v2")],
            )
            resp = await encoder.encode(req)

            # Must serialize without numpy type errors
            json_str = json.dumps(resp.model_dump())
            parsed = json.loads(json_str)
            assert len(parsed["named_vectors"]) == 1
            assert parsed["named_vectors"][0]["name"] == "test-vec"


class TestScoringServiceCompatibility:
    """Named vectors must also produce valid ORTB segments for the scoring service."""

    @pytest.mark.asyncio
    async def test_multi_model_score_request(self):
        """Build a score request with multiple segments from multi-model output."""
        config = AppConfig()
        encoder = MultiModelEncoder(config)

        with patch.object(encoder, "_get_provider") as mock_get:
            provider = AsyncMock()
            provider.encode_text = AsyncMock(side_effect=[
                _mock_result(384, "all-MiniLM-L6-v2"),
                _mock_result(768, "all-mpnet-base-v2"),
            ])
            mock_get.return_value = provider

            req = MultiEncodeRequest(
                text="NFL Championship game preview",
                models=[
                    ModelSpec(name="text-minilm", model="all-MiniLM-L6-v2"),
                    ModelSpec(name="text-mpnet", model="all-mpnet-base-v2"),
                ],
            )
            resp = await encoder.encode(req)

            # Build ORTB score request — scoring service will route each
            # segment to matching campaign heads by model:type partition
            score_request = {
                "id": "bid-req-multi",
                "user": {
                    "id": "page-context",
                    "data": [resp.ortb_data.model_dump()],
                },
                "top_k": 5,
            }

            segments = score_request["user"]["data"][0]["segment"]
            assert len(segments) == 2

            # Each segment has different model → routes to different campaign heads
            models = {s["ext"]["model"] for s in segments}
            assert "all-MiniLM-L6-v2" in models
            assert "all-mpnet-base-v2" in models

            # Serializes cleanly
            json.dumps(score_request)
