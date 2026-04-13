"""Tests for multi-model encoding with named vectors."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import numpy as np
import pytest

from app.engine.multi_encoder import MultiModelEncoder
from app.models.encode import ModelSpec, MultiEncodeRequest
from app.providers.base import EmbeddingResult


def _mock_result(dim, model):
    return EmbeddingResult(vector=np.random.randn(dim).tolist(), model=model, dimension=dim)


class TestMultiModelEncode:

    @pytest.mark.asyncio
    async def test_produces_named_vectors(self):
        enc = MultiModelEncoder()
        provider = AsyncMock()
        provider.encode_text = AsyncMock(side_effect=[
            _mock_result(384, "all-MiniLM-L6-v2"),
            _mock_result(768, "all-mpnet-base-v2"),
        ])
        enc._get_provider = lambda spec: provider

        resp = await enc.encode(MultiEncodeRequest(
            text="NFL playoffs",
            models=[ModelSpec(name="v1", model="all-MiniLM-L6-v2"), ModelSpec(name="v2", model="all-mpnet-base-v2")],
        ))

        assert len(resp.named_vectors) == 2
        assert resp.named_vectors[0].name == "v1"
        assert resp.named_vectors[0].dimension == 384
        assert resp.named_vectors[1].name == "v2"
        assert resp.named_vectors[1].dimension == 768

    @pytest.mark.asyncio
    async def test_metadata_attached(self):
        enc = MultiModelEncoder()
        provider = AsyncMock()
        provider.encode_text = AsyncMock(return_value=_mock_result(384, "all-MiniLM-L6-v2"))
        enc._get_provider = lambda spec: provider

        resp = await enc.encode(MultiEncodeRequest(
            text="Test", models=[ModelSpec(name="t", model="all-MiniLM-L6-v2")],
        ))

        meta = resp.named_vectors[0].metadata
        assert meta.architecture == "transformer"
        assert meta.pooling == "mean"

    @pytest.mark.asyncio
    async def test_ortb_segments_generated(self):
        enc = MultiModelEncoder()
        provider = AsyncMock()
        provider.encode_text = AsyncMock(side_effect=[
            _mock_result(384, "all-MiniLM-L6-v2"),
            _mock_result(768, "all-mpnet-base-v2"),
        ])
        enc._get_provider = lambda spec: provider

        resp = await enc.encode(MultiEncodeRequest(
            text="Test",
            models=[ModelSpec(name="a", model="all-MiniLM-L6-v2"), ModelSpec(name="b", model="all-mpnet-base-v2")],
        ))

        assert len(resp.ortb_data.segment) == 2
        models = {s.ext.model for s in resp.ortb_data.segment}
        assert "all-MiniLM-L6-v2" in models
        assert "all-mpnet-base-v2" in models

    @pytest.mark.asyncio
    async def test_storage_example(self):
        enc = MultiModelEncoder()
        provider = AsyncMock()
        provider.encode_text = AsyncMock(return_value=_mock_result(384, "all-MiniLM-L6-v2"))
        enc._get_provider = lambda spec: provider

        resp = await enc.encode(MultiEncodeRequest(
            text="Test", models=[ModelSpec(name="vec1", model="all-MiniLM-L6-v2")],
        ))

        store = resp.storage_example
        assert "point_id" in store
        assert "vec1" in store["named_vectors"]
        assert store["payload"]["models"]["vec1"]["model"] == "all-MiniLM-L6-v2"

    @pytest.mark.asyncio
    async def test_serializes_to_json(self):
        enc = MultiModelEncoder()
        provider = AsyncMock()
        provider.encode_text = AsyncMock(return_value=_mock_result(384, "all-MiniLM-L6-v2"))
        enc._get_provider = lambda spec: provider

        resp = await enc.encode(MultiEncodeRequest(
            text="Test", models=[ModelSpec(name="t", model="all-MiniLM-L6-v2")],
        ))
        parsed = json.loads(json.dumps(resp.model_dump()))
        assert len(parsed["named_vectors"]) == 1
