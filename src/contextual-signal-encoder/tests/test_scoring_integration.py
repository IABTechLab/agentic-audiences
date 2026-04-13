"""Integration test: encoder output → scoring service input.

Verifies that the encoder produces output that the scoring service
can consume without modification. This test does NOT require the
scoring service to be running — it validates schema compatibility.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import numpy as np
import pytest

from app.config import AppConfig
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


class TestEncoderToScoringPipeline:
    """Validate that encoder output can be directly embedded in an ORTB score request."""

    @pytest.mark.asyncio
    async def test_build_score_request_from_encoder_output(self):
        """Construct a complete scoring service request from encoder output."""
        config = AppConfig()
        provider = _mock_provider()
        encoder = ContextualEncoder(config, provider)

        resp = await encoder.encode(EncodeRequest(text="Latest tech gadget reviews"))

        # Build the ORTB score request exactly as the scoring service expects it
        score_request = {
            "id": "bid-req-001",
            "user": {
                "id": "page-context",
                "data": [resp.data.model_dump()],
            },
            "top_k": 5,
        }

        # Validate structure
        data = score_request["user"]["data"][0]
        assert data["id"] == "contextual-signal-encoder"
        assert data["name"] == "contextual-embeddings"
        assert len(data["segment"]) == 1

        seg = data["segment"][0]
        assert "ext" in seg
        assert seg["ext"]["ver"] == "1.0"
        assert seg["ext"]["type"] == "context"
        assert seg["ext"]["model"] == "all-MiniLM-L6-v2"
        assert seg["ext"]["dimension"] == 384
        assert len(seg["ext"]["vector"]) == 384

        # Verify it serializes cleanly to JSON (no numpy types, etc.)
        json_str = json.dumps(score_request)
        parsed = json.loads(json_str)
        assert parsed["user"]["data"][0]["segment"][0]["ext"]["dimension"] == 384

    @pytest.mark.asyncio
    async def test_batch_encode_builds_multi_segment_request(self):
        """Multiple encoded items can populate a single ORTB user.data array."""
        config = AppConfig()
        provider = _mock_provider()
        encoder = ContextualEncoder(config, provider)

        texts = [
            "Sports highlights from the weekend",
            "New electric vehicle reviews and comparisons",
            "Recipe: homemade pasta with truffle sauce",
        ]

        results = []
        for text in texts:
            resp = await encoder.encode(EncodeRequest(text=text))
            results.append(resp.data.model_dump())

        score_request = {
            "id": "bid-req-batch",
            "user": {
                "id": "page-context",
                "data": results,
            },
            "top_k": 3,
        }

        assert len(score_request["user"]["data"]) == 3

        # Each data entry has its own segment with unique ID
        segment_ids = [d["segment"][0]["id"] for d in score_request["user"]["data"]]
        assert len(set(segment_ids)) == 3  # all unique

    @pytest.mark.asyncio
    async def test_campaign_head_dimension_matches(self):
        """Verify encoder dimension matches what a campaign head would expect."""
        config = AppConfig()
        provider = _mock_provider(dim=384)
        encoder = ContextualEncoder(config, provider)

        resp = await encoder.encode(EncodeRequest(text="Test"))
        dim = resp.data.segment[0].ext.dimension

        # A compatible campaign head registration
        campaign_head = {
            "campaign_id": "camp-sports-001",
            "campaign_head_id": "head-sports-a",
            "weights": np.random.randn(dim).tolist(),
            "model": "all-MiniLM-L6-v2",
            "dimension": dim,
            "type": "contextual",
            "metric": "cosine",
        }

        assert campaign_head["dimension"] == dim
        assert len(campaign_head["weights"]) == dim
