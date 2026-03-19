from __future__ import annotations

import numpy as np
import pytest

from app.engine.store import CampaignHeadStore
from tests.conftest import _test_config


def _make_head(campaign_id: str, head_id: str, dim: int = 8) -> dict:
    return {
        "campaign_id": campaign_id,
        "campaign_head_id": head_id,
        "weights": np.random.randn(dim).tolist(),
        "model": "test-model",
        "dimension": dim,
        "type": "identity",
    }


@pytest.fixture
def store():
    return CampaignHeadStore(config=_test_config())


@pytest.mark.asyncio
async def test_get_heads_returns_cached_object(store):
    await store.register([_make_head("c1", "h1")])

    ids1, w1 = await store.get_heads("test-model", "identity")
    ids2, w2 = await store.get_heads("test-model", "identity")

    # Same object from cache (identity check)
    assert ids1 is ids2
    assert w1 is w2


@pytest.mark.asyncio
async def test_register_invalidates_cache(store):
    await store.register([_make_head("c1", "h1")])
    _, w1 = await store.get_heads("test-model", "identity")

    await store.register([_make_head("c2", "h2")])
    _, w2 = await store.get_heads("test-model", "identity")

    assert w1 is not w2


@pytest.mark.asyncio
async def test_empty_partition_not_cached(store):
    ids, weights = await store.get_heads("test-model", "identity")
    assert ids == []
    # Empty results should not be cached
    assert "test-model:identity" not in store._cache


@pytest.mark.asyncio
async def test_cache_tensor_device():
    """If torch is available, verify the cached weight matrix is a Tensor on the expected device."""
    try:
        import torch
        from app.engine.device import DEVICE
    except ImportError:
        pytest.skip("torch not installed")

    store = CampaignHeadStore(config=_test_config())
    await store.register([_make_head("c1", "h1")])

    _, weights = await store.get_heads("test-model", "identity")
    assert isinstance(weights, torch.Tensor)
    assert weights.device == DEVICE
