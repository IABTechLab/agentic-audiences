from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from app.engine.scorer import score_embedding


@pytest.fixture
def head_ids():
    return [
        ("camp-1", "head-1a"),
        ("camp-1", "head-1b"),
        ("camp-2", "head-2a"),
    ]


@pytest.fixture
def head_weights():
    return torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float16,
    )


def test_cosine_scoring(head_ids, head_weights):
    embedding = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    results = score_embedding(
        embedding, head_ids, head_weights, metric="cosine", top_k=3
    )
    assert len(results) == 3
    assert results[0].campaign_head_id == "head-1a"
    assert results[0].score == pytest.approx(1.0, abs=1e-3)


def test_dot_product_scoring(head_ids, head_weights):
    embedding = np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    results = score_embedding(
        embedding, head_ids, head_weights, metric="dot", top_k=2
    )
    assert len(results) == 2
    assert results[0].campaign_head_id == "head-1a"
    assert results[0].score == pytest.approx(2.0, abs=0.01)


def test_l2_scoring(head_ids, head_weights):
    embedding = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    results = score_embedding(
        embedding, head_ids, head_weights, metric="l2", top_k=3
    )
    assert len(results) == 3
    assert results[0].campaign_head_id == "head-1a"
    assert results[0].score == pytest.approx(0.0, abs=0.01)


def test_top_k_limits(head_ids, head_weights):
    embedding = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    results = score_embedding(
        embedding, head_ids, head_weights, metric="cosine", top_k=1
    )
    assert len(results) == 1


def test_l2_normalization(head_ids, head_weights):
    embedding = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    results = score_embedding(
        embedding,
        head_ids,
        head_weights,
        metric="cosine",
        apply_l2_norm=True,
        top_k=3,
    )
    assert results[0].campaign_head_id == "head-1a"
    assert results[0].score == pytest.approx(1.0, abs=1e-3)


def test_empty_heads():
    results = score_embedding(
        np.array([1.0, 0.0]),
        [],
        torch.empty((0, 2), dtype=torch.float16),
        metric="cosine",
    )
    assert results == []


def test_unknown_metric(head_ids, head_weights):
    embedding = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="Unknown metric"):
        score_embedding(embedding, head_ids, head_weights, metric="unknown")


def test_torch_tensor_embedding(head_ids, head_weights):
    """Verify that a torch.Tensor embedding also works."""
    embedding = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    results = score_embedding(
        embedding, head_ids, head_weights, metric="cosine", top_k=3
    )
    assert len(results) == 3
    assert results[0].campaign_head_id == "head-1a"
    assert results[0].score == pytest.approx(1.0, abs=1e-3)
