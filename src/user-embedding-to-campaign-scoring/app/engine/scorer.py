from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.engine.device import HAS_TORCH

if HAS_TORCH:
    import torch
    import torch.nn.functional as F


@dataclass
class ScoredHead:
    campaign_id: str
    campaign_head_id: str
    score: float


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    norm = np.where(norm == 0, 1.0, norm)
    return v / norm


def score_embedding(
    embedding: np.ndarray,
    head_ids: list[tuple[str, str]],
    head_weights: np.ndarray,
    metric: str = "cosine",
    apply_l2_norm: bool = False,
    top_k: int = 10,
) -> list[ScoredHead]:
    """Score an embedding against a matrix of campaign head weights.

    Args:
        embedding: 1-D embedding vector.
        head_ids: List of (campaign_id, campaign_head_id) tuples, aligned with rows of head_weights.
        head_weights: 2-D array/tensor of shape (n_heads, dim).
        metric: One of "cosine", "dot", "l2".
        apply_l2_norm: Whether to L2-normalize vectors before scoring.
        top_k: Number of top results to return.

    Returns:
        Sorted list of ScoredHead (descending by score).
    """
    if HAS_TORCH and isinstance(head_weights, torch.Tensor):
        return _score_torch(embedding, head_ids, head_weights, metric, apply_l2_norm, top_k)
    return _score_numpy(embedding, head_ids, head_weights, metric, apply_l2_norm, top_k)


def _score_numpy(
    embedding: np.ndarray,
    head_ids: list[tuple[str, str]],
    head_weights: np.ndarray,
    metric: str,
    apply_l2_norm: bool,
    top_k: int,
) -> list[ScoredHead]:
    if head_weights.shape[0] == 0:
        return []

    emb = embedding.astype(np.float32)
    weights = head_weights.astype(np.float32)

    if apply_l2_norm:
        emb = _l2_normalize(emb)
        weights = _l2_normalize(weights)

    if metric == "cosine":
        emb_norm = _l2_normalize(emb.reshape(1, -1)).squeeze()
        weights_norm = _l2_normalize(weights)
        scores = weights_norm @ emb_norm
    elif metric == "dot":
        scores = weights @ emb
    elif metric == "l2":
        dists = np.linalg.norm(weights - emb, axis=1)
        scores = -dists
    else:
        raise ValueError(f"Unknown metric: {metric}")

    top_k = min(top_k, len(scores))
    top_indices = np.argpartition(scores, -top_k)[-top_k:]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

    return [
        ScoredHead(
            campaign_id=head_ids[i][0],
            campaign_head_id=head_ids[i][1],
            score=float(scores[i]),
        )
        for i in top_indices
    ]


def _score_torch(
    embedding: np.ndarray | torch.Tensor,
    head_ids: list[tuple[str, str]],
    head_weights: torch.Tensor,
    metric: str,
    apply_l2_norm: bool,
    top_k: int,
) -> list[ScoredHead]:
    if head_weights.shape[0] == 0:
        return []

    device = head_weights.device

    if isinstance(embedding, np.ndarray):
        emb = torch.tensor(embedding, dtype=torch.float32, device=device)
    else:
        emb = embedding.to(dtype=torch.float32, device=device)

    weights = head_weights.to(dtype=torch.float32)

    if apply_l2_norm:
        emb = F.normalize(emb, dim=-1)
        weights = F.normalize(weights, dim=-1)

    if metric == "cosine":
        emb_n = emb if apply_l2_norm else F.normalize(emb, dim=-1)
        weights_n = weights if apply_l2_norm else F.normalize(weights, dim=-1)
        scores = weights_n @ emb_n
    elif metric == "dot":
        scores = weights @ emb
    elif metric == "l2":
        scores = -torch.norm(weights - emb, dim=1)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    k = min(top_k, scores.shape[0])
    top_scores, top_indices = torch.topk(scores, k)

    top_indices_cpu = top_indices.cpu().tolist()
    top_scores_cpu = top_scores.cpu().tolist()

    return [
        ScoredHead(
            campaign_id=head_ids[idx][0],
            campaign_head_id=head_ids[idx][1],
            score=top_scores_cpu[i],
        )
        for i, idx in enumerate(top_indices_cpu)
    ]
