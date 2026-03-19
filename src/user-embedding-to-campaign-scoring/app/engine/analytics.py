from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from sklearn.decomposition import PCA

from app.config import AnalyticsConfig
from app.models.analytics import (
    AnalyticsResponse,
    CampaignAnalytics,
    ScoreBucket,
)


@dataclass
class ScoringRecord:
    embedding: np.ndarray
    score: float


@dataclass
class AnalyticsTracker:
    config: AnalyticsConfig
    _records: dict[str, deque[ScoringRecord]] = field(default_factory=dict)
    _campaign_ids: dict[str, str] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def record(
        self,
        campaign_head_id: str,
        campaign_id: str,
        embedding: np.ndarray,
        score: float,
    ) -> None:
        async with self._lock:
            key = campaign_head_id
            if key not in self._records:
                self._records[key] = deque(
                    maxlen=self.config.max_embeddings_stored,
                )
            self._records[key].append(
                ScoringRecord(embedding=np.asarray(embedding, dtype=np.float32), score=score)
            )
            self._campaign_ids[campaign_head_id] = campaign_id

    async def get_analytics(
        self,
        campaign_id: str | None = None,
        campaign_head_id: str | None = None,
    ) -> AnalyticsResponse:
        async with self._lock:
            campaign_ids_map = self._campaign_ids
            results: list[CampaignAnalytics] = []

            for head_id, records in self._records.items():
                cid = campaign_ids_map.get(head_id, "unknown")
                if campaign_id and cid != campaign_id:
                    continue
                if campaign_head_id and head_id != campaign_head_id:
                    continue

                if not records:
                    continue

                embeddings = np.array([r.embedding for r in records])
                scores = np.array([r.score for r in records])
                buckets = self._compute_buckets(
                    embeddings, scores, self.config.score_buckets
                )
                results.append(
                    CampaignAnalytics(
                        campaign_id=cid,
                        campaign_head_id=head_id,
                        total_scored=len(records),
                        score_buckets=buckets,
                    )
                )

        return AnalyticsResponse(
            campaigns=results,
            reduction_method="pca",
            reduced_dimensions=self.config.pca_dimensions,
        )

    def _compute_buckets(
        self,
        embeddings: np.ndarray,
        scores: np.ndarray,
        bucket_edges: list[int],
    ) -> list[ScoreBucket]:
        percentiles = np.percentile(scores, bucket_edges)
        n_dims = min(self.config.pca_dimensions, embeddings.shape[1])

        reduced: np.ndarray | None = None
        if embeddings.shape[0] >= n_dims:
            pca = PCA(n_components=n_dims)
            reduced = pca.fit_transform(embeddings)

        buckets: list[ScoreBucket] = []
        for i in range(len(percentiles) - 1):
            lo, hi = percentiles[i], percentiles[i + 1]
            label = f"p{bucket_edges[i]}-p{bucket_edges[i + 1]}"
            if i < len(percentiles) - 2:
                mask = (scores >= lo) & (scores < hi)
            else:
                mask = (scores >= lo) & (scores <= hi)

            count = int(mask.sum())
            centroid = None
            if reduced is not None and count > 0:
                centroid = reduced[mask].mean(axis=0).tolist()

            buckets.append(
                ScoreBucket(
                    bucket_label=label,
                    count=count,
                    reduced_centroid=centroid,
                )
            )

        return buckets
