from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class ScoreBucket(BaseModel):
    bucket_label: str  # e.g. "p0-p25"
    count: int
    reduced_centroid: Optional[list[float]] = None


class CampaignAnalytics(BaseModel):
    campaign_id: str
    campaign_head_id: str
    total_scored: int
    score_buckets: list[ScoreBucket]


class AnalyticsResponse(BaseModel):
    campaigns: list[CampaignAnalytics]
    reduction_method: str = "pca"
    reduced_dimensions: int = 3
