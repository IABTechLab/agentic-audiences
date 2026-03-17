from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class CampaignScore(BaseModel):
    campaign_id: str
    campaign_head_id: str
    score: float


class ScoreResponse(BaseModel):
    request_id: Optional[str] = None
    scores: list[CampaignScore]
    model: str
    embedding_type: str
    metric: str
