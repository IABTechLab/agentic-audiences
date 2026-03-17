from __future__ import annotations

from pydantic import BaseModel


class CampaignHead(BaseModel):
    campaign_id: str
    campaign_head_id: str
    weights: list[float]
    model: str
    dimension: int
    type: str  # embedding type (canonical or alias)
    # Model configuration — travels with the head
    metric: str = "cosine"  # cosine | dot | l2
    apply_l2_norm: bool = False
    embedding_space_id: str = ""
    compatible_with: list[str] = []


class CampaignHeadRegistration(BaseModel):
    heads: list[CampaignHead]


class CampaignHeadResponse(BaseModel):
    registered: int
    ids: list[str]
