from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Request

from app.models.analytics import AnalyticsResponse

router = APIRouter()


@router.get("/analytics", response_model=AnalyticsResponse)
async def get_analytics(
    request: Request,
    campaign_id: Optional[str] = None,
    campaign_head_id: Optional[str] = None,
):
    tracker = request.app.state.tracker
    return await tracker.get_analytics(
        campaign_id=campaign_id,
        campaign_head_id=campaign_head_id,
    )
