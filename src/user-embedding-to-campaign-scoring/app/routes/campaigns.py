from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.models.campaigns import (
    CampaignHeadRegistration,
    CampaignHeadResponse,
)

router = APIRouter()


@router.post("/campaigns/heads", response_model=CampaignHeadResponse)
async def register_heads(request: Request, body: CampaignHeadRegistration):
    store = request.app.state.store
    try:
        registered_ids = await store.register(
            [h.model_dump() for h in body.heads]
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return CampaignHeadResponse(
        registered=len(registered_ids),
        ids=registered_ids,
    )
