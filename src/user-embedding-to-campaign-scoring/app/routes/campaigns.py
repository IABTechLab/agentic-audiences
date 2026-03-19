from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.models.campaigns import (
    CampaignHeadDeleteResponse,
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


@router.put("/campaigns/heads", response_model=CampaignHeadResponse)
async def update_heads(request: Request, body: CampaignHeadRegistration):
    store = request.app.state.store
    try:
        updated_ids = await store.update(
            [h.model_dump() for h in body.heads]
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return CampaignHeadResponse(
        registered=len(updated_ids),
        ids=updated_ids,
    )


@router.delete("/campaigns/heads/{campaign_head_id}", response_model=CampaignHeadDeleteResponse)
async def delete_head(request: Request, campaign_head_id: str):
    store = request.app.state.store
    try:
        await store.delete(campaign_head_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return CampaignHeadDeleteResponse(deleted=campaign_head_id)
