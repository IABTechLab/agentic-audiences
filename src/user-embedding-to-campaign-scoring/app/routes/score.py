from __future__ import annotations

import uuid

import numpy as np
from fastapi import APIRouter, HTTPException, Request

from app.models.ortb import ScoreRequest
from app.models.scoring import CampaignScore, ScoreResponse
from app.engine.scorer import score_embedding

router = APIRouter()

DEFAULT_TOP_K = 10


@router.post("/score", response_model=list[ScoreResponse])
async def score(request: Request, body: ScoreRequest):
    config = request.app.state.config
    store = request.app.state.store
    tracker = request.app.state.tracker
    top_k = body.top_k or DEFAULT_TOP_K
    request_id = body.id or str(uuid.uuid4())

    embeddings_found: list[dict] = []
    for data_block in body.user.data:
        for segment in data_block.segment:
            if segment.ext is None:
                continue
            ext = segment.ext
            model_cfg = store.get_model_config(ext.model)
            metric = model_cfg.metric if model_cfg else "cosine"
            apply_l2_norm = model_cfg.apply_l2_norm if model_cfg else False
            embeddings_found.append({
                "vector": ext.vector,
                "model": ext.model,
                "type": ext.type,
                "metric": metric,
                "apply_l2_norm": apply_l2_norm,
            })

    if not embeddings_found:
        raise HTTPException(status_code=400, detail="No embeddings found in request")

    responses: list[ScoreResponse] = []
    for emb_info in embeddings_found:
        emb_type = config.resolve_type(emb_info["type"])
        head_ids, head_weights = await store.get_heads(
            emb_info["model"], emb_info["type"]
        )
        if not head_ids:
            responses.append(
                ScoreResponse(
                    request_id=request_id,
                    scores=[],
                    model=emb_info["model"],
                    embedding_type=emb_type,
                    metric=emb_info["metric"],
                )
            )
            continue

        embedding = np.array(emb_info["vector"], dtype=np.float32)
        scored = score_embedding(
            embedding=embedding,
            head_ids=head_ids,
            head_weights=head_weights,
            metric=emb_info["metric"],
            apply_l2_norm=emb_info["apply_l2_norm"],
            top_k=top_k,
        )

        for s in scored:
            await tracker.record(
                campaign_head_id=s.campaign_head_id,
                campaign_id=s.campaign_id,
                embedding=embedding,
                score=s.score,
            )

        responses.append(
            ScoreResponse(
                request_id=request_id,
                scores=[
                    CampaignScore(
                        campaign_id=s.campaign_id,
                        campaign_head_id=s.campaign_head_id,
                        score=round(s.score, 6),
                    )
                    for s in scored
                ],
                model=emb_info["model"],
                embedding_type=emb_type,
                metric=emb_info["metric"],
            )
        )

    return responses
