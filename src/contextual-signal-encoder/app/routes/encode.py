"""Encode route — generate contextual embeddings from content."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.models.encode import EncodeRequest, EncodeResponse, MultiEncodeRequest, MultiEncodeResponse

router = APIRouter()

_encoder = None
_multi_encoder = None


def set_encoder(encoder):
    global _encoder
    _encoder = encoder


def set_multi_encoder(multi_encoder):
    global _multi_encoder
    _multi_encoder = multi_encoder


@router.post("/encode", response_model=EncodeResponse)
async def encode_content(request: EncodeRequest) -> EncodeResponse:
    """Generate an ORTB-compatible contextual embedding from content.

    Returns an embedding segment ready to be placed into an OpenRTB
    ``user.data[]`` array for scoring by the campaign scoring service.
    """
    if _encoder is None:
        raise HTTPException(status_code=503, detail="Encoder not initialized.")

    if not any([request.text, request.url]):
        raise HTTPException(status_code=422, detail="Either text or url is required.")

    try:
        return await _encoder.encode(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Encoding failed: {e}")


@router.post("/encode/multi", response_model=MultiEncodeResponse)
async def encode_multi_model(request: MultiEncodeRequest) -> MultiEncodeResponse:
    """Encode content through multiple models, producing named vectors.

    Each model produces a named vector with full model metadata (architecture,
    pooling, normalization, training domain, version). The response includes
    backwards-compatible ORTB segments and a storage example showing how
    named vectors map to a single point with indexed payload metadata.
    """
    if _multi_encoder is None:
        raise HTTPException(status_code=503, detail="Multi-model encoder not initialized.")

    if not request.models:
        raise HTTPException(status_code=422, detail="At least one model spec is required.")

    if not any([request.text, request.url, request.image_url, request.video_url]):
        raise HTTPException(status_code=422, detail="At least one content field is required.")

    try:
        return await _multi_encoder.encode(request)
    except NotImplementedError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multi-model encoding failed: {e}")
