"""Encode route — generate contextual embeddings from content."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.models.encode import EncodeRequest, EncodeResponse

router = APIRouter()

_encoder = None


def set_encoder(encoder):
    global _encoder
    _encoder = encoder


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
