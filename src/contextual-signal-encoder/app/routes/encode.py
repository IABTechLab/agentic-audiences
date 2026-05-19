"""Encode route."""

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
    """Generate an ORTB-compatible contextual embedding with model descriptor.

    The response includes a model_descriptor envelope so the receiving party
    knows the model name, version, embedding space, and metric needed to
    process the vector.
    """
    if _encoder is None:
        raise HTTPException(status_code=503, detail="Encoder not initialized.")
    if not any([request.text, request.url]):
        raise HTTPException(status_code=422, detail="Either text or url is required.")
    try:
        return await _encoder.encode(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Encoding failed: {e}")
