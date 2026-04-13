"""Encode routes — generate contextual embeddings from content."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.models.encode import EncodeRequest, EncodeResponse

router = APIRouter()

# Injected at startup by main.py
_encoder = None


def set_encoder(encoder):
    global _encoder
    _encoder = encoder


@router.post("/encode", response_model=EncodeResponse)
async def encode_content(request: EncodeRequest) -> EncodeResponse:
    """Generate an ORTB-compatible contextual embedding from content.

    Accepts text, a URL to scrape, an image URL, or a video URL.
    Returns an embedding segment ready to be placed into an OpenRTB
    ``user.data[]`` array for scoring by the campaign scoring service.
    """
    if _encoder is None:
        raise HTTPException(status_code=503, detail="Encoder not initialized.")

    if not any([request.text, request.url, request.image_url, request.video_url]):
        raise HTTPException(
            status_code=422,
            detail="At least one of text, url, image_url, or video_url is required.",
        )

    try:
        return await _encoder.encode(request)
    except NotImplementedError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Encoding failed: {e}")


@router.post("/encode/batch", response_model=list[EncodeResponse])
async def encode_batch(requests: list[EncodeRequest]) -> list[EncodeResponse]:
    """Encode multiple content items in a single request."""
    if _encoder is None:
        raise HTTPException(status_code=503, detail="Encoder not initialized.")

    results = []
    for req in requests:
        try:
            result = await _encoder.encode(req)
            results.append(result)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Encoding failed for item {len(results)}: {e}",
            )
    return results
