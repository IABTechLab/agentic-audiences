"""Encode routes — generate contextual embeddings from content."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.models.encode import EncodeRequest, EncodeResponse, MultiEncodeRequest, MultiEncodeResponse

router = APIRouter()

# Injected at startup by main.py
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


@router.post("/encode/multi", response_model=MultiEncodeResponse)
async def encode_multi_model(request: MultiEncodeRequest) -> MultiEncodeResponse:
    """Encode content through multiple models, producing named vectors.

    Each model in the request produces a named vector with full model metadata.
    The response includes:

    - **named_vectors**: One per model, each with metadata (architecture, pooling,
      normalization, training domain, version) for routing and compatibility.
    - **ortb_data**: Backwards-compatible ORTB segments for the scoring service.
    - **storage_example**: Shows how named vectors map to a single storage point
      with multiple vectors and indexed payload metadata.

    This addresses the named vector / model permutation problem: instead of
    separate ORTB segments partitioned by string-based model:type keys, content
    is encoded once and stored as a single point with multiple named vectors,
    each independently queryable and filterable by model characteristics.
    """
    if _multi_encoder is None:
        raise HTTPException(status_code=503, detail="Multi-model encoder not initialized.")

    if not request.models:
        raise HTTPException(status_code=422, detail="At least one model spec is required.")

    if not any([request.text, request.url, request.image_url, request.video_url]):
        raise HTTPException(
            status_code=422,
            detail="At least one of text, url, image_url, or video_url is required.",
        )

    try:
        return await _multi_encoder.encode(request)
    except NotImplementedError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multi-model encoding failed: {e}")
