"""Request and response models for the contextual signal encoder."""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Encode request
# ---------------------------------------------------------------------------

class EncodeRequest(BaseModel):
    """Content to encode into a contextual embedding."""

    text: str | None = Field(
        default=None,
        description="Raw text content to encode.",
    )
    url: str | None = Field(
        default=None,
        description="URL to fetch and extract text from.",
    )
    image_url: str | None = Field(
        default=None,
        description="Image URL for multimodal encoding (requires Mixpeek provider).",
    )
    video_url: str | None = Field(
        default=None,
        description="Video URL for multimodal encoding (requires Mixpeek provider).",
    )
    provider: str | None = Field(
        default=None,
        description="Override the configured provider for this request.",
    )


# ---------------------------------------------------------------------------
# ORTB-compatible embedding output
# ---------------------------------------------------------------------------

class EmbeddingExt(BaseModel):
    """Matches the scoring service's EmbeddingSegmentExt schema."""

    ver: str = "1.0"
    vector: list[float]
    model: str
    dimension: int
    type: str


class Segment(BaseModel):
    id: str
    name: str | None = None
    ext: EmbeddingExt


class ContextualData(BaseModel):
    """A single data entry ready to be placed into an ORTB user.data array."""

    id: str = "contextual-signal-encoder"
    name: str = "contextual-embeddings"
    segment: list[Segment]


# ---------------------------------------------------------------------------
# Encode response
# ---------------------------------------------------------------------------

class EncodeResponse(BaseModel):
    """Full encode result with ORTB-ready output and optional metadata."""

    data: ContextualData
    metadata: EncodeMetadata | None = None


class EncodeMetadata(BaseModel):
    """Optional enrichment metadata returned alongside the embedding."""

    source: str = Field(description="Input source: 'text', 'url', 'image_url', 'video_url'.")
    provider: str = Field(description="Provider used: 'sentence_transformers' or 'mixpeek'.")
    content_length: int = Field(description="Character count of the text that was encoded.")
    iab_categories: list[IABCategory] | None = Field(
        default=None,
        description="IAB taxonomy categories (when available from Mixpeek provider).",
    )


class IABCategory(BaseModel):
    """An IAB content taxonomy classification."""

    name: str
    tier1: str
    path: str
    score: float = Field(ge=0.0, le=1.0)
