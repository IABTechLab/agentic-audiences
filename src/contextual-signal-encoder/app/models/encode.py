"""Request and response models for the contextual signal encoder."""

from __future__ import annotations

from pydantic import BaseModel, Field


class EncodeRequest(BaseModel):
    """Content to encode into a contextual embedding."""

    text: str | None = Field(default=None, description="Raw text content to encode.")
    url: str | None = Field(default=None, description="URL to fetch and extract text from.")


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


class EncodeResponse(BaseModel):
    """Encode result with ORTB-ready output."""

    data: ContextualData
    source: str = Field(description="Input source: 'text' or 'url'.")
    content_length: int = Field(description="Character count of the encoded text.")
