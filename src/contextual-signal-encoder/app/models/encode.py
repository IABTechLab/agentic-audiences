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


# ---------------------------------------------------------------------------
# Multi-model (named vectors)
# ---------------------------------------------------------------------------

class ModelSpec(BaseModel):
    """A single model to encode with."""

    name: str = Field(description="Vector name (e.g., 'text-minilm', 'image-siglip').")
    provider: str = Field(default="sentence_transformers")
    model: str | None = Field(default=None, description="Model name override.")
    modality: str = Field(default="text", description="'text', 'image', or 'video'.")


class MultiEncodeRequest(BaseModel):
    text: str | None = None
    url: str | None = None
    image_url: str | None = None
    video_url: str | None = None
    models: list[ModelSpec]


class ModelMetadata(BaseModel):
    """Model characteristics for routing and compatibility."""

    embedding_type: str = "context"
    architecture: str | None = None
    pooling: str | None = None
    normalization: str = "l2_unit"
    training_domain: list[str] = Field(default_factory=list)
    version: str | None = None


class NamedVector(BaseModel):
    name: str
    vector: list[float]
    model: str
    dimension: int
    modality: str
    provider: str
    metadata: ModelMetadata


class MultiEncodeResponse(BaseModel):
    named_vectors: list[NamedVector]
    ortb_data: ContextualData = Field(description="Backwards-compatible ORTB segments.")
    storage_example: dict = Field(description="How named vectors map to a single storage point.")
