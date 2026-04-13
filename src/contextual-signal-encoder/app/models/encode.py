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


# ---------------------------------------------------------------------------
# Multi-model (named vectors) support
# ---------------------------------------------------------------------------

class ModelSpec(BaseModel):
    """Specification for a single embedding model in a multi-model request."""

    name: str = Field(description="Unique name for this vector (e.g., 'text-minilm', 'image-siglip').")
    provider: str = Field(
        default="sentence_transformers",
        description="Provider to use: 'sentence_transformers' or 'mixpeek'.",
    )
    model: str | None = Field(
        default=None,
        description="Model name override (e.g., 'all-MiniLM-L6-v2', 'all-mpnet-base-v2').",
    )
    modality: str = Field(
        default="text",
        description="Input modality: 'text', 'image', or 'video'.",
    )


class MultiEncodeRequest(BaseModel):
    """Encode content through multiple models, producing named vectors."""

    text: str | None = Field(default=None, description="Text content.")
    url: str | None = Field(default=None, description="URL to fetch text from.")
    image_url: str | None = Field(default=None, description="Image URL.")
    video_url: str | None = Field(default=None, description="Video URL.")
    models: list[ModelSpec] = Field(
        description="List of models to encode with. Each produces a named vector.",
    )


class NamedVector(BaseModel):
    """A single named embedding vector with full model metadata."""

    name: str = Field(description="Vector name (e.g., 'text-minilm').")
    vector: list[float]
    model: str = Field(description="Model that produced this vector.")
    dimension: int
    modality: str = Field(description="Input modality used: text, image, or video.")
    provider: str = Field(description="Provider used.")
    metadata: ModelMetadata


class ModelMetadata(BaseModel):
    """Model characteristics attached to each named vector for routing and compatibility."""

    embedding_type: str = Field(default="context", description="Wire type for scoring service.")
    architecture: str | None = Field(default=None, description="Model architecture (e.g., 'transformer', 'siglip').")
    pooling: str | None = Field(default=None, description="Pooling strategy (mean, cls, max).")
    normalization: str = Field(default="l2_unit", description="Vector normalization (none, l2_unit).")
    training_domain: list[str] = Field(default_factory=list, description="Training domains (e.g., 'general', 'adtech').")
    version: str | None = Field(default=None, description="Model version string.")


class MultiEncodeResponse(BaseModel):
    """Multi-model encoding result with named vectors and ORTB output."""

    named_vectors: list[NamedVector] = Field(
        description="One named vector per model, each with full metadata.",
    )
    ortb_data: ContextualData = Field(
        description="ORTB-compatible data entry with all vectors as segments.",
    )
    storage_example: dict = Field(
        description="Example of how named vectors map to storage (point with multiple named vectors + payload).",
    )
