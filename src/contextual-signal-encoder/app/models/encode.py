"""Models for contextual signal encoding.

The core output is an ORTB-compatible embedding segment with a model descriptor
envelope — model name, version, embedding space, and metric — so the receiving
party knows exactly how to process the vector.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class EncodeRequest(BaseModel):
    text: str | None = Field(default=None, description="Raw text content.")
    url: str | None = Field(default=None, description="URL to fetch and extract text from.")


class ModelDescriptor(BaseModel):
    """Metadata envelope describing how the embedding was produced.

    This is the key interoperability primitive: the receiving party loads the
    same model (or a compatible one) using this descriptor to read the vector.
    """

    name: str = Field(description="Model identifier (e.g., 'all-MiniLM-L6-v2').")
    version: str = Field(description="Model version for reproducibility.")
    embedding_space_id: str = Field(
        description="URI identifying the embedding space (e.g., 'aa://spaces/contextual/en-v1'). "
        "Two vectors are comparable only within the same space.",
    )
    metric: str = Field(default="cosine", description="Similarity metric: cosine, dot, or l2.")


class EmbeddingExt(BaseModel):
    """Matches the scoring service's EmbeddingSegmentExt schema, extended
    with model descriptor for interoperability."""

    ver: str = "1.0"
    vector: list[float]
    model: str
    dimension: int
    type: str
    model_descriptor: ModelDescriptor | None = Field(
        default=None,
        description="Full model metadata envelope for cross-party interoperability.",
    )


class Segment(BaseModel):
    id: str
    name: str | None = None
    ext: EmbeddingExt


class ContextualData(BaseModel):
    id: str = "contextual-signal-encoder"
    name: str = "contextual-embeddings"
    segment: list[Segment]


class EncodeResponse(BaseModel):
    data: ContextualData
    source: str
    content_length: int
