"""Models for contextual signal encoding.

The output is an ORTB-compatible embedding segment using the model descriptor
fields defined in specs/v1.0/embedding_format.schema.json — this encoder is a
reference implementation that produces embeddings conforming to the existing spec.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class EncodeRequest(BaseModel):
    text: str | None = Field(default=None, description="Raw text content.")
    url: str | None = Field(default=None, description="URL to fetch and extract text from.")


class EmbeddingExt(BaseModel):
    """Matches the scoring service's EmbeddingSegmentExt schema.

    Fields align with specs/v1.0/embedding_format.schema.json:
    - model → model.id
    - dimension → model.dimension
    - type → signal type (context, identity, etc.)
    - version → model.version
    - embedding_space_id → model.embedding_space_id
    - metric → model.metric
    """

    ver: str = "1.0"
    vector: list[float]
    model: str
    dimension: int
    type: str
    version: str = Field(description="Model version (per embedding_format.schema.json).")
    embedding_space_id: str = Field(
        description="Embedding space URI (per embedding_format.schema.json). "
        "Two vectors are comparable only within the same space.",
    )
    metric: str = Field(default="cosine", description="Similarity metric: cosine, dot, or l2.")


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
