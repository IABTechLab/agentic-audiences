from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class EmbeddingType(str, Enum):
    identity = "identity"
    contextual = "contextual"
    reinforcement = "reinforcement"
    capi = "capi"
    intent = "intent"


class EmbeddingSegmentExt(BaseModel):
    ver: str = "1.0"
    vector: list[float]
    model: str
    dimension: int
    type: str  # raw wire type — resolved via config aliases


class Segment(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    value: Optional[str] = None
    ext: Optional[EmbeddingSegmentExt] = None


class Data(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    segment: list[Segment] = Field(default_factory=list)


class User(BaseModel):
    id: Optional[str] = None
    data: list[Data] = Field(default_factory=list)


class ScoreRequest(BaseModel):
    id: Optional[str] = None
    user: User
    top_k: Optional[int] = None
