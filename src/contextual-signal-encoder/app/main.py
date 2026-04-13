"""Contextual Signal Encoder — generates ORTB-compatible contextual embeddings."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.engine.encoder import ContextualEncoder
from app.engine.multi_encoder import MultiModelEncoder
from app.providers.sentence_transformers import SentenceTransformersProvider
from app.routes.encode import router as encode_router, set_encoder, set_multi_encoder

_provider = None
_multi_encoder = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _provider, _multi_encoder
    _provider = SentenceTransformersProvider()
    _multi_encoder = MultiModelEncoder()
    set_encoder(ContextualEncoder(_provider))
    set_multi_encoder(_multi_encoder)
    yield
    if _provider:
        await _provider.close()
    if _multi_encoder:
        await _multi_encoder.close()


app = FastAPI(
    title="Agentic Audiences — Contextual Signal Encoder",
    description="Reference implementation for generating contextual embeddings "
    "compatible with the Agentic Audiences embedding exchange protocol.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(encode_router)


@app.get("/health")
async def health():
    return {"status": "ok", "provider": "sentence_transformers"}
