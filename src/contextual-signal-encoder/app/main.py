"""Contextual Signal Encoder — generates ORTB-compatible contextual embeddings."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.engine.encoder import ContextualEncoder
from app.providers.sentence_transformers import SentenceTransformersProvider
from app.routes.encode import router as encode_router, set_encoder

_provider = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _provider
    _provider = SentenceTransformersProvider()
    set_encoder(ContextualEncoder(_provider))
    yield
    if _provider:
        await _provider.close()


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
