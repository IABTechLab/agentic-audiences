"""Contextual Signal Encoder — reference implementation for generating
ORTB-compatible contextual embeddings for the Agentic Audiences protocol.

This service fills the "left side" of the embedding pipeline: it takes raw
content (text, URLs, images, video) and produces contextual embedding vectors
in the wire format consumed by the Agentic Audiences scoring service.

Providers:
    - sentence_transformers (default): Open-source, runs locally, text only.
    - mixpeek: Production multimodal (text + image + video) with IAB taxonomy
      enrichment. Requires MIXPEEK_API_KEY.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import load_config
from app.engine.encoder import ContextualEncoder
from app.engine.multi_encoder import MultiModelEncoder
from app.routes.encode import router as encode_router, set_encoder, set_multi_encoder

_provider = None
_multi_encoder = None


def _create_provider(config):
    if config.encoder.provider == "mixpeek":
        from app.providers.mixpeek import MixpeekProvider
        return MixpeekProvider(config.mixpeek, model_name="mixpeek-multimodal")
    else:
        from app.providers.sentence_transformers import SentenceTransformersProvider
        return SentenceTransformersProvider(config.encoder)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _provider, _multi_encoder
    config = load_config()
    _provider = _create_provider(config)
    encoder = ContextualEncoder(config, _provider)
    _multi_encoder = MultiModelEncoder(config)
    set_encoder(encoder)
    set_multi_encoder(_multi_encoder)
    yield
    if _provider:
        await _provider.close()
    if _multi_encoder:
        await _multi_encoder.close()


app = FastAPI(
    title="Agentic Audiences — Contextual Signal Encoder",
    description=(
        "Reference implementation for generating contextual embeddings "
        "compatible with the Agentic Audiences embedding exchange protocol."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(encode_router)


@app.get("/health")
async def health():
    config = load_config()
    return {
        "status": "ok",
        "provider": config.encoder.provider,
        "model": config.encoder.model_name,
    }
