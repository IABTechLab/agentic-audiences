"""Mixpeek provider — multimodal encoding with optional IAB enrichment.

Setup:
    export MIXPEEK_API_KEY=your_key
    export MIXPEEK_NAMESPACE=your_namespace
"""

from __future__ import annotations

import os

import httpx

from app.providers.base import EmbeddingProvider, EmbeddingResult

BASE_URL = os.getenv("MIXPEEK_BASE_URL", "https://api.mixpeek.com")
API_KEY = os.getenv("MIXPEEK_API_KEY", "")
NAMESPACE = os.getenv("MIXPEEK_NAMESPACE", "")


class MixpeekProvider(EmbeddingProvider):

    def __init__(self, model_name: str = "mixpeek-multimodal") -> None:
        self._model_name = model_name
        self._client = httpx.AsyncClient(
            base_url=BASE_URL.rstrip("/"),
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            timeout=30,
        )

    async def encode_text(self, text: str) -> EmbeddingResult:
        headers = {"X-Namespace": NAMESPACE} if NAMESPACE else {}
        resp = await self._client.post("/v1/embed", json={"input": text, "input_type": "text"}, headers=headers)
        resp.raise_for_status()
        vec = resp.json()["embedding"]
        return EmbeddingResult(vector=vec, model=self._model_name, dimension=len(vec))

    async def encode_image(self, image_url: str) -> EmbeddingResult:
        headers = {"X-Namespace": NAMESPACE} if NAMESPACE else {}
        resp = await self._client.post("/v1/embed", json={"input": image_url, "input_type": "image_url"}, headers=headers)
        resp.raise_for_status()
        vec = resp.json()["embedding"]
        return EmbeddingResult(vector=vec, model=self._model_name, dimension=len(vec))

    async def encode_video(self, video_url: str) -> EmbeddingResult:
        headers = {"X-Namespace": NAMESPACE} if NAMESPACE else {}
        resp = await self._client.post("/v1/embed", json={"input": video_url, "input_type": "video_url"}, headers=headers)
        resp.raise_for_status()
        vec = resp.json()["embedding"]
        return EmbeddingResult(vector=vec, model=self._model_name, dimension=len(vec))

    async def close(self) -> None:
        await self._client.aclose()
