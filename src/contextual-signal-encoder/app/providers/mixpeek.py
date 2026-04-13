"""Mixpeek provider — production multimodal encoding with IAB enrichment.

Mixpeek generates contextual embeddings from text, images, and video via its
retriever pipeline. When configured with an IAB taxonomy retriever, it also
returns IAB content category classifications alongside the embedding vector.

Setup:
    export MIXPEEK_API_KEY=your_key
    export MIXPEEK_NAMESPACE=your_namespace
    # Optional: export MIXPEEK_RETRIEVER_ID=ret_xxx (auto-discovered if omitted)
"""

from __future__ import annotations

import httpx

from app.config import MixpeekConfig
from app.providers.base import EmbeddingProvider, EmbeddingResult


class MixpeekProvider(EmbeddingProvider):
    """Multimodal contextual encoding via the Mixpeek API.

    Supports text, image, and video content. When an IAB retriever is
    configured, responses include taxonomy classifications as metadata.
    """

    def __init__(self, config: MixpeekConfig, model_name: str = "mixpeek-multimodal") -> None:
        self._config = config
        self._model_name = model_name
        self._client = httpx.AsyncClient(
            base_url=config.base_url.rstrip("/"),
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            },
            timeout=30,
        )
        self._retriever_id: str | None = config.retriever_id or None

    async def _discover_retriever(self) -> str | None:
        """Auto-discover an IAB text retriever in the configured namespace."""
        if self._retriever_id:
            return self._retriever_id

        headers = {}
        if self._config.namespace:
            headers["X-Namespace"] = self._config.namespace

        resp = await self._client.get("/v1/retrievers", headers=headers)
        if resp.status_code != 200:
            return None

        for r in resp.json():
            name = (r.get("retriever_name") or "").lower()
            if "iab" in name and "text" in name:
                self._retriever_id = r["retriever_id"]
                return self._retriever_id

        return None

    async def _execute_retriever(self, query: str) -> dict | None:
        """Execute a retriever search and return the raw response."""
        retriever_id = await self._discover_retriever()
        if not retriever_id:
            return None

        headers = {}
        if self._config.namespace:
            headers["X-Namespace"] = self._config.namespace

        payload = {"queries": [{"type": "text", "value": query, "limit": 10}]}
        resp = await self._client.post(
            f"/v1/retrievers/{retriever_id}/execute",
            json=payload,
            headers=headers,
        )
        if resp.status_code != 200:
            return None

        return resp.json()

    async def encode_text(self, text: str) -> EmbeddingResult:
        """Encode text via Mixpeek, with optional IAB taxonomy enrichment."""
        headers = {}
        if self._config.namespace:
            headers["X-Namespace"] = self._config.namespace

        # Generate embedding via the embed endpoint
        resp = await self._client.post(
            "/v1/embed",
            json={"input": text, "input_type": "text"},
            headers=headers,
        )
        resp.raise_for_status()
        embed_data = resp.json()
        vector = embed_data["embedding"]

        # Optionally enrich with IAB taxonomy via retriever
        iab_categories = None
        retriever_result = await self._execute_retriever(text)
        if retriever_result:
            iab_categories = self._extract_iab_categories(retriever_result)

        return EmbeddingResult(
            vector=vector,
            model=self._model_name,
            dimension=len(vector),
            iab_categories=iab_categories,
        )

    async def encode_image(self, image_url: str) -> EmbeddingResult:
        """Encode image content via Mixpeek."""
        headers = {}
        if self._config.namespace:
            headers["X-Namespace"] = self._config.namespace

        resp = await self._client.post(
            "/v1/embed",
            json={"input": image_url, "input_type": "image_url"},
            headers=headers,
        )
        resp.raise_for_status()
        embed_data = resp.json()

        return EmbeddingResult(
            vector=embed_data["embedding"],
            model=self._model_name,
            dimension=len(embed_data["embedding"]),
        )

    async def encode_video(self, video_url: str) -> EmbeddingResult:
        """Encode video content via Mixpeek."""
        headers = {}
        if self._config.namespace:
            headers["X-Namespace"] = self._config.namespace

        resp = await self._client.post(
            "/v1/embed",
            json={"input": video_url, "input_type": "video_url"},
            headers=headers,
        )
        resp.raise_for_status()
        embed_data = resp.json()

        return EmbeddingResult(
            vector=embed_data["embedding"],
            model=self._model_name,
            dimension=len(embed_data["embedding"]),
        )

    @staticmethod
    def _extract_iab_categories(retriever_result: dict) -> list[dict]:
        """Extract IAB categories from a retriever execution result."""
        categories = []
        results = retriever_result.get("results") or retriever_result.get("documents") or []
        for doc in results[:5]:
            payload = doc.get("payload") or doc
            cat = {
                "name": payload.get("iab_category_name", payload.get("name", "")),
                "tier1": payload.get("iab_tier1", ""),
                "path": payload.get("iab_path", ""),
                "score": doc.get("score", 0.0),
            }
            if cat["name"]:
                categories.append(cat)
        return categories

    async def close(self) -> None:
        await self._client.aclose()
