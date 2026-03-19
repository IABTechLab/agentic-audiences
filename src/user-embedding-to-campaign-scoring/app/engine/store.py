from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Union

import numpy as np

from app.config import AppConfig
from app.engine.device import HAS_TORCH, DEVICE

if HAS_TORCH:
    import torch


@dataclass
class ModelConfig:
    """Model-level configuration captured from the first head registered for a model."""
    dimension: int
    metric: str
    apply_l2_norm: bool
    compatible_with: list[str]


@dataclass
class StoredHead:
    campaign_id: str
    campaign_head_id: str
    weights: np.ndarray  # float16


@dataclass
class CampaignHeadStore:
    config: AppConfig
    _heads: dict[str, list[StoredHead]] = field(default_factory=dict)
    _model_configs: dict[str, ModelConfig] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _cache: dict[str, tuple[list[tuple[str, str]], Union[np.ndarray, "torch.Tensor"]]] = field(default_factory=dict)
    _head_partition: dict[str, str] = field(default_factory=dict)  # campaign_head_id -> partition_key
    _model_head_count: dict[str, int] = field(default_factory=dict)  # model -> count of heads

    def _partition_key(self, model: str, embedding_type: str) -> str:
        return f"{model}:{embedding_type}"

    def get_model_config(self, model: str) -> ModelConfig | None:
        return self._model_configs.get(model)

    def is_compatible(self, model_a: str, model_b: str) -> bool:
        """Check whether two model names are compatible."""
        if model_a == model_b:
            return True
        cfg_a = self._model_configs.get(model_a)
        if cfg_a and model_b in cfg_a.compatible_with:
            return True
        cfg_b = self._model_configs.get(model_b)
        if cfg_b and model_a in cfg_b.compatible_with:
            return True
        return False

    def _upsert_one(self, h: dict) -> str:
        """Insert or replace a single head. Must be called under the lock."""
        model = h["model"]
        raw_type = h["type"]
        dimension = h["dimension"]
        weights = h["weights"]
        campaign_head_id = h["campaign_head_id"]

        if len(weights) != dimension:
            raise ValueError(
                f"Dimension mismatch for model {model}: "
                f"len(weights)={len(weights)}, declared dimension={dimension}"
            )

        # Register or validate model config
        existing = self._model_configs.get(model)
        if existing is not None:
            if existing.dimension != dimension:
                raise ValueError(
                    f"Dimension mismatch for model {model}: "
                    f"expected {existing.dimension}, got {dimension}"
                )
        else:
            self._model_configs[model] = ModelConfig(
                dimension=dimension,
                metric=h.get("metric", "cosine"),
                apply_l2_norm=h.get("apply_l2_norm", False),
                compatible_with=h.get("compatible_with", []),
            )

        # If head already exists, remove old entry
        if campaign_head_id in self._head_partition:
            old_key = self._head_partition[campaign_head_id]
            old_model = old_key.split(":", 1)[0]
            self._heads[old_key] = [
                s for s in self._heads[old_key] if s.campaign_head_id != campaign_head_id
            ]
            if not self._heads[old_key]:
                del self._heads[old_key]
            self._model_head_count[old_model] -= 1
            if self._model_head_count[old_model] == 0:
                del self._model_head_count[old_model]
                del self._model_configs[old_model]

        emb_type = self.config.resolve_type(raw_type)
        key = self._partition_key(model, emb_type)

        arr = np.array(weights, dtype=np.float16)
        stored = StoredHead(
            campaign_id=h["campaign_id"],
            campaign_head_id=campaign_head_id,
            weights=arr,
        )

        if key not in self._heads:
            self._heads[key] = []
        self._heads[key].append(stored)
        self._head_partition[campaign_head_id] = key
        self._model_head_count[model] = self._model_head_count.get(model, 0) + 1

        return campaign_head_id

    async def register(self, heads: list[dict]) -> list[str]:
        """Register campaign heads (upsert semantics). Each dict must have:
        campaign_id, campaign_head_id, weights, model, dimension, type.
        Optionally: metric, apply_l2_norm, compatible_with.

        Returns list of registered campaign_head_ids.
        """
        registered: list[str] = []
        async with self._lock:
            for h in heads:
                registered.append(self._upsert_one(h))
            self._cache.clear()
        return registered

    async def delete(self, campaign_head_id: str) -> None:
        """Delete a single campaign head by ID. Raises KeyError if not found."""
        async with self._lock:
            if campaign_head_id not in self._head_partition:
                raise KeyError(f"Campaign head '{campaign_head_id}' not found")

            key = self._head_partition[campaign_head_id]
            model = key.split(":", 1)[0]

            self._heads[key] = [
                s for s in self._heads[key] if s.campaign_head_id != campaign_head_id
            ]
            if not self._heads[key]:
                del self._heads[key]

            self._model_head_count[model] -= 1
            if self._model_head_count[model] == 0:
                del self._model_head_count[model]
                del self._model_configs[model]

            del self._head_partition[campaign_head_id]
            self._cache.clear()

    async def update(self, heads: list[dict]) -> list[str]:
        """Update existing campaign heads. Raises KeyError if any head doesn't exist.
        Atomic: no heads are updated if any are missing.
        """
        async with self._lock:
            # Pre-check: all must exist
            for h in heads:
                if h["campaign_head_id"] not in self._head_partition:
                    raise KeyError(f"Campaign head '{h['campaign_head_id']}' not found")

            updated: list[str] = []
            for h in heads:
                updated.append(self._upsert_one(h))
            self._cache.clear()
        return updated

    async def get_heads(
        self, model: str, embedding_type: str
    ) -> tuple[list[tuple[str, str]], Union[np.ndarray, "torch.Tensor"]]:
        """Return (head_ids, weight_matrix) for a given model+type.

        Also includes heads from compatible models.
        Weight matrix is a torch.Tensor on DEVICE when torch is available,
        otherwise a NumPy array.
        """
        emb_type = self.config.resolve_type(embedding_type)
        cache_key = self._partition_key(model, emb_type)

        if cache_key in self._cache:
            return self._cache[cache_key]

        matching: list[StoredHead] = []

        async with self._lock:
            for key, stored_heads in self._heads.items():
                stored_model, stored_type = key.split(":", 1)
                if stored_type != emb_type:
                    continue
                if not self.is_compatible(model, stored_model):
                    continue
                matching.extend(stored_heads)

        if not matching:
            return [], np.empty((0, 0), dtype=np.float16)

        head_ids = [(h.campaign_id, h.campaign_head_id) for h in matching]
        np_matrix = np.stack([h.weights for h in matching])

        if HAS_TORCH:
            weight_matrix = torch.tensor(np_matrix, dtype=torch.float16, device=DEVICE)
        else:
            weight_matrix = np_matrix

        result = (head_ids, weight_matrix)
        self._cache[cache_key] = result
        return result
