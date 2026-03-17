from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


class AnalyticsConfig(BaseModel):
    pca_dimensions: int = 3
    score_buckets: list[int] = [0, 25, 50, 75, 100]
    max_embeddings_stored: int = 10000


class AppConfig(BaseModel):
    embedding_types: list[str] = [
        "identity", "contextual", "reinforcement", "capi", "intent",
    ]
    type_aliases: dict[str, str] = {}
    analytics: AnalyticsConfig = AnalyticsConfig()

    def resolve_type(self, raw_type: str) -> str:
        """Resolve a wire-format type to the canonical embedding type."""
        return self.type_aliases.get(raw_type, raw_type)


def load_config(path: str | Path | None = None) -> AppConfig:
    if path is None:
        path = os.environ.get(
            "CONFIG_PATH",
            str(Path(__file__).resolve().parent.parent / "config.yaml"),
        )
    path = Path(path)
    if not path.exists():
        return AppConfig()
    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}
    return AppConfig.model_validate(raw)
