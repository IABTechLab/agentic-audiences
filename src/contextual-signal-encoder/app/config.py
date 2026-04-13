"""Configuration loader for the contextual signal encoder."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class EncoderConfig:
    provider: str = "sentence_transformers"
    model_name: str = "all-MiniLM-L6-v2"
    dimension: int = 384
    embedding_type: str = "context"
    max_tokens: int = 512


@dataclass
class MixpeekConfig:
    base_url: str = "https://api.mixpeek.com"
    api_key: str = ""
    namespace: str = ""
    retriever_id: str = ""


@dataclass
class ExtractionConfig:
    user_agent: str = "AATech-ContextualEncoder/1.0"
    timeout: int = 10


@dataclass
class AppConfig:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    mixpeek: MixpeekConfig = field(default_factory=MixpeekConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)


def load_config() -> AppConfig:
    """Load config from YAML file, with env var overrides."""
    config_path = Path(os.getenv("CONFIG_PATH", "config.yaml"))
    cfg = AppConfig()

    if config_path.exists():
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}

        if "encoder" in raw:
            for k, v in raw["encoder"].items():
                if hasattr(cfg.encoder, k):
                    setattr(cfg.encoder, k, v)

        if "mixpeek" in raw:
            for k, v in raw["mixpeek"].items():
                if hasattr(cfg.mixpeek, k):
                    setattr(cfg.mixpeek, k, v)

        if "extraction" in raw:
            for k, v in raw["extraction"].items():
                if hasattr(cfg.extraction, k):
                    setattr(cfg.extraction, k, v)

    # Env var overrides
    cfg.encoder.provider = os.getenv("ENCODER_PROVIDER", cfg.encoder.provider)
    cfg.encoder.model_name = os.getenv("ENCODER_MODEL", cfg.encoder.model_name)
    cfg.mixpeek.api_key = os.getenv("MIXPEEK_API_KEY", cfg.mixpeek.api_key)
    cfg.mixpeek.base_url = os.getenv("MIXPEEK_BASE_URL", cfg.mixpeek.base_url)
    cfg.mixpeek.namespace = os.getenv("MIXPEEK_NAMESPACE", cfg.mixpeek.namespace)
    cfg.mixpeek.retriever_id = os.getenv("MIXPEEK_RETRIEVER_ID", cfg.mixpeek.retriever_id)

    return cfg
