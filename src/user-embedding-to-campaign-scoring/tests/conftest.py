from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import AppConfig, AnalyticsConfig
from app.engine.analytics import AnalyticsTracker
from app.engine.store import CampaignHeadStore
from app.main import app


def _test_config() -> AppConfig:
    return AppConfig(
        embedding_types=["identity", "contextual", "reinforcement", "capi", "intent"],
        type_aliases={"context": "contextual", "creative": "capi"},
        analytics=AnalyticsConfig(
            pca_dimensions=3,
            score_buckets=[0, 25, 50, 75, 100],
            max_embeddings_stored=100,
        ),
    )


@pytest.fixture
def test_config() -> AppConfig:
    return _test_config()


@pytest.fixture
def client(test_config: AppConfig) -> TestClient:
    app.state.config = test_config
    app.state.store = CampaignHeadStore(config=test_config)
    app.state.tracker = AnalyticsTracker(config=test_config.analytics)
    return TestClient(app, raise_server_exceptions=True)
