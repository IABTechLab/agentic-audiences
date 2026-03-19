from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import load_config
from app.engine.analytics import AnalyticsTracker
from app.engine.device import HAS_TORCH, HAS_GPU, DEVICE
from app.engine.store import CampaignHeadStore
from app.routes import analytics, campaigns, score

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config()
    app.state.config = config
    app.state.store = CampaignHeadStore(config=config)
    app.state.tracker = AnalyticsTracker(config=config.analytics)
    if HAS_TORCH:
        logger.info("Scoring device: %s (GPU: %s)", DEVICE, HAS_GPU)
    else:
        logger.info("Scoring device: numpy (torch not installed)")
    yield


app = FastAPI(
    title="Agentic Audiences Scoring Service",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(score.router)
app.include_router(campaigns.router)
app.include_router(analytics.router)


@app.get("/health")
async def health():
    if HAS_GPU:
        device = str(DEVICE)
    elif HAS_TORCH:
        device = "cpu"
    else:
        device = "numpy"
    return {"status": "ok", "device": device}
