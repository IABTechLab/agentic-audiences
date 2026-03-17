"""Device detection for optional GPU acceleration."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    import torch

    HAS_TORCH = True
    HAS_GPU = torch.cuda.is_available()
    DEVICE = torch.device("cuda:0" if HAS_GPU else "cpu")
    logger.info("PyTorch available — device: %s", DEVICE)
except ImportError:
    HAS_TORCH = False
    HAS_GPU = False
    DEVICE = None
    logger.info("PyTorch not installed — using NumPy")
