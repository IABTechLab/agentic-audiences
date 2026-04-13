"""Content extraction from URLs and raw text."""

from __future__ import annotations

import httpx
from bs4 import BeautifulSoup

from app.config import ExtractionConfig


async def extract_text_from_url(url: str, config: ExtractionConfig) -> str:
    """Fetch a URL and extract the main text content."""
    async with httpx.AsyncClient(timeout=config.timeout) as client:
        resp = await client.get(url, headers={"User-Agent": config.user_agent}, follow_redirects=True)
        resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove script, style, nav, footer, header elements
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    # Prefer article or main content
    main = soup.find("article") or soup.find("main") or soup.find("body")
    if main is None:
        return soup.get_text(separator=" ", strip=True)

    return main.get_text(separator=" ", strip=True)


def truncate_text(text: str, max_tokens: int) -> str:
    """Truncate text to approximate token limit (word-based heuristic)."""
    words = text.split()
    if len(words) <= max_tokens:
        return text
    return " ".join(words[:max_tokens])
