"""Content extraction from URLs."""

from __future__ import annotations

import httpx
from bs4 import BeautifulSoup


async def extract_text_from_url(url: str, timeout: int = 10) -> str:
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.get(
            url, headers={"User-Agent": "AATech-ContextualEncoder/1.0"}, follow_redirects=True,
        )
        resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    main = soup.find("article") or soup.find("main") or soup.find("body")
    return (main or soup).get_text(separator=" ", strip=True)
