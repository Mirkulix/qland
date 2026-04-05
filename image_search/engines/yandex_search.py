"""Yandex Reverse Image Search via web scraping."""

import logging
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from image_search.config import Config
from image_search.engines.base import BaseSearchEngine
from image_search.models import SearchResult
from image_search.utils.image_utils import image_to_base64

logger = logging.getLogger(__name__)


class YandexSearchEngine(BaseSearchEngine):
    """Yandex reverse image search.

    Uses Yandex's public image upload endpoint (no API key required).
    Note: May be rate-limited for heavy usage.
    """

    name = "yandex"
    UPLOAD_URL = "https://yandex.com/images/search"

    def __init__(self, config: Config):
        self.timeout = config.search_timeout

    def is_available(self) -> bool:
        return True  # No API key needed

    async def search(self, image_bytes: bytes, max_results: int = 20) -> list[SearchResult]:
        results = []
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=True,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    )
                },
            ) as client:
                # Upload image to Yandex
                resp = await client.post(
                    "https://yandex.com/images/search",
                    params={"rpt": "imageview", "format": "json"},
                    files={"upfile": ("image.jpg", image_bytes, "image/jpeg")},
                )

                if resp.status_code != 200:
                    logger.warning(f"Yandex returned status {resp.status_code}")
                    return []

                # Try JSON response first
                try:
                    data = resp.json()
                    for item in data.get("blocks", []):
                        if item.get("name") == "cbir-similar":
                            for img in item.get("items", [])[:max_results]:
                                domain = urlparse(img.get("url", "")).netloc
                                results.append(
                                    SearchResult(
                                        title=img.get("title", ""),
                                        url=img.get("url", ""),
                                        image_url=img.get("originalImage", {}).get(
                                            "url", ""
                                        ),
                                        thumbnail_url=img.get("thumb", {}).get(
                                            "url", ""
                                        ),
                                        source_engine=self.name,
                                        domain=domain,
                                    )
                                )
                except ValueError:
                    # Fallback: parse HTML
                    soup = BeautifulSoup(resp.text, "html.parser")
                    for item in soup.select(".serp-item")[:max_results]:
                        link = item.select_one("a")
                        img = item.select_one("img")
                        if link and img:
                            url = link.get("href", "")
                            domain = urlparse(url).netloc
                            results.append(
                                SearchResult(
                                    title=img.get("alt", ""),
                                    url=url,
                                    image_url=img.get("src", ""),
                                    thumbnail_url=img.get("src", ""),
                                    source_engine=self.name,
                                    domain=domain,
                                )
                            )

        except Exception as e:
            logger.error(f"Yandex search error: {e}")
            raise

        return results[:max_results]
