"""TinEye Reverse Image Search API integration."""

import logging
from urllib.parse import urlparse

import httpx

from image_search.config import Config
from image_search.engines.base import BaseSearchEngine
from image_search.models import SearchResult

logger = logging.getLogger(__name__)


class TinEyeSearchEngine(BaseSearchEngine):
    """TinEye reverse image search API.

    Requires TINEYE_API_KEY environment variable.
    Get your key at https://tineye.com/developer
    """

    name = "tineye"
    API_URL = "https://api.tineye.com/rest/search/"

    def __init__(self, config: Config):
        self.api_key = config.tineye_api_key
        self.timeout = config.search_timeout

    def is_available(self) -> bool:
        return bool(self.api_key)

    async def search(self, image_bytes: bytes, max_results: int = 20) -> list[SearchResult]:
        if not self.is_available():
            return []

        results = []
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    self.API_URL,
                    headers={"x-api-key": self.api_key},
                    files={"image": ("image.jpg", image_bytes, "image/jpeg")},
                    data={"limit": str(max_results)},
                )
                resp.raise_for_status()
                data = resp.json()

                for match in data.get("result", {}).get("matches", []):
                    for backlink in match.get("backlinks", []):
                        domain = urlparse(backlink.get("url", "")).netloc
                        results.append(
                            SearchResult(
                                title=backlink.get("url", ""),
                                url=backlink.get("url", ""),
                                image_url=match.get("image_url", ""),
                                thumbnail_url=match.get("image_url", ""),
                                source_engine=self.name,
                                similarity_score=match.get("score", 0.0) / 100.0,
                                width=match.get("width", 0),
                                height=match.get("height", 0),
                                domain=domain,
                            )
                        )

        except Exception as e:
            logger.error(f"TinEye search error: {e}")
            raise

        return results[:max_results]
