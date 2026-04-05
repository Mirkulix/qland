"""Google Reverse Image Search via SerpAPI."""

import logging
from urllib.parse import urlparse

import httpx

from image_search.config import Config
from image_search.engines.base import BaseSearchEngine
from image_search.models import SearchResult
from image_search.utils.image_utils import image_to_base64

logger = logging.getLogger(__name__)


class GoogleSearchEngine(BaseSearchEngine):
    """Google reverse image search using SerpAPI.

    Requires SERPAPI_KEY environment variable.
    SerpAPI provides a clean API wrapper around Google's reverse image search.
    Sign up at https://serpapi.com/
    """

    name = "google"

    def __init__(self, config: Config):
        self.api_key = config.serpapi_key
        self.timeout = config.search_timeout
        self.max_results = config.max_results_per_engine

    def is_available(self) -> bool:
        return bool(self.api_key)

    async def search(self, image_bytes: bytes, max_results: int = 20) -> list[SearchResult]:
        if not self.is_available():
            return []

        results = []
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Upload image and search via SerpAPI
                b64 = image_to_base64(image_bytes)
                resp = await client.post(
                    "https://serpapi.com/search.json",
                    data={
                        "engine": "google_reverse_image",
                        "image_content": b64,
                        "api_key": self.api_key,
                    },
                )
                resp.raise_for_status()
                data = resp.json()

                # Parse inline images
                for item in data.get("image_results", [])[:max_results]:
                    domain = urlparse(item.get("link", "")).netloc
                    results.append(
                        SearchResult(
                            title=item.get("title", ""),
                            url=item.get("link", ""),
                            image_url=item.get("original", item.get("thumbnail", "")),
                            thumbnail_url=item.get("thumbnail", ""),
                            source_engine=self.name,
                            domain=domain,
                        )
                    )

                # Parse visually similar images
                for item in data.get("visual_matches", [])[:max_results]:
                    domain = urlparse(item.get("link", "")).netloc
                    results.append(
                        SearchResult(
                            title=item.get("title", ""),
                            url=item.get("link", ""),
                            image_url=item.get("thumbnail", ""),
                            thumbnail_url=item.get("thumbnail", ""),
                            source_engine=self.name,
                            similarity_score=item.get("similarity", 0.0),
                            domain=domain,
                        )
                    )

        except Exception as e:
            logger.error(f"Google search error: {e}")
            raise

        return results[:max_results]
