"""Bing Visual Search API integration."""

import logging
from urllib.parse import urlparse

import httpx

from image_search.config import Config
from image_search.engines.base import BaseSearchEngine
from image_search.models import SearchResult

logger = logging.getLogger(__name__)


class BingSearchEngine(BaseSearchEngine):
    """Bing Visual Search API.

    Requires BING_API_KEY environment variable.
    Get your key at https://www.microsoft.com/en-us/bing/apis/bing-visual-search-api
    """

    name = "bing"
    API_URL = "https://api.bing.microsoft.com/v7.0/images/visualsearch"

    def __init__(self, config: Config):
        self.api_key = config.bing_api_key
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
                    headers={"Ocp-Apim-Subscription-Key": self.api_key},
                    files={"image": ("image.jpg", image_bytes, "image/jpeg")},
                )
                resp.raise_for_status()
                data = resp.json()

                tags = data.get("tags", [])
                for tag in tags:
                    for action in tag.get("actions", []):
                        action_type = action.get("actionType", "")
                        if action_type not in (
                            "VisualSearch",
                            "PagesIncluding",
                            "SimilarImages",
                        ):
                            continue

                        for item in action.get("data", {}).get("value", []):
                            domain = urlparse(
                                item.get("hostPageUrl", item.get("contentUrl", ""))
                            ).netloc
                            results.append(
                                SearchResult(
                                    title=item.get("name", ""),
                                    url=item.get(
                                        "hostPageUrl", item.get("contentUrl", "")
                                    ),
                                    image_url=item.get("contentUrl", ""),
                                    thumbnail_url=item.get("thumbnailUrl", ""),
                                    source_engine=self.name,
                                    width=item.get("width", 0),
                                    height=item.get("height", 0),
                                    domain=domain,
                                )
                            )

        except Exception as e:
            logger.error(f"Bing search error: {e}")
            raise

        return results[:max_results]
