"""Search orchestrator - coordinates all search engines."""

import asyncio
import logging

from image_search.config import Config
from image_search.engines.bing_search import BingSearchEngine
from image_search.engines.clip_search import CLIPSearchEngine
from image_search.engines.google_search import GoogleSearchEngine
from image_search.engines.tineye_search import TinEyeSearchEngine
from image_search.engines.yandex_search import YandexSearchEngine
from image_search.models import SearchEngine, SearchResponse
from image_search.utils.image_utils import get_image_hash

logger = logging.getLogger(__name__)


class SearchOrchestrator:
    """Coordinates reverse image search across multiple engines."""

    def __init__(self, config: Config):
        self.config = config
        self.engines = {
            SearchEngine.GOOGLE: GoogleSearchEngine(config),
            SearchEngine.BING: BingSearchEngine(config),
            SearchEngine.TINEYE: TinEyeSearchEngine(config),
            SearchEngine.YANDEX: YandexSearchEngine(config),
            SearchEngine.CLIP: CLIPSearchEngine(config),
        }

    def get_available_engines(self) -> list[str]:
        """Return list of available engine names."""
        return [name.value for name, engine in self.engines.items() if engine.is_available()]

    async def search(
        self,
        image_bytes: bytes,
        engines: list[str] | None = None,
        max_results: int = 20,
    ) -> SearchResponse:
        """Run reverse image search across selected engines in parallel.

        Args:
            image_bytes: Raw image bytes to search for.
            engines: List of engine names to use. None = all available.
            max_results: Max results per engine.

        Returns:
            Aggregated SearchResponse with results from all engines.
        """
        response = SearchResponse(query_image_hash=get_image_hash(image_bytes))

        # Determine which engines to use
        if engines:
            selected = {
                SearchEngine(e): self.engines[SearchEngine(e)]
                for e in engines
                if SearchEngine(e) in self.engines
            }
        else:
            selected = {
                name: engine
                for name, engine in self.engines.items()
                if engine.is_available()
            }

        if not selected:
            logger.warning("No search engines available")
            return response

        # Run all searches in parallel
        async def _run_engine(name: SearchEngine, engine):
            try:
                results = await engine.search(image_bytes, max_results)
                return name.value, results, None
            except Exception as e:
                logger.error(f"Engine {name.value} failed: {e}")
                return name.value, [], str(e)

        tasks = [_run_engine(name, engine) for name, engine in selected.items()]
        completed = await asyncio.gather(*tasks)

        for engine_name, results, error in completed:
            if error:
                response.add_error(engine_name, error)
            if results:
                response.add_results(engine_name, results)

        response.deduplicate()
        response.sort_by_similarity()

        return response
