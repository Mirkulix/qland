"""Base class for search engines."""

from abc import ABC, abstractmethod

from image_search.models import SearchResult


class BaseSearchEngine(ABC):
    """Abstract base class for reverse image search engines."""

    name: str = "base"

    @abstractmethod
    async def search(self, image_bytes: bytes, max_results: int = 20) -> list[SearchResult]:
        """Search for similar images. Returns list of SearchResult."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this engine is configured and available."""
        ...
