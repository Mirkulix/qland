"""Data models for Image Search Product."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class SearchEngine(str, Enum):
    GOOGLE = "google"
    BING = "bing"
    TINEYE = "tineye"
    YANDEX = "yandex"
    CLIP = "clip"


@dataclass
class SearchResult:
    """A single image search result."""

    title: str
    url: str  # Page URL where the image was found
    image_url: str  # Direct image URL
    thumbnail_url: str = ""
    source_engine: str = ""
    similarity_score: float = 0.0  # 0.0 - 1.0
    width: int = 0
    height: int = 0
    domain: str = ""


@dataclass
class SearchResponse:
    """Aggregated search response from all engines."""

    results: list[SearchResult] = field(default_factory=list)
    engines_used: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)
    total_results: int = 0
    query_image_hash: str = ""

    def add_results(self, engine: str, results: list[SearchResult]):
        self.engines_used.append(engine)
        self.results.extend(results)
        self.total_results = len(self.results)

    def add_error(self, engine: str, error: str):
        self.errors[engine] = error

    def sort_by_similarity(self):
        self.results.sort(key=lambda r: r.similarity_score, reverse=True)

    def deduplicate(self):
        """Remove duplicate results based on image URL."""
        seen = set()
        unique = []
        for r in self.results:
            if r.image_url not in seen:
                seen.add(r.image_url)
                unique.append(r)
        self.results = unique
        self.total_results = len(self.results)
