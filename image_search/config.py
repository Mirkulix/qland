"""Configuration for Image Search Product."""

import os
from dataclasses import dataclass, field


@dataclass
class Config:
    """Application configuration loaded from environment variables."""

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # Upload
    upload_dir: str = "uploads"
    max_file_size_mb: int = 20
    allowed_extensions: set = field(
        default_factory=lambda: {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
    )

    # API Keys (optional - enables respective engines)
    google_api_key: str = ""
    google_cx: str = ""  # Custom Search Engine ID
    bing_api_key: str = ""
    serpapi_key: str = ""  # SerpAPI for Google reverse image search
    tineye_api_key: str = ""

    # CLIP local search
    clip_model: str = "ViT-B/32"
    clip_index_dir: str = "clip_index"

    # Search settings
    max_results_per_engine: int = 20
    search_timeout: int = 30  # seconds

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            host=os.getenv("IMAGE_SEARCH_HOST", "0.0.0.0"),
            port=int(os.getenv("IMAGE_SEARCH_PORT", "8000")),
            debug=os.getenv("IMAGE_SEARCH_DEBUG", "false").lower() == "true",
            upload_dir=os.getenv("IMAGE_SEARCH_UPLOAD_DIR", "uploads"),
            max_file_size_mb=int(os.getenv("IMAGE_SEARCH_MAX_FILE_SIZE_MB", "20")),
            google_api_key=os.getenv("GOOGLE_API_KEY", ""),
            google_cx=os.getenv("GOOGLE_CX", ""),
            bing_api_key=os.getenv("BING_API_KEY", ""),
            serpapi_key=os.getenv("SERPAPI_KEY", ""),
            tineye_api_key=os.getenv("TINEYE_API_KEY", ""),
            clip_model=os.getenv("CLIP_MODEL", "ViT-B/32"),
            clip_index_dir=os.getenv("CLIP_INDEX_DIR", "clip_index"),
            max_results_per_engine=int(
                os.getenv("IMAGE_SEARCH_MAX_RESULTS", "20")
            ),
            search_timeout=int(os.getenv("IMAGE_SEARCH_TIMEOUT", "30")),
        )
