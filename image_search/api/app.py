"""FastAPI application for Image Search Product."""

import logging
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from image_search.config import Config
from image_search.engines.search_orchestrator import SearchOrchestrator
from image_search.utils.image_utils import (
    get_image_hash,
    resize_for_upload,
    save_upload,
    validate_image,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = Config.from_env()
orchestrator = SearchOrchestrator(config)

app = FastAPI(
    title="Image Search",
    description="Reverse Image Search - Find images on the web",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
UI_DIR = Path(__file__).parent.parent / "ui"
if (UI_DIR / "static").exists():
    app.mount("/static", StaticFiles(directory=str(UI_DIR / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main web UI."""
    template_path = UI_DIR / "templates" / "index.html"
    if template_path.exists():
        return template_path.read_text()
    return "<h1>Image Search API</h1><p>Upload an image to /api/search</p>"


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "available_engines": orchestrator.get_available_engines(),
    }


@app.get("/api/engines")
async def list_engines():
    """List all search engines and their availability."""
    return {
        name.value: {
            "available": engine.is_available(),
            "name": engine.name,
        }
        for name, engine in orchestrator.engines.items()
    }


@app.post("/api/search")
async def search_image(
    image: UploadFile = File(...),
    engines: str = Query(
        default=None,
        description="Comma-separated engine names (google,bing,tineye,yandex,clip). Default: all available.",
    ),
    max_results: int = Query(default=20, ge=1, le=100),
):
    """Upload an image and search for it across the web.

    Returns matching images found on various search engines.
    """
    # Read and validate image
    file_bytes = await image.read()

    if len(file_bytes) > config.max_file_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {config.max_file_size_mb}MB",
        )

    if not validate_image(file_bytes, config.allowed_extensions):
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Save upload
    save_upload(file_bytes, config.upload_dir, image.filename or "upload.jpg")

    # Resize for search APIs
    search_bytes = resize_for_upload(file_bytes)

    # Parse engine selection
    engine_list = None
    if engines:
        engine_list = [e.strip() for e in engines.split(",")]

    # Run search
    response = await orchestrator.search(search_bytes, engine_list, max_results)

    return {
        "query_image_hash": response.query_image_hash,
        "total_results": response.total_results,
        "engines_used": response.engines_used,
        "errors": response.errors,
        "results": [asdict(r) for r in response.results],
    }


@app.post("/api/search/url")
async def search_by_url(
    url: str = Query(..., description="URL of the image to search for"),
    engines: str = Query(default=None),
    max_results: int = Query(default=20, ge=1, le=100),
):
    """Search for an image by its URL."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            file_bytes = resp.content
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not fetch image: {e}")

    if not validate_image(file_bytes, config.allowed_extensions):
        raise HTTPException(status_code=400, detail="URL does not point to a valid image")

    search_bytes = resize_for_upload(file_bytes)

    engine_list = None
    if engines:
        engine_list = [e.strip() for e in engines.split(",")]

    response = await orchestrator.search(search_bytes, engine_list, max_results)

    return {
        "query_image_hash": response.query_image_hash,
        "total_results": response.total_results,
        "engines_used": response.engines_used,
        "errors": response.errors,
        "results": [asdict(r) for r in response.results],
    }
