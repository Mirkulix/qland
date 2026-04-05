"""CLIP-based local image similarity search.

Uses OpenAI's CLIP model to compute image embeddings and find similar images
in a local index. This enables offline similarity search without API keys.
"""

import json
import logging
import os
from pathlib import Path

import numpy as np
from PIL import Image

from image_search.config import Config
from image_search.engines.base import BaseSearchEngine
from image_search.models import SearchResult

logger = logging.getLogger(__name__)


class CLIPSearchEngine(BaseSearchEngine):
    """Local CLIP-based image similarity search.

    Maintains an index of image embeddings for fast similarity lookup.
    Can be used standalone or to re-rank results from other engines.
    """

    name = "clip"

    def __init__(self, config: Config):
        self.model_name = config.clip_model
        self.index_dir = config.clip_index_dir
        self._model = None
        self._preprocess = None
        self._index: dict[str, np.ndarray] = {}
        self._metadata: dict[str, dict] = {}

    def is_available(self) -> bool:
        try:
            import torch
            import clip
            return True
        except ImportError:
            logger.warning("CLIP not available. Install with: pip install git+https://github.com/openai/CLIP.git")
            return False

    def _load_model(self):
        if self._model is not None:
            return

        import torch
        import clip

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model, self._preprocess = clip.load(self.model_name, device=device)
        self._device = device
        logger.info(f"CLIP model {self.model_name} loaded on {device}")

    def compute_embedding(self, image_bytes: bytes) -> np.ndarray:
        """Compute CLIP embedding for an image."""
        import io
        import torch
        import clip

        self._load_model()

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_tensor = self._preprocess(img).unsqueeze(0).to(self._device)

        with torch.no_grad():
            features = self._model.encode_image(img_tensor)
            features = features / features.norm(dim=-1, keepdim=True)

        return features.cpu().numpy().flatten()

    def add_to_index(self, image_id: str, image_bytes: bytes, metadata: dict = None):
        """Add an image to the local search index."""
        embedding = self.compute_embedding(image_bytes)
        self._index[image_id] = embedding
        self._metadata[image_id] = metadata or {}

    def save_index(self):
        """Save the index to disk."""
        os.makedirs(self.index_dir, exist_ok=True)
        index_path = Path(self.index_dir) / "embeddings.npz"
        meta_path = Path(self.index_dir) / "metadata.json"

        if self._index:
            ids = list(self._index.keys())
            embeddings = np.stack([self._index[i] for i in ids])
            np.savez(index_path, ids=np.array(ids), embeddings=embeddings)

        with open(meta_path, "w") as f:
            json.dump(self._metadata, f)

    def load_index(self):
        """Load the index from disk."""
        index_path = Path(self.index_dir) / "embeddings.npz"
        meta_path = Path(self.index_dir) / "metadata.json"

        if index_path.exists():
            data = np.load(index_path, allow_pickle=True)
            ids = data["ids"]
            embeddings = data["embeddings"]
            self._index = {str(id_): emb for id_, emb in zip(ids, embeddings)}

        if meta_path.exists():
            with open(meta_path) as f:
                self._metadata = json.load(f)

    async def search(self, image_bytes: bytes, max_results: int = 20) -> list[SearchResult]:
        """Find similar images in the local index."""
        if not self.is_available() or not self._index:
            return []

        query_embedding = self.compute_embedding(image_bytes)
        similarities = {}

        for image_id, embedding in self._index.items():
            similarity = float(np.dot(query_embedding, embedding))
            similarities[image_id] = similarity

        # Sort by similarity (cosine similarity, higher = more similar)
        sorted_ids = sorted(similarities, key=similarities.get, reverse=True)

        results = []
        for image_id in sorted_ids[:max_results]:
            meta = self._metadata.get(image_id, {})
            results.append(
                SearchResult(
                    title=meta.get("title", image_id),
                    url=meta.get("url", ""),
                    image_url=meta.get("image_url", ""),
                    thumbnail_url=meta.get("thumbnail_url", ""),
                    source_engine=self.name,
                    similarity_score=similarities[image_id],
                    domain=meta.get("domain", "local"),
                )
            )

        return results

    async def compute_similarity(self, image1_bytes: bytes, image2_bytes: bytes) -> float:
        """Compute similarity between two images (0.0 to 1.0)."""
        if not self.is_available():
            return 0.0

        emb1 = self.compute_embedding(image1_bytes)
        emb2 = self.compute_embedding(image2_bytes)
        return float(np.dot(emb1, emb2))
