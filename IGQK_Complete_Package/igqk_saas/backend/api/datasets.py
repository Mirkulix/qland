"""
Datasets API endpoints
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter()


class Dataset(BaseModel):
    """Dataset model"""
    id: str
    name: str
    description: Optional[str] = None
    source: str  # huggingface, kaggle, custom
    # Unknown for live Hub results (the Hub search does not report them): None, never guessed.
    size: Optional[str] = None
    num_samples: Optional[int] = None
    task: Optional[str] = None  # classification, detection, etc.
    format: Optional[str] = None  # images, text, audio


# Standard machine learning benchmark catalog
KNOWN_DATASETS = {
    "mnist": {
        "id": "mnist",
        "name": "MNIST",
        "description": "Handwritten digits dataset (0-9)",
        "source": "torchvision",
        "size": "11 MB",
        "num_samples": 70000,
        "task": "classification",
        "format": "images",
        "train_size": 60000,
        "test_size": 10000,
        "image_size": "28x28",
        "channels": 1,
        "classes": 10,
    },
    "cifar10": {
        "id": "cifar10",
        "name": "CIFAR-10",
        "description": "10-class natural image classification",
        "source": "torchvision",
        "size": "163 MB",
        "num_samples": 60000,
        "task": "classification",
        "format": "images",
        "train_size": 50000,
        "test_size": 10000,
        "image_size": "32x32",
        "channels": 3,
        "classes": 10,
    },
    "fashion_mnist": {
        "id": "fashion_mnist",
        "name": "Fashion-MNIST",
        "description": "Zalando article image classification",
        "source": "torchvision",
        "size": "30 MB",
        "num_samples": 70000,
        "task": "classification",
        "format": "images",
        "train_size": 60000,
        "test_size": 10000,
        "image_size": "28x28",
        "channels": 1,
        "classes": 10,
    },
    "imdb": {
        "id": "imdb",
        "name": "IMDB Reviews",
        "description": "Binary sentiment classification on movie reviews",
        "source": "huggingface",
        "size": "80 MB",
        "num_samples": 50000,
        "task": "text-classification",
        "format": "text",
        "train_size": 25000,
        "test_size": 25000,
        "classes": 2,
    }
}


@router.get("/public", response_model=List[Dataset])
async def list_public_datasets(
    task: Optional[str] = None,
    source: Optional[str] = None,
    limit: int = 10
):
    """List public datasets with optional task or source filter"""
    results = list(KNOWN_DATASETS.values())

    if task:
        results = [d for d in results if d["task"] == task]
    if source:
        results = [d for d in results if d["source"] == source]

    return [
        Dataset(
            id=d["id"],
            name=d["name"],
            description=d["description"],
            source=d["source"],
            size=d["size"],
            num_samples=d["num_samples"],
            task=d["task"],
            format=d["format"]
        )
        for d in results[:limit]
    ]


@router.get("/search/huggingface", response_model=List[Dataset])
async def search_huggingface_datasets(query: str, limit: int = 10):
    """Searches the HuggingFace Hub (live, not the built-in catalog of `/public`)."""
    from huggingface_hub import list_datasets
    try:
        found = list(list_datasets(search=query, limit=max(1, min(limit, 50))))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"HuggingFace Hub search failed: {e}")
    return [
        Dataset(
            id=d.id,
            name=d.id.split("/")[-1],
            description=(getattr(d, "description", None) or "")[:300] or None,
            source="huggingface",
            size=None,
            num_samples=None,
            task=next((t.split(":", 1)[1] for t in (getattr(d, "tags", None) or []) if t.startswith("task_categories:")), None),
            format=None,
        )
        for d in found
    ]


@router.get("/{dataset_id}")
async def get_dataset_info(dataset_id: str):
    """Get detailed dataset information"""
    dataset_id_clean = dataset_id.lower().replace("-", "").replace("_", "")
    for k, v in KNOWN_DATASETS.items():
        if k.replace("-", "").replace("_", "") == dataset_id_clean:
            return v

    raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found in registry")
