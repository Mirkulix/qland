"""
Models API endpoints - Model Registry
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from services.huggingface_service import HuggingFaceService

router = APIRouter()

# Initialize HuggingFace service
hf_service = HuggingFaceService()


class Model(BaseModel):
    """Model registry entry"""
    id: str
    name: str
    description: Optional[str]
    architecture: str
    size_mb: float
    accuracy: Optional[float]
    task: str
    created_at: str
    source: str  # trained, compressed, huggingface


@router.get("/my-models", response_model=List[Model])
async def list_user_models(limit: int = 10):
    """List models owned by current user (dynamically discovered from disk)"""
    import os
    import time
    user_models = []

    search_dirs = [
        os.path.join(os.getcwd(), "compressed_models"),
        os.path.join(os.getcwd(), "models"),
        os.getcwd()
    ]
    seen_files = set()

    for sdir in search_dirs:
        if not os.path.exists(sdir):
            continue
        for fname in os.listdir(sdir):
            if fname.endswith(('.pt', '.pth', '.bin')) and fname not in seen_files:
                seen_files.add(fname)
                fpath = os.path.join(sdir, fname)
                try:
                    fsize = os.path.getsize(fpath) / (1024 * 1024)
                    mtime = time.strftime('%Y-%m-%d', time.localtime(os.path.getmtime(fpath)))
                    is_compressed = "compress" in fname.lower() or "ternary" in fname.lower()
                    user_models.append(Model(
                        id=fname,
                        name=fname.replace('.pt', '').replace('.pth', ''),
                        description=f"{'IGQK 16× Compressed' if is_compressed else 'PyTorch'} model on disk",
                        architecture="transformer" if "gpt" in fname.lower() or "bert" in fname.lower() else "custom_net",
                        size_mb=round(fsize, 2),
                        accuracy=None,
                        task="language-modeling" if "gpt" in fname.lower() or "bert" in fname.lower() else "classification",
                        created_at=mtime,
                        source="compressed" if is_compressed else "trained"
                    ))
                except Exception:
                    pass

    return user_models[:limit]


@router.get("/{model_id}")
async def get_model_info(model_id: str):
    """Get detailed model information from disk or registry"""
    import os
    search_dirs = [os.path.join(os.getcwd(), "compressed_models"), os.path.join(os.getcwd(), "models"), os.getcwd()]
    for sdir in search_dirs:
        p = os.path.join(sdir, model_id)
        if os.path.exists(p):
            sz = round(os.path.getsize(p) / (1024 * 1024), 2)
            return {
                "id": model_id,
                "name": model_id.replace('.pt', '').replace('.pth', ''),
                "architecture": "transformer" if "gpt" in model_id.lower() or "bert" in model_id.lower() else "custom_net",
                "size_mb": sz,
                "accuracy": None,
                "parameters": 0,
                "download_url": f"/api/models/download/{model_id}"
            }

    raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")


def find_model_path(model_id: str):
    """The path of a model file on disk, or None (the same places `list_user_models` searches)."""
    import os
    if os.path.basename(model_id) != model_id:
        return None
    for sdir in (os.path.join(os.getcwd(), "compressed_models"), os.path.join(os.getcwd(), "models"), os.getcwd()):
        p = os.path.join(sdir, model_id)
        if os.path.isfile(p):
            return p
    return None


def servable(model_id: str):
    """(True, identifier) if the model can answer `predict`, else (False, reason). A model can if its
    file holds a state dict and the HuggingFace identifier of its architecture (what the
    compression service saves); a bare state dict (a trained custom net) has no architecture."""
    import torch
    path = find_model_path(model_id)
    if path is None:
        return False, f"model '{model_id}' not found"
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        return False, f"cannot read '{model_id}': {e}"
    ident = isinstance(ckpt, dict) and ckpt.get("metadata", {}).get("original_identifier")
    if not ident or "model_state_dict" not in ckpt:
        return False, f"'{model_id}' holds no architecture to serve (only weights)"
    return True, ident


_loaded = {}


class PredictRequest(BaseModel):
    text: str


@router.post("/{model_id}/predict")
async def predict(model_id: str, req: PredictRequest):
    """Runs the model on `text`: the mean of its last hidden states (feature extraction, the output
    of the base architecture the compression service keeps)."""
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    ok, info = servable(model_id)
    if not ok:
        raise HTTPException(status_code=404 if "not found" in info else 422, detail=info)
    if model_id not in _loaded:
        ckpt = torch.load(find_model_path(model_id), map_location="cpu", weights_only=False)
        model = AutoModel.from_config(AutoConfig.from_pretrained(info))
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        _loaded[model_id] = (model, AutoTokenizer.from_pretrained(info))
    model, tok = _loaded[model_id]
    inputs = tok(req.text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        hidden = model(**inputs).last_hidden_state[0]
    emb = hidden.mean(dim=0)
    return {"model_id": model_id, "architecture": info, "tokens": int(hidden.shape[0]), "embedding_dim": int(emb.shape[0]), "embedding": [round(float(x), 6) for x in emb]}


@router.get("/search/huggingface")
async def search_huggingface_models(query: str, limit: int = 10):
    """Search models on HuggingFace Hub"""
    try:
        # Use real HuggingFace search
        results = hf_service.search_models(query=query, limit=limit)

        # Transform results to match expected format
        return [
            {
                "id": model["id"],
                "name": model["name"],
                "size_mb": 0,  # Size not available from search
                "downloads": model["downloads"] or 0,
                "task": model["task"]
            }
            for model in results
        ]
    except Exception as e:
        print(f"⚠️ HuggingFace search failed: {e}")
        raise HTTPException(
            status_code=502,
            detail=f"HuggingFace search failed: {str(e)}"
        )
