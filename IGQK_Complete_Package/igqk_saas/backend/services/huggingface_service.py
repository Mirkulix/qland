"""
HuggingFace Integration Service
Downloads and manages models from HuggingFace Hub
"""

import os
import torch
import ssl
import urllib3
from typing import Optional, Dict, Any
from transformers import AutoModel, AutoTokenizer, AutoConfig
from huggingface_hub import hf_hub_download, list_models, model_info

# Fix for SSL certificate verification issues on Windows/Anaconda
# This disables SSL warnings for development/local use
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Disable SSL verification for huggingface_hub
# NOTE: Only for local development! For production, install proper certificates.
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

class SSLAdapter(HTTPAdapter):
    """Custom adapter to disable SSL verification"""
    def init_poolmanager(self, *args, **kwargs):
        kwargs['ssl_version'] = ssl.PROTOCOL_TLS
        kwargs['cert_reqs'] = ssl.CERT_NONE
        return super().init_poolmanager(*args, **kwargs)

# Patch the default session
_original_session = requests.Session
def _patched_session():
    session = _original_session()
    session.mount('https://', SSLAdapter())
    session.verify = False
    return session

# Only patch if SSL verification is causing issues
import os as _os
if _os.environ.get('DISABLE_SSL_VERIFY', 'true').lower() == 'true':
    requests.Session = _patched_session


class HuggingFaceService:
    """Service for interacting with HuggingFace Hub"""

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize HuggingFace service

        Args:
            cache_dir: Directory to cache downloaded models
        """
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), "models_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

    def download_model(
        self,
        model_identifier: str,
        include_tokenizer: bool = True
    ) -> Dict[str, Any]:
        """
        Download a model from HuggingFace Hub

        Args:
            model_identifier: HuggingFace model name (e.g., "bert-base-uncased")
            include_tokenizer: Whether to download tokenizer as well

        Returns:
            Dictionary with model, tokenizer, config, and metadata
        """
        print(f"📥 Downloading model: {model_identifier}")

        try:
            # Get model info
            info = model_info(model_identifier)

            # Download config
            config = AutoConfig.from_pretrained(
                model_identifier,
                cache_dir=self.cache_dir
            )

            # Download model
            model = AutoModel.from_pretrained(
                model_identifier,
                cache_dir=self.cache_dir
            )

            # Download tokenizer if requested
            tokenizer = None
            if include_tokenizer:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_identifier,
                        cache_dir=self.cache_dir
                    )
                except Exception as e:
                    print(f"⚠️ Could not load tokenizer: {e}")

            # Calculate model size
            param_count = sum(p.numel() for p in model.parameters())
            size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
            size_mb = size_bytes / (1024 ** 2)

            print(f"✅ Downloaded: {param_count:,} parameters, {size_mb:.2f} MB")

            return {
                "model": model,
                "tokenizer": tokenizer,
                "config": config,
                "metadata": {
                    "identifier": model_identifier,
                    "parameters": param_count,
                    "size_mb": size_mb,
                    "size_bytes": size_bytes,
                    "model_type": config.model_type if hasattr(config, "model_type") else "unknown",
                    "downloads": info.downloads if hasattr(info, "downloads") else 0,
                    "tags": info.tags if hasattr(info, "tags") else [],
                }
            }

        except Exception as e:
            raise RuntimeError(f"Failed to download model '{model_identifier}': {str(e)}")

    def search_models(
        self,
        query: str,
        task: Optional[str] = None,
        limit: int = 10
    ) -> list:
        """
        Search for models on HuggingFace Hub

        Args:
            query: Search query
            task: Filter by task (e.g., "text-classification")
            limit: Maximum number of results

        Returns:
            List of model information dictionaries
        """
        try:
            models = list_models(
                search=query,
                task=task,
                limit=limit,
                sort="downloads",
                direction=-1
            )

            results = []
            for model in models:
                results.append({
                    "id": model.modelId,
                    "name": model.modelId.split("/")[-1] if "/" in model.modelId else model.modelId,
                    "downloads": model.downloads,
                    "tags": model.tags,
                    "task": model.pipeline_tag if hasattr(model, "pipeline_tag") else "unknown"
                })

            return results

        except Exception as e:
            # Raised, not swallowed: an unreachable Hub is an error (HTTP 502), not "no results".
            print(f"⚠️ Search failed: {e}")
            raise

    def get_model_size(self, model: torch.nn.Module) -> Dict[str, Any]:
        """
        Get detailed size information for a model

        Args:
            model: PyTorch model

        Returns:
            Dictionary with size information
        """
        param_count = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        size_mb = size_bytes / (1024 ** 2)
        size_gb = size_bytes / (1024 ** 3)

        return {
            "total_parameters": param_count,
            "trainable_parameters": trainable_params,
            "size_bytes": size_bytes,
            "size_mb": round(size_mb, 2),
            "size_gb": round(size_gb, 4)
        }

    def save_model(
        self,
        model: torch.nn.Module,
        save_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Save a model to disk

        Args:
            model: PyTorch model to save
            save_path: Path to save the model
            metadata: Optional metadata to save with model
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Save model
        torch.save({
            "model_state_dict": model.state_dict(),
            "metadata": metadata or {}
        }, save_path)

        print(f"💾 Model saved to: {save_path}")

    def load_local_model(self, model_path: str) -> Dict[str, Any]:
        """
        Load a locally saved model

        Args:
            model_path: Path to the saved model file

        Returns:
            Dictionary with model and metadata
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        checkpoint = torch.load(model_path)

        return {
            "state_dict": checkpoint["model_state_dict"],
            "metadata": checkpoint.get("metadata", {})
        }
