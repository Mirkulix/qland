"""
Compression Service
Handles actual model compression using IGQK
"""

import sys
import os
import torch
import time
from typing import Dict, Any, Optional

# Add IGQK to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'igqk'))

from igqk import IGQKOptimizer, TernaryProjector, BinaryProjector, SparseProjector, LowRankProjector
from .huggingface_service import HuggingFaceService
from .validation_service import ValidationService
from .packing import save_packed


class CompressionService:
    """Service for compressing models with IGQK"""

    def __init__(self):
        self.hf_service = HuggingFaceService()
        self.validation_service = ValidationService()
        self.models_dir = os.path.join(os.getcwd(), "compressed_models")
        os.makedirs(self.models_dir, exist_ok=True)

    def compress_huggingface_model(
        self,
        model_identifier: str,
        method: str = "ternary",
        job_id: str = None,
        auto_validate: bool = True
    ) -> Dict[str, Any]:
        """
        Download and compress a HuggingFace model

        Args:
            model_identifier: HuggingFace model name
            method: Compression method (ternary, binary, sparse, lowrank)
            job_id: Job ID for tracking

        Returns:
            Dictionary with compression results
        """
        results = {
            "job_id": job_id,
            "model_identifier": model_identifier,
            "method": method,
            "status": "started",
            "original": {},
            "compressed": {},
            "comparison": {}
        }

        try:
            # Step 1: Download model from HuggingFace
            print(f"📥 Step 1/4: Downloading model from HuggingFace...")
            download_result = self.hf_service.download_model(model_identifier)

            model = download_result["model"]
            metadata = download_result["metadata"]

            # Get original model stats
            original_size = self.hf_service.get_model_size(model)
            results["original"] = {
                "size_mb": original_size["size_mb"],
                "parameters": original_size["total_parameters"],
                "identifier": model_identifier
            }

            print(f"✅ Original model: {original_size['size_mb']:.2f} MB, {original_size['total_parameters']:,} params")

            # Step 2: Choose compression method
            print(f"🔧 Step 2/4: Applying {method} compression...")

            if method == "ternary":
                projector = TernaryProjector()
            elif method == "binary":
                projector = BinaryProjector()
            elif method == "sparse":
                projector = SparseProjector(sparsity=0.5)
            elif method == "lowrank":
                projector = LowRankProjector(rank=10)
            else:
                projector = TernaryProjector()  # Default

            # Step 3: Compress with IGQK
            print(f"🗜️  Step 3/4: Compressing with IGQK...")
            start_time = time.time()

            optimizer = IGQKOptimizer(
                model.parameters(),
                projector=projector
            )

            # Apply compression
            optimizer.compress(model)

            compression_time = time.time() - start_time

            # Save the compressed model (ternary tensors 2-bit packed); its size is measured on the file
            print(f"💾 Step 4/5: Saving compressed model...")
            save_path = os.path.join(
                self.models_dir,
                f"{job_id}_{model_identifier.replace('/', '_')}_compressed.pt"
            )
            saved = save_packed(model.state_dict(), save_path, metadata={
                "original_identifier": model_identifier,
                "compression_method": method,
                "original_size_mb": original_size["size_mb"]
            })
            compressed_size = {
                "size_mb": round(saved["file_bytes"] / (1024 ** 2), 2),
                "total_parameters": original_size["total_parameters"]
            }
            results["compressed"] = {
                "size_mb": compressed_size["size_mb"],
                "file_bytes": saved["file_bytes"],
                "packed_tensors": saved["packed_tensors"],
                "dense_tensors": saved["dense_tensors"],
                "parameters": compressed_size["total_parameters"],
                "compression_time": compression_time
            }

            # Calculate metrics (original: FP32 parameter bytes; compressed: the written file)
            compression_ratio = original_size["size_bytes"] / saved["file_bytes"]
            memory_saved_percent = (1 - saved["file_bytes"] / original_size["size_bytes"]) * 100

            results["comparison"] = {
                "compression_ratio": round(compression_ratio, 2),
                "memory_saved_percent": round(memory_saved_percent, 2),
                "compression_time_seconds": round(compression_time, 2)
            }

            print(f"✅ Compressed: {compressed_size['size_mb']:.2f} MB ({compression_ratio:.1f}× smaller)")

            # Step 3.5: Validate compressed model (if requested)
            if auto_validate:
                print(f"🔍 Step 3.5/5: Validating compressed model...")

                # Make a copy of original model for validation
                original_model_copy = self.hf_service.download_model(model_identifier)["model"]

                validation_results = self.validation_service.quick_validate(
                    original_model_copy,
                    model
                )

                results["validation"] = validation_results
                if validation_results.get("original_accuracy") is not None:
                    print(f"✅ Validation: Original={validation_results['original_accuracy']:.1f}%, "
                          f"Compressed={validation_results['compressed_accuracy']:.1f}%, "
                          f"Loss={validation_results['accuracy_loss']:.1f}%")
            else:
                # Not validated: no accuracy is known.
                results["validation"] = {
                    "original_accuracy": None,
                    "compressed_accuracy": None,
                    "accuracy_loss": None,
                    "validation_error": "not validated"
                }

            results["save_path"] = save_path
            results["status"] = "completed"

            print(f"🎉 Compression complete! Saved to: {save_path}")

            return results

        except Exception as e:
            print(f"❌ Error during compression: {str(e)}")
            results["status"] = "failed"
            results["error"] = str(e)
            return results

    def compress_uploaded_model(
        self,
        model_path: str,
        method: str = "ternary",
        job_id: str = None
    ) -> Dict[str, Any]:
        """
        Compress an uploaded model file

        Args:
            model_path: Path to the model file
            method: Compression method
            job_id: Job ID for tracking

        Returns:
            Dictionary with compression results
        """
        results = {
            "job_id": job_id,
            "model_path": model_path,
            "method": method,
            "status": "started"
        }

        try:
            # Load model
            print(f"📂 Loading model from: {model_path}")
            model = torch.load(model_path)

            # Get size
            original_size = self.hf_service.get_model_size(model)
            results["original"] = {
                "size_mb": original_size["size_mb"],
                "parameters": original_size["total_parameters"]
            }

            # Compress (same logic as HF models)
            if method == "ternary":
                projector = TernaryProjector()
            elif method == "binary":
                projector = BinaryProjector()
            else:
                projector = TernaryProjector()

            optimizer = IGQKOptimizer(model.parameters(), projector=projector)
            optimizer.compress(model)

            # Save (ternary tensors 2-bit packed); the compressed size is measured on the file
            save_path = os.path.join(self.models_dir, f"{job_id}_compressed.pt")
            saved = save_packed(model.state_dict(), save_path, metadata={"compression_method": method})
            results["compressed"] = {
                "size_mb": round(saved["file_bytes"] / (1024 ** 2), 2),
                "file_bytes": saved["file_bytes"],
                "parameters": original_size["total_parameters"]
            }
            results["comparison"] = {
                "compression_ratio": round(original_size["size_bytes"] / saved["file_bytes"], 2),
                "memory_saved_percent": round((1 - saved["file_bytes"] / original_size["size_bytes"]) * 100, 2)
            }

            results["save_path"] = save_path
            results["status"] = "completed"

            return results

        except Exception as e:
            results["status"] = "failed"
            results["error"] = str(e)
            return results
