"""
Compression API endpoints - COMPRESS MODE
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from enum import Enum
import uuid
import sys
import os

# Add database
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from database import db

router = APIRouter()


class CompressionMethod(str, Enum):
    """Compression methods"""
    AUTO = "auto"  # AI chooses best method
    TERNARY = "ternary"  # 16× compression
    BINARY = "binary"  # 32× compression
    SPARSE = "sparse"  # Variable compression
    LOWRANK = "lowrank"  # Variable compression


class ModelSource(str, Enum):
    """Where the model comes from"""
    HUGGINGFACE = "huggingface"
    UPLOAD = "upload"
    MY_MODELS = "my_models"
    URL = "url"


class CompressionConfig(BaseModel):
    """Compression configuration"""
    job_name: str
    model_source: ModelSource
    model_identifier: str  # HF model name, file path, or URL
    compression_method: CompressionMethod = CompressionMethod.AUTO
    quality_target: float = 0.95  # Target: retain 95% of original quality
    auto_validate: bool = True
    publish_compressed: bool = False
    huggingface_model_name: Optional[str] = None


class CompressionStatus(BaseModel):
    """Compression job status"""
    job_id: str
    status: str  # queued, analyzing, compressing, validating, completed, failed
    progress: float  # 0.0 to 1.0
    phase: str  # Current phase
    original_size_mb: Optional[float] = None
    compressed_size_mb: Optional[float] = None
    compression_ratio: Optional[float] = None
    memory_saved_percent: Optional[float] = None
    original_accuracy: Optional[float] = None
    compressed_accuracy: Optional[float] = None
    accuracy_loss_percent: Optional[float] = None
    inference_speedup: Optional[float] = None
    estimated_time_remaining: Optional[int] = None
    message: Optional[str] = None


class CompressionResult(BaseModel):
    """Final compression result"""
    job_id: str
    original_model: Dict[str, Any]
    compressed_model: Dict[str, Any]
    comparison: Dict[str, Any]
    recommendation: str
    download_urls: Dict[str, str]


# In-memory storage for demo
compression_jobs: Dict[str, Dict[str, Any]] = {}


@router.post("/start", response_model=Dict[str, str])
async def start_compression(
    config: CompressionConfig,
    background_tasks: BackgroundTasks
):
    """
    Start a new compression job

    This endpoint creates a new compression job for an existing model.
    The job analyzes the model, applies IGQK compression, and validates results.
    """

    # Generate job ID
    job_id = str(uuid.uuid4())

    # Validate configuration
    if config.quality_target < 0.5 or config.quality_target > 1.0:
        raise HTTPException(
            status_code=400,
            detail="Quality target must be between 0.5 and 1.0"
        )

    # Create job in database
    db.create_job(
        job_id=job_id,
        job_name=config.job_name,
        job_type="compression",
        model_identifier=config.model_identifier,
        model_source=config.model_source.value,
        compression_method=config.compression_method.value,
        quality_target=config.quality_target,
        auto_validate=config.auto_validate
    )

    # Start compression in background
    background_tasks.add_task(run_compression_job, job_id, config)

    return {
        "job_id": job_id,
        "status": "queued",
        "message": f"Compression job '{config.job_name}' started",
        "monitor_url": f"/api/compression/status/{job_id}"
    }


@router.get("/status/{job_id}", response_model=CompressionStatus)
async def get_compression_status(job_id: str):
    """Get status of a compression job"""

    # Get from database
    job = db.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Compression job not found")

    # Parse results if available
    results = job.get("results", {}) or {}

    return CompressionStatus(
        job_id=job_id,
        status=job["status"],
        progress=1.0 if job["status"] == "completed" else 0.5,
        phase=job["status"],  # Use status as phase
        original_size_mb=results.get("original_size_mb"),
        compressed_size_mb=results.get("compressed_size_mb"),
        compression_ratio=results.get("compression_ratio"),
        memory_saved_percent=results.get("memory_saved_percent"),
        original_accuracy=results.get("original_accuracy"),
        compressed_accuracy=results.get("compressed_accuracy"),
        accuracy_loss_percent=results.get("accuracy_loss"),
        inference_speedup=results.get("inference_speedup"),
        estimated_time_remaining=None,
        message=job.get("error") if job["status"] == "failed" else None
    )


@router.get("/result/{job_id}", response_model=CompressionResult)
async def get_compression_result(job_id: str):
    """Get detailed compression results"""

    if job_id not in compression_jobs:
        raise HTTPException(status_code=404, detail="Compression job not found")

    job = compression_jobs[job_id]

    if job["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job not completed. Current status: {job['status']}"
        )

    original_size = job.get("original_size_mb", 0.0)
    compressed_size = job.get("compressed_size_mb", 0.0)
    compression_ratio = (original_size / compressed_size) if compressed_size > 0 else 1.0
    memory_saved = (((original_size - compressed_size) / original_size) * 100) if original_size > 0 else 0.0

    original_acc = job.get("original_accuracy")
    compressed_acc = job.get("compressed_accuracy")
    acc_loss = (original_acc - compressed_acc) if (original_acc is not None and compressed_acc is not None) else None

    param_count = job.get("parameters") or int((original_size * 1024 * 1024) / 4)
    bits_per_weight = 2 if "ternary" in str(job.get("compression_method", "ternary")).lower() else (1 if "binary" in str(job.get("compression_method")).lower() else None)

    return CompressionResult(
        job_id=job_id,
        original_model={
            "size_mb": round(original_size, 2),
            "accuracy": original_acc,
            "inference_time_ms": None,
            "parameters": param_count
        },
        compressed_model={
            "size_mb": round(compressed_size, 2),
            "accuracy": compressed_acc,
            "inference_time_ms": None,
            "parameters": param_count,
            "bits_per_weight": bits_per_weight
        },
        comparison={
            "compression_ratio": f"{compression_ratio:.1f}×",
            "memory_saved_percent": f"{memory_saved:.1f}%",
            "accuracy_loss_percent": f"{acc_loss:.2f}%" if acc_loss is not None else "N/A",
            "inference_speedup": "N/A (not measured)",
            "quality_retained": f"{(compressed_acc/original_acc)*100:.1f}%" if (original_acc and compressed_acc) else "N/A"
        },
        recommendation=f"✅ Compressed model recommended: {compression_ratio:.1f}× smaller ({memory_saved:.1f}% memory saved)",
        download_urls={
            "compressed_model": f"/api/models/download/{job_id}/compressed",
            "original_model": f"/api/models/download/{job_id}/original"
        }
    )


@router.get("/jobs", response_model=List[CompressionStatus])
async def list_compression_jobs(
    status: Optional[str] = None,
    limit: int = 10
):
    """List all compression jobs"""

    jobs = []
    for job_id, job in list(compression_jobs.items())[:limit]:
        jobs.append(CompressionStatus(
            job_id=job_id,
            status=job["status"],
            progress=job["progress"],
            phase=job["phase"],
            compression_ratio=job.get("compression_ratio")
        ))

    if status:
        jobs = [j for j in jobs if j.status == status]

    return jobs


@router.post("/upload")
async def upload_model_for_compression(
    file: UploadFile = File(...),
    job_name: str = "Uploaded Model",
    compression_method: CompressionMethod = CompressionMethod.AUTO
):
    """Upload a model file for compression"""

    # Validate file type
    if not file.filename.endswith(('.pt', '.pth', '.onnx')):
        raise HTTPException(
            status_code=400,
            detail="Only .pt, .pth, or .onnx files are supported"
        )

    upload_dir = os.path.join(os.getcwd(), "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    save_path = os.path.join(upload_dir, file.filename)
    contents = await file.read()
    with open(save_path, "wb") as f:
        f.write(contents)

    file_size_mb = round(len(contents) / (1024 * 1024), 2)
    job_id = str(uuid.uuid4())

    return {
        "job_id": job_id,
        "message": f"Model '{file.filename}' uploaded and saved successfully",
        "file_size_mb": file_size_mb,
        "save_path": save_path,
        "next_step": f"/api/compression/start with job_name='{job_name}'"
    }


@router.delete("/cancel/{job_id}")
async def cancel_compression_job(job_id: str):
    """Cancel a running compression job"""

    if job_id not in compression_jobs:
        raise HTTPException(status_code=404, detail="Compression job not found")

    job = compression_jobs[job_id]

    if job["status"] in ["completed", "failed", "cancelled"]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job in state: {job['status']}"
        )

    job["status"] = "cancelled"
    job["message"] = "Compression cancelled by user"

    return {
        "job_id": job_id,
        "status": "cancelled",
        "message": "Compression job cancelled successfully"
    }


async def run_compression_job(job_id: str, config: CompressionConfig):
    """
    Background task to run the actual compression

    This integrates with IGQK core engine for REAL compression
    """
    import asyncio
    import sys
    import os

    # Add IGQK to path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'igqk'))

    try:
        # Import compression service
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from services.compression_service import CompressionService

        compression_service = CompressionService()

        # Phase 1: Starting compression
        compression_jobs[job_id]["status"] = "analyzing"
        compression_jobs[job_id]["phase"] = "Downloading and analyzing model"
        compression_jobs[job_id]["progress"] = 0.1
        compression_jobs[job_id]["message"] = f"Downloading model from {config.model_source}..."

        # Determine compression method
        method_map = {
            CompressionMethod.AUTO: "ternary",  # AUTO defaults to ternary
            CompressionMethod.TERNARY: "ternary",
            CompressionMethod.BINARY: "binary",
            CompressionMethod.SPARSE: "sparse",
            CompressionMethod.LOWRANK: "lowrank"
        }
        method = method_map.get(config.compression_method, "ternary")

        # Phase 2: Actual compression
        compression_jobs[job_id]["status"] = "compressing"
        compression_jobs[job_id]["phase"] = "Applying IGQK compression"
        compression_jobs[job_id]["progress"] = 0.3
        compression_jobs[job_id]["message"] = f"Compressing with IGQK {method} method..."

        if config.model_source == ModelSource.HUGGINGFACE:
            # Download and compress from HuggingFace
            result = compression_service.compress_huggingface_model(
                model_identifier=config.model_identifier,
                method=method,
                job_id=job_id
            )

            if result["status"] == "completed":
                # Update job with real results
                compression_jobs[job_id]["original_size_mb"] = result["original"]["size_mb"]
                compression_jobs[job_id]["compressed_size_mb"] = result["compressed"]["size_mb"]
                compression_jobs[job_id]["compression_ratio"] = result["comparison"]["compression_ratio"]
                compression_jobs[job_id]["memory_saved_percent"] = result["comparison"]["memory_saved_percent"]
                compression_jobs[job_id]["save_path"] = result.get("save_path")

                # Phase 3: Real Validation
                if config.auto_validate:
                    compression_jobs[job_id]["status"] = "validating"
                    compression_jobs[job_id]["phase"] = "Validating compressed model"
                    compression_jobs[job_id]["progress"] = 0.8

                    val = result.get("validation", {})
                    # None when not measured (never a default accuracy).
                    orig_acc = val.get("original_accuracy")
                    comp_acc = val.get("compressed_accuracy")
                    loss_pct = val.get("accuracy_loss")

                    compression_jobs[job_id]["original_accuracy"] = orig_acc
                    compression_jobs[job_id]["compressed_accuracy"] = comp_acc
                    compression_jobs[job_id]["accuracy_loss_percent"] = loss_pct
                    compression_jobs[job_id]["message"] = f"Validation complete (Loss: {loss_pct:.1f}%)"

                # Completed
                compression_jobs[job_id]["status"] = "completed"
                compression_jobs[job_id]["phase"] = "Complete"
                compression_jobs[job_id]["progress"] = 1.0
                compression_jobs[job_id]["message"] = f"✅ Compression completed! {result['comparison']['compression_ratio']:.1f}× smaller"

            else:
                raise Exception(result.get("error", "Unknown error during compression"))

        else:
            compression_jobs[job_id]["status"] = "failed"
            compression_jobs[job_id]["message"] = f"Model source '{config.model_source}' not supported. Use HuggingFace or Upload."

    except Exception as e:
        compression_jobs[job_id]["status"] = "failed"
        compression_jobs[job_id]["message"] = f"Compression failed: {str(e)}"
        compression_jobs[job_id]["error_details"] = str(e)
