"""
Training API endpoints - CREATE MODE
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from enum import Enum
import uuid
import asyncio

router = APIRouter()


class OptimizerType(str, Enum):
    """Training optimizer types"""
    IGQK = "igqk"
    ADAM = "adam"
    SGD = "sgd"
    ADAMW = "adamw"


class ArchitectureType(str, Enum):
    """Model architecture types"""
    RESNET18 = "resnet18"
    RESNET50 = "resnet50"
    VGG16 = "vgg16"
    EFFICIENTNET = "efficientnet"
    BERT = "bert"
    GPT2 = "gpt2"
    CUSTOM = "custom"


class TrainingConfig(BaseModel):
    """Training configuration"""
    job_name: str
    dataset_id: str
    architecture: ArchitectureType
    optimizer: OptimizerType = OptimizerType.IGQK
    epochs: int = 10
    batch_size: int = 32
    learning_rate: Optional[float] = None  # Auto if None
    num_classes: int = 10
    auto_compress: bool = True
    max_samples: Optional[int] = None  # cap on the training samples (all if None)
    publish_to_huggingface: bool = False
    huggingface_model_name: Optional[str] = None


class TrainingStatus(BaseModel):
    """Training job status"""
    job_id: str
    status: str  # queued, running, completed, failed
    progress: float  # 0.0 to 1.0
    current_epoch: int
    total_epochs: int
    current_loss: Optional[float] = None
    current_accuracy: Optional[float] = None
    quantum_entropy: Optional[float] = None
    quantum_purity: Optional[float] = None
    estimated_time_remaining: Optional[int] = None  # seconds
    message: Optional[str] = None
    test_accuracy: Optional[float] = None  # on the held-out test split
    compressed_test_accuracy: Optional[float] = None
    model_size_mb: Optional[float] = None  # FP32 parameter bytes
    compressed_size_mb: Optional[float] = None  # size of the written packed file
    checkpoint_path: Optional[str] = None
    compressed_path: Optional[str] = None


# In-memory storage for demo (replace with database in production)
training_jobs: Dict[str, Dict[str, Any]] = {}


@router.post("/start", response_model=Dict[str, str])
async def start_training(
    config: TrainingConfig,
    background_tasks: BackgroundTasks
):
    """
    Start a new training job

    This endpoint creates a new training job with the specified configuration.
    The job runs in the background and can be monitored via /status/{job_id}
    """

    # Generate job ID
    job_id = str(uuid.uuid4())

    # Validate configuration
    if config.epochs < 1 or config.epochs > 1000:
        raise HTTPException(status_code=400, detail="Epochs must be between 1 and 1000")

    if config.batch_size < 1 or config.batch_size > 512:
        raise HTTPException(status_code=400, detail="Batch size must be between 1 and 512")

    # Initialize job status
    training_jobs[job_id] = {
        "config": config.dict(),
        "status": "queued",
        "progress": 0.0,
        "current_epoch": 0,
        "created_at": None,  # TODO: Add timestamp
        "updated_at": None
    }

    # Start training in background
    background_tasks.add_task(run_training_job, job_id, config)

    return {
        "job_id": job_id,
        "status": "queued",
        "message": f"Training job '{config.job_name}' started",
        "monitor_url": f"/api/training/status/{job_id}"
    }


@router.get("/status/{job_id}", response_model=TrainingStatus)
async def get_training_status(job_id: str):
    """Get status of a training job"""

    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Training job not found")

    job = training_jobs[job_id]
    config = job["config"]

    return TrainingStatus(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        current_epoch=job["current_epoch"],
        total_epochs=config["epochs"],
        current_loss=job.get("current_loss"),
        current_accuracy=job.get("current_accuracy"),
        quantum_entropy=job.get("quantum_entropy"),
        quantum_purity=job.get("quantum_purity"),
        estimated_time_remaining=job.get("eta"),
        message=job.get("message"),
        test_accuracy=job.get("test_accuracy"),
        compressed_test_accuracy=job.get("compressed_test_accuracy"),
        model_size_mb=job.get("model_size_mb"),
        compressed_size_mb=job.get("compressed_size_mb"),
        checkpoint_path=job.get("checkpoint_path"),
        compressed_path=job.get("compressed_path")
    )


@router.get("/jobs", response_model=List[TrainingStatus])
async def list_training_jobs(
    status: Optional[str] = None,
    limit: int = 10
):
    """List all training jobs"""

    jobs = []
    for job_id, job in list(training_jobs.items())[:limit]:
        config = job["config"]
        jobs.append(TrainingStatus(
            job_id=job_id,
            status=job["status"],
            progress=job["progress"],
            current_epoch=job["current_epoch"],
            total_epochs=config["epochs"],
            current_loss=job.get("current_loss"),
            current_accuracy=job.get("current_accuracy")
        ))

    # Filter by status if provided
    if status:
        jobs = [j for j in jobs if j.status == status]

    return jobs


@router.delete("/cancel/{job_id}")
async def cancel_training_job(job_id: str):
    """Cancel a running training job"""

    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Training job not found")

    job = training_jobs[job_id]

    if job["status"] in ["completed", "failed", "cancelled"]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job in state: {job['status']}"
        )

    job["status"] = "cancelled"
    job["message"] = "Training cancelled by user"

    return {
        "job_id": job_id,
        "status": "cancelled",
        "message": "Training job cancelled successfully"
    }


@router.get("/metrics/{job_id}")
async def get_training_metrics(job_id: str):
    """Get detailed training metrics"""

    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Training job not found")

    job = training_jobs[job_id]

    return {
        "job_id": job_id,
        "metrics": {
            "loss_history": job.get("loss_history", []),
            "accuracy_history": job.get("accuracy_history", []),
            "entropy_history": job.get("entropy_history", []),
            "purity_history": job.get("purity_history", []),
            "learning_rate_history": job.get("lr_history", [])
        },
        "model_info": {
            "parameters": job.get("num_parameters"),
            "size_mb": job.get("model_size_mb"),
            "architecture": job["config"].get("architecture")
        }
    }


def _datasets(config: "TrainingConfig", is_lm: bool):
    """(train, test) datasets for `config.dataset_id`; raises ValueError for one that is not available.

    Images: MNIST and CIFAR-10 from torchvision (downloaded into $IGQK_DATA_DIR, default
    ~/.cache/igqk/data), as 3x32x32 tensors. Language models: the bytes of a local text file
    (`dataset_id` is its path), in sequences of 33 tokens; 10% of them are held out. GPT predicts
    the next byte; BERT predicts the 15% of bytes that are masked (token 256)."""
    import os
    import torch
    from torch.utils.data import Subset, TensorDataset

    if is_lm:
        path = config.dataset_id
        if not os.path.isfile(path):
            raise ValueError(f"language models train on a local text file; '{path}' is not one")
        data = torch.tensor(list(open(path, "rb").read()), dtype=torch.long)
        n = len(data) // 33
        if n < 10:
            raise ValueError(f"'{path}' is too short ({len(data)} bytes) for sequences of 33 tokens")
        seqs = data[: n * 33].view(n, 33)
        cut = max(1, n // 10)
        train, test = TensorDataset(seqs[cut:]), TensorDataset(seqs[:cut])
    else:
        import torchvision
        import torchvision.transforms as T
        root = os.environ.get("IGQK_DATA_DIR", os.path.expanduser("~/.cache/igqk/data"))
        name = config.dataset_id.lower()
        if name == "mnist":
            tf = T.Compose([T.Resize(32), T.Grayscale(3), T.ToTensor()])
            make = torchvision.datasets.MNIST
        elif name == "cifar10":
            tf = T.ToTensor()
            make = torchvision.datasets.CIFAR10
        else:
            raise ValueError(f"dataset '{config.dataset_id}' is not available (supported: mnist, cifar10, or a text file for GPT/BERT)")
        train = make(root, train=True, download=True, transform=tf)
        test = make(root, train=False, download=True, transform=tf)
        if config.max_samples:
            test = Subset(test, range(min(len(test), max(100, config.max_samples // 5))))
    if config.max_samples:
        train = Subset(train, range(min(len(train), config.max_samples)))
    return train, test


def _image_model(arch: str, num_classes: int):
    """The torchvision architecture for `arch` (untrained), or a small MLP for 'custom'."""
    import torch.nn as nn
    import torchvision.models as tvm
    if arch == "resnet18":
        return tvm.resnet18(weights=None, num_classes=num_classes)
    if arch == "resnet50":
        return tvm.resnet50(weights=None, num_classes=num_classes)
    if arch == "vgg16":
        return tvm.vgg16(weights=None, num_classes=num_classes)
    if arch == "efficientnet":
        return tvm.efficientnet_b0(weights=None, num_classes=num_classes)
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, num_classes))


async def run_training_job(job_id: str, config: TrainingConfig):
    """
    Background task to run the actual training

    Trains on the chosen dataset, measures the accuracy on its held-out test split, and with
    auto_compress projects the weights to ternary, measures the test accuracy again and stores the
    model 2-bit packed (sizes measured on the written files).
    """
    import asyncio
    import sys
    import os
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    saas_dir = os.path.dirname(backend_dir)
    pkg_dir = os.path.dirname(saas_dir)
    for p in [os.path.join(pkg_dir, 'igqk'), os.path.join(pkg_dir, 'igqk_v4'), pkg_dir]:
        if p not in sys.path:
            sys.path.insert(0, p)

    try:
        from igqk.optimizers.igqk_optimizer import IGQKOptimizer
        from igqk.compression.projectors import TernaryProjector
        from services.packing import save_packed

        training_jobs[job_id]["status"] = "running"
        training_jobs[job_id]["message"] = "Loading dataset and building the architecture..."

        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu')

        arch = config.architecture.value.lower()
        is_lm = arch in ['gpt2', 'gpt', 'bert']
        if is_lm:
            from igqk_v4.quantum_training.trainers.quantum_training_config import QuantumTrainingConfig
            m_cfg = QuantumTrainingConfig(
                model_type='BERT' if arch == 'bert' else 'GPT',
                n_layers=2,
                n_heads=2,
                d_model=64,
                d_ff=128,
                vocab_size=257,  # bytes, plus 256 = [MASK] for BERT
                max_seq_len=32,
                train_compressed=False
            )
            if arch == 'bert':
                from igqk_v4.models.bert import QuantumBERT
                model = QuantumBERT(m_cfg).to(device)
            else:
                from igqk_v4.models.gpt import QuantumGPT
                model = QuantumGPT(m_cfg).to(device)
        else:
            model = _image_model(arch, config.num_classes).to(device)

        train_set, test_set = _datasets(config, is_lm)
        data_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=max(64, config.batch_size))
        lr = config.learning_rate or (0.01 if config.optimizer in (OptimizerType.IGQK, OptimizerType.SGD) else 0.001)

        def logits_and_targets(batch, gen=None):
            """Model outputs and their targets: the class; for GPT the next byte; for BERT the
            masked bytes (masks drawn from `gen`, so a test split is masked the same way twice)."""
            if arch == 'bert':
                seq = batch[0][:, :-1]
                masked = torch.rand(seq.shape, generator=gen) < 0.15
                masked[:, 0] |= ~masked.any(dim=1)
                seq, masked = seq.to(device), masked.to(device)
                hidden = model(seq.masked_fill(masked, 256))[0]
                return model.lm_head(hidden)[masked], seq[masked]
            if is_lm:
                seq = batch[0].to(device)
                outputs = model(seq[:, :-1])
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                return outputs.reshape(-1, outputs.size(-1)), seq[:, 1:].reshape(-1)
            bx, by = batch[0].to(device), batch[1].to(device)
            return model(bx), by

        def test_accuracy():
            """Percentage of correct predictions (classes, next or masked bytes) on the test split."""
            model.eval()
            correct = total = 0
            gen = torch.Generator().manual_seed(0)
            with torch.no_grad():
                for batch in test_loader:
                    out, tgt = logits_and_targets(batch, gen)
                    correct += out.argmax(dim=-1).eq(tgt).sum().item()
                    total += tgt.numel()
            model.train()
            return round(100.0 * correct / max(1, total), 2)

        if config.optimizer == OptimizerType.IGQK:
            optimizer = IGQKOptimizer(
                model.parameters(),
                lr=lr,
                hbar=0.1,
                gamma=0.01,
                use_quantum=True,
                projector=TernaryProjector(method='optimal')
            )
        elif config.optimizer == OptimizerType.ADAMW:
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        elif config.optimizer == OptimizerType.SGD:
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        model_size_mb = sum(p.numel() * 4 for p in model.parameters()) / (1024 * 1024)
        training_jobs[job_id]["model_size_mb"] = round(model_size_mb, 4)
        training_jobs[job_id]["num_parameters"] = sum(p.numel() for p in model.parameters())

        for epoch in range(config.epochs):
            if training_jobs[job_id]["status"] == "cancelled":
                return
            model.train()
            total_loss = 0.0
            total_samples = 0

            for batch in data_loader:
                optimizer.zero_grad()
                outputs, targets = logits_and_targets(batch)
                loss = F.cross_entropy(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch[0].size(0)
                total_samples += batch[0].size(0)
                await asyncio.sleep(0)

            avg_loss = total_loss / max(1, total_samples)
            acc = test_accuracy()

            training_jobs[job_id]["progress"] = (epoch + 1) / config.epochs
            training_jobs[job_id]["current_epoch"] = epoch + 1
            training_jobs[job_id]["current_loss"] = round(avg_loss, 4)
            training_jobs[job_id]["current_accuracy"] = acc
            training_jobs[job_id]["test_accuracy"] = acc
            training_jobs[job_id].setdefault("loss_history", []).append(round(avg_loss, 4))
            training_jobs[job_id].setdefault("accuracy_history", []).append(acc)

            if hasattr(optimizer, 'entropy'):
                training_jobs[job_id]["quantum_entropy"] = round(optimizer.entropy(), 4)
                training_jobs[job_id]["quantum_purity"] = round(optimizer.purity(), 4)

            training_jobs[job_id]["message"] = f"Epoch {epoch+1}/{config.epochs} completed (Loss: {avg_loss:.4f}, test accuracy {acc:.2f}%)"

        models_dir = os.path.join(os.getcwd(), "models")
        os.makedirs(models_dir, exist_ok=True)
        checkpoint_path = os.path.join(models_dir, f"{job_id}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        training_jobs[job_id]["checkpoint_path"] = checkpoint_path

        if config.auto_compress:
            training_jobs[job_id]["message"] = "Compressing model to ternary weights..."
            IGQKOptimizer(model.parameters(), projector=TernaryProjector()).compress(model)
            compressed_acc = test_accuracy()

            save_dir = os.path.join(os.getcwd(), "compressed_models")
            save_path = os.path.join(save_dir, f"{job_id}_{config.job_name}.pt")
            saved = save_packed(model.state_dict(), save_path, metadata={"job_id": job_id, "architecture": arch})

            training_jobs[job_id]["compressed_test_accuracy"] = compressed_acc
            training_jobs[job_id]["compressed_size_mb"] = round(saved["file_bytes"] / (1024 * 1024), 4)
            training_jobs[job_id]["compressed_path"] = save_path
            training_jobs[job_id]["save_path"] = save_path
            ratio = model_size_mb * 1024 * 1024 / saved["file_bytes"]
            training_jobs[job_id]["message"] = (
                f"✅ Training & compression completed: {ratio:.1f}× smaller file, "
                f"test accuracy {training_jobs[job_id]['test_accuracy']:.2f}% → {compressed_acc:.2f}%"
            )
        else:
            training_jobs[job_id]["message"] = f"✅ Training completed. Checkpoint: {checkpoint_path}"

        training_jobs[job_id]["status"] = "completed"
        training_jobs[job_id]["progress"] = 1.0

    except Exception as e:
        training_jobs[job_id]["status"] = "failed"
        training_jobs[job_id]["message"] = f"Training failed: {str(e)}"
