"""
IGQK v4.0 - Unified Quantum-Classical Hybrid AI Platform

This is the next generation of IGQK that combines:
- Quantum Training from Scratch (v2.0 vision)
- Enterprise SaaS Platform (v3.0)
- Advanced Mathematical Frameworks (HLWT, TLGT, FCHL)
- Multi-Modal AI Support
- Distributed Training
- Hardware Acceleration

Version: 4.0.0
Release Date: 2026-02-05
"""

__version__ = "4.0.0"
__author__ = "IGQK Development Team"
__license__ = "MIT"

# Core Quantum Training
from .quantum_training.trainers.quantum_llm_trainer import QuantumLLMTrainer
from .quantum_training.trainers.quantum_training_config import QuantumTrainingConfig

# Models (Phase 1 - IMPLEMENTED)
from .models.gpt import QuantumGPT
from .models.bert import QuantumBERT
from .models.vit import QuantumViT

# Advanced Theory
from .theory.hlwt.hybrid_laplace_wavelet import HybridLaplaceWavelet
from .theory.tlgt.ternary_lie_group import TernaryLieGroup
from .theory.fchl.fractional_hebbian import FractionalHebbian

# Phase 2 - Multi-Modal AI (IMPLEMENTED)
from .multimodal import (
    QuantumVisionEncoder,
    QuantumLanguageEncoder,
    QuantumMultiModalFusion,
    MultiModalModel,
)

# Phase 3 - Compression & 2-Bit Quantization (IMPLEMENTED)
from .compression import (
    TernaryBitPacker,
    LLMWeightCompressor,
)

# TODO: Phase 3+ - Advanced Features (Not yet implemented)
# from .distributed.ddp.distributed_quantum_trainer import DistributedQuantumTrainer
# from .distributed.fsdp.fully_sharded_trainer import FullyShardedTrainer
# from .automl.tuning.auto_tuner import AutoTuner
# from .automl.tuning.hyperparameter_search import HyperparameterSearch
# from .deployment.edge.edge_deployer import EdgeDeployer
# from .deployment.cloud.cloud_deployer import CloudDeployer
# from .deployment.progressive.progressive_loader import ProgressiveLoader
# from .hardware.cuda.ternary_kernels import TernaryMatMul, TernaryConv2d
# from .hardware.optimization.model_optimizer import HardwareOptimizer

__all__ = [
    # Core Training
    "QuantumLLMTrainer",
    "QuantumTrainingConfig",

    # Models (Phase 1 - IMPLEMENTED)
    "QuantumGPT",
    "QuantumBERT",
    "QuantumViT",

    # Multi-Modal (Phase 2 - IMPLEMENTED)
    "QuantumVisionEncoder",
    "QuantumLanguageEncoder",
    "QuantumMultiModalFusion",
    "MultiModalModel",

    # Compression (Phase 3 - IMPLEMENTED)
    "TernaryBitPacker",
    "LLMWeightCompressor",

    # Theory
    "HybridLaplaceWavelet",
    "TernaryLieGroup",
    "FractionalHebbian",
]

# Version info
def get_version():
    """Get current IGQK version."""
    return __version__

def get_features():
    """Get list of available features in v4.0."""
    return {
        # Phase 1, 2 & 3 - IMPLEMENTED
        "quantum_training": True,
        "models_gpt": True,
        "models_bert": True,
        "models_vit": True,
        "multimodal": True,
        "compression": True,
        "hlwt": True,
        "tlgt": True,
        "fchl": True,

        # Phase 4+ - NOT YET IMPLEMENTED
        "distributed": False,
        "automl": False,
        "hardware_acceleration": False,
        "deployment": False,
    }

print(f"IGQK v{__version__} loaded successfully!")
print(f"Available features: {len([v for v in get_features().values() if v])} modules implemented")
