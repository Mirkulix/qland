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

# TODO: Phase 2 - Multi-Modal AI (Not yet implemented)
# from .multimodal.fusion.quantum_fusion import QuantumMultiModalFusion
# from .multimodal.models.multimodal_model import MultiModalModel

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

    # Theory
    "HybridLaplaceWavelet",
    "TernaryLieGroup",
    "FractionalHebbian",

    # TODO: Add these in Phase 2+
    # "QuantumMultiModalFusion",
    # "MultiModalModel",
    # "DistributedQuantumTrainer",
    # "FullyShardedTrainer",
    # "AutoTuner",
    # "HyperparameterSearch",
    # "EdgeDeployer",
    # "CloudDeployer",
    # "ProgressiveLoader",
    # "TernaryMatMul",
    # "TernaryConv2d",
    # "HardwareOptimizer",
]

# Version info
def get_version():
    """Get current IGQK version."""
    return __version__

def get_features():
    """Get list of available features in v4.0."""
    return {
        # Phase 1 - IMPLEMENTED
        "quantum_training": True,
        "models_gpt": True,
        "models_bert": True,
        "models_vit": True,
        "hlwt": True,
        "tlgt": True,
        "fchl": True,

        # Phase 2+ - NOT YET IMPLEMENTED
        "multimodal": False,
        "distributed": False,
        "automl": False,
        "hardware_acceleration": False,
        "edge_deployment": False,
        "cloud_deployment": False,
        "progressive_loading": False,
    }

print(f"IGQK v{__version__} loaded successfully!")
print(f"Available features: {len([v for v in get_features().values() if v])} modules implemented")
