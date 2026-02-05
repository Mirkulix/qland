"""IGQK v4.0 Model Library.

This module contains quantum-enhanced neural network models:
- QuantumGPT: Autoregressive language model
- QuantumBERT: Bidirectional encoder
- QuantumViT: Vision transformer

All models support optional ternary weight compression via TernaryLinear.
"""

from .gpt import QuantumGPT
from .bert import QuantumBERT
from .vit import QuantumViT

__all__ = ['QuantumGPT', 'QuantumBERT', 'QuantumViT']

__version__ = '4.0.0'
