"""
Quantum Vision Encoder (ViT-based).

Encodes input images into feature vectors (batch, d_model) using
Quantum-enhanced Vision Transformer architecture with optional ternary compression.
"""

import torch
import torch.nn as nn
from typing import Optional

from ...quantum_training.trainers.quantum_training_config import QuantumTrainingConfig
from ...models.vit import QuantumViT


class QuantumVisionEncoder(nn.Module):
    """
    Quantum Vision Encoder.

    Transforms images into (batch_size, d_model) feature vectors using
    Quantum-enhanced Vision Transformer (ViT) architecture.
    """

    def __init__(self, config: QuantumTrainingConfig):
        super().__init__()
        self.config = config
        self.vit = QuantumViT(config)
        self.proj = nn.Linear(config.d_model, config.d_model)
        self.ln = nn.LayerNorm(config.d_model)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images into feature vectors.

        Args:
            images: (batch_size, 3, 224, 224) RGB image tensor

        Returns:
            features: (batch_size, d_model) image representation
        """
        # Get [CLS] token features from ViT
        features = self.vit(images, return_features=True)
        features = self.proj(features)
        features = self.ln(features)
        return features
