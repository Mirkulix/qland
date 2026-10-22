"""
Quantum Language Encoder (BERT-based).

Encodes input text tokens into feature vectors (batch, d_model) using
Quantum-enhanced Bidirectional Transformer architecture with optional ternary compression.
"""

import torch
import torch.nn as nn
from typing import Optional

from ...quantum_training.trainers.quantum_training_config import QuantumTrainingConfig
from ...models.bert import QuantumBERT


class QuantumLanguageEncoder(nn.Module):
    """
    Quantum Language Encoder.

    Transforms text token sequences into (batch_size, d_model) feature vectors
    using Quantum-enhanced BERT architecture.
    """

    def __init__(self, config: QuantumTrainingConfig):
        super().__init__()
        self.config = config
        self.bert = QuantumBERT(config)
        self.proj = nn.Linear(config.d_model, config.d_model)
        self.ln = nn.LayerNorm(config.d_model)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode text into feature vectors.

        Args:
            input_ids: (batch_size, seq_len) token IDs
            attention_mask: optional attention mask (batch_size, seq_len)

        Returns:
            features: (batch_size, d_model) text representation
        """
        cls_rep = self.bert.get_cls_representation(input_ids, attention_mask=attention_mask)
        features = self.proj(cls_rep)
        features = self.ln(features)
        return features
