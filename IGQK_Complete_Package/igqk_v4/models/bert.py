"""
Quantum-Enhanced BERT Model.

Implements a BERT-style bidirectional transformer encoder with optional ternary weight compression.
Similar to GPT but bidirectional (no causal masking).
"""

import torch
import torch.nn as nn
from typing import Optional

from ..quantum_training.trainers.quantum_training_config import QuantumTrainingConfig
from .gpt import QuantumTransformerBlock  # Reuse from GPT!


class QuantumBERT(nn.Module):
    """
    Quantum-Enhanced BERT Model.

    Features:
    - Bidirectional transformer encoder
    - No causal masking (can attend to all positions)
    - [CLS] token for classification tasks
    - Optional ternary weight compression
    - Compatible with Quantum Gradient Flow training

    Args:
        config: QuantumTrainingConfig with model parameters
    """

    def __init__(self, config: QuantumTrainingConfig):
        super().__init__()

        self.config = config
        use_ternary = config.train_compressed

        # Embeddings
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.d_model)

        # Segment embeddings (for sentence pairs, like original BERT)
        self.segment_embed = nn.Embedding(2, config.d_model)

        # Transformer blocks (reuse from GPT!)
        self.blocks = nn.ModuleList([
            QuantumTransformerBlock(
                config.d_model,
                config.n_heads,
                config.d_ff,
                config.dropout,
                use_ternary
            )
            for _ in range(config.n_layers)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.d_model)

        # Pooler for [CLS] token (classification head)
        self.pooler = nn.Linear(config.d_model, config.d_model)
        self.pooler_activation = nn.Tanh()

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

        # Initialize weights
        self.apply(self._init_weights)

        # Count parameters
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f" QuantumBERT initialized:")
        print(f"   Layers: {config.n_layers}, Heads: {config.n_heads}, d_model: {config.d_model}")
        print(f"   Ternary: {use_ternary}")
        print(f"   Parameters: {n_params:,}")

    def _init_weights(self, module):
        """Initialize weights (BERT style)."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass.

        Args:
            input_ids: (batch_size, seq_len) Long tensor of token indices
            attention_mask: (batch_size, seq_len) optional attention mask (1 = attend, 0 = ignore)
            token_type_ids: (batch_size, seq_len) optional segment IDs (0 or 1)

        Returns:
            sequence_output: (batch_size, seq_len, d_model) all token representations
            pooled_output: (batch_size, d_model) [CLS] token representation
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Token embeddings
        token_embeds = self.token_embed(input_ids)  # (batch, seq_len, d_model)

        # Positional embeddings
        positions = torch.arange(0, seq_len, device=device).unsqueeze(0)  # (1, seq_len)
        pos_embeds = self.pos_embed(positions)  # (1, seq_len, d_model)

        # Segment embeddings
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        segment_embeds = self.segment_embed(token_type_ids)  # (batch, seq_len, d_model)

        # Combine embeddings
        x = token_embeds + pos_embeds + segment_embeds
        x = self.dropout(x)

        # Create attention mask for transformer
        # IMPORTANT: BERT is bidirectional, so NO causal mask!
        if attention_mask is not None:
            # Convert attention mask to shape (batch, 1, 1, seq_len)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # Invert mask: 1 -> can attend, 0 -> cannot attend
            # For masking, we want 0 -> -inf, so we need to convert
            attention_mask = attention_mask.to(dtype=x.dtype)
        else:
            # No mask = attend to all positions
            attention_mask = None

        # Transformer blocks
        for block in self.blocks:
            x, _ = block(x, mask=attention_mask)

        # Final layer norm
        sequence_output = self.ln_f(x)  # (batch, seq_len, d_model)

        # Pool [CLS] token (first token)
        cls_output = sequence_output[:, 0]  # (batch, d_model)
        pooled_output = self.pooler(cls_output)
        pooled_output = self.pooler_activation(pooled_output)

        return sequence_output, pooled_output

    def get_cls_representation(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get [CLS] token representation for classification/encoding.

        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len) optional

        Returns:
            cls_representation: (batch_size, d_model)
        """
        _, pooled_output = self.forward(input_ids, attention_mask)
        return pooled_output


# Export
__all__ = ['QuantumBERT']
