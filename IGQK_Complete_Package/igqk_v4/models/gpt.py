"""
Quantum-Enhanced GPT Model.

Implements a GPT-style autoregressive transformer with optional ternary weight compression.
Uses TernaryLinear from theory.tlgt when train_compressed=True.
"""

import torch
import torch.nn as nn
from typing import Optional
import math

from ..quantum_training.trainers.quantum_training_config import QuantumTrainingConfig


class QuantumMultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention with optional ternary weights."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, use_ternary: bool = False):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Choose Linear layer type based on ternary mode
        if use_ternary:
            try:
                from ..theory.tlgt.ternary_lie_group import TernaryLinear
                LinearLayer = TernaryLinear
            except ImportError:
                print("Warning:  TernaryLinear not found, falling back to nn.Linear")
                LinearLayer = nn.Linear
        else:
            LinearLayer = nn.Linear

        # Q, K, V projections
        self.q_proj = LinearLayer(d_model, d_model)
        self.k_proj = LinearLayer(d_model, d_model)
        self.v_proj = LinearLayer(d_model, d_model)
        self.out_proj = LinearLayer(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Forward pass.

        Args:
            x: (batch_size, seq_len, d_model)
            mask: (batch_size, 1, seq_len, seq_len) or (1, 1, seq_len, seq_len)

        Returns:
            output: (batch_size, seq_len, d_model)
            attention_weights: (batch_size, n_heads, seq_len, seq_len)
        """
        batch_size, seq_len, d_model = x.shape

        # Project to Q, K, V
        Q = self.q_proj(x)  # (batch, seq_len, d_model)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape for multi-head: (batch, n_heads, seq_len, head_dim)
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply mask (for causal attention)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to V
        attn_output = torch.matmul(attn_weights, V)  # (batch, n_heads, seq_len, head_dim)

        # Reshape back to (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, d_model)

        # Output projection
        output = self.out_proj(attn_output)

        return output, attn_weights


class QuantumFeedForward(nn.Module):
    """Feed-Forward Network with optional ternary weights."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, use_ternary: bool = False):
        super().__init__()

        # Choose Linear layer type
        if use_ternary:
            try:
                from ..theory.tlgt.ternary_lie_group import TernaryLinear
                LinearLayer = TernaryLinear
            except ImportError:
                LinearLayer = nn.Linear
        else:
            LinearLayer = nn.Linear

        self.fc1 = LinearLayer(d_model, d_ff)
        self.fc2 = LinearLayer(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (batch_size, seq_len, d_model)

        Returns:
            output: (batch_size, seq_len, d_model)
        """
        x = self.fc1(x)
        x = torch.nn.functional.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class QuantumTransformerBlock(nn.Module):
    """Single Transformer Block with Pre-LN architecture."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1, use_ternary: bool = False):
        super().__init__()

        self.attention = QuantumMultiHeadAttention(d_model, n_heads, dropout, use_ternary)
        self.feed_forward = QuantumFeedForward(d_model, d_ff, dropout, use_ternary)

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Forward pass with Pre-LN (like GPT-3).

        Args:
            x: (batch_size, seq_len, d_model)
            mask: Attention mask

        Returns:
            output: (batch_size, seq_len, d_model)
            attention_weights: Attention weights
        """
        # Self-Attention with residual (Pre-LN)
        attn_output, attn_weights = self.attention(self.ln1(x), mask)
        x = x + self.dropout(attn_output)

        # Feed-Forward with residual (Pre-LN)
        ff_output = self.feed_forward(self.ln2(x))
        x = x + self.dropout(ff_output)

        return x, attn_weights


class QuantumGPT(nn.Module):
    """
    Quantum-Enhanced GPT Model.

    Features:
    - Standard GPT architecture (autoregressive transformer decoder)
    - Optional ternary weight compression
    - Compatible with Quantum Gradient Flow training
    - HLWT/TLGT/FCHL integration support

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

        # Transformer blocks
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

        # Language model head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Tie weights (standard GPT practice)
        self.lm_head.weight = self.token_embed.weight

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

        # Initialize weights
        self.apply(self._init_weights)

        # Count parameters
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f" QuantumGPT initialized:")
        print(f"   Layers: {config.n_layers}, Heads: {config.n_heads}, d_model: {config.d_model}")
        print(f"   Ternary: {use_ternary}")
        print(f"   Parameters: {n_params:,}")

    def _init_weights(self, module):
        """Initialize weights (GPT-2 style)."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        Forward pass.

        Args:
            input_ids: (batch_size, seq_len) Long tensor of token indices
            attention_mask: (batch_size, seq_len) optional attention mask

        Returns:
            logits: (batch_size, seq_len, vocab_size) next-token predictions
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Token embeddings
        token_embeds = self.token_embed(input_ids)  # (batch, seq_len, d_model)

        # Positional embeddings
        positions = torch.arange(0, seq_len, device=device).unsqueeze(0)  # (1, seq_len)
        pos_embeds = self.pos_embed(positions)  # (1, seq_len, d_model)

        # Combine embeddings
        x = token_embeds + pos_embeds
        x = self.dropout(x)

        # Create causal mask (GPT-style autoregressive)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        causal_mask = causal_mask.view(1, 1, seq_len, seq_len)  # (1, 1, seq_len, seq_len)

        # Combine with attention mask if provided
        if attention_mask is not None:
            attention_mask = attention_mask.view(batch_size, 1, 1, seq_len)
            causal_mask = causal_mask * attention_mask

        # Transformer blocks
        for block in self.blocks:
            x, _ = block(x, mask=causal_mask)

        # Final layer norm
        x = self.ln_f(x)

        # Language model head
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)

        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.

        Args:
            input_ids: (batch_size, seq_len) starting tokens
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature (higher = more random)
            top_k: if set, only sample from top k tokens

        Returns:
            generated: (batch_size, seq_len + max_new_tokens) generated sequence
        """
        self.eval()

        for _ in range(max_new_tokens):
            # Crop context if too long
            if input_ids.size(1) > self.config.max_seq_len:
                input_ids = input_ids[:, -self.config.max_seq_len:]

            # Get logits for next token
            logits = self.forward(input_ids)  # (batch, seq_len, vocab_size)
            logits = logits[:, -1, :] / temperature  # (batch, vocab_size)

            # Top-k sampling
            if top_k is not None:
                top_logits, top_indices = torch.topk(logits, top_k)
                logits = torch.full_like(logits, -float('inf'))
                logits.scatter_(1, top_indices, top_logits)

            # Sample next token
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (batch, 1)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


# Export
__all__ = ['QuantumGPT', 'QuantumTransformerBlock']
