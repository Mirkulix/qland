"""
Quantum-Enhanced Vision Transformer (ViT).

Implements a Vision Transformer with optional ternary weight compression.
Processes images as sequences of patches.
"""

import torch
import torch.nn as nn
from typing import Optional

from ..quantum_training.trainers.quantum_training_config import QuantumTrainingConfig
from .gpt import QuantumTransformerBlock  # Reuse from GPT!


class PatchEmbedding(nn.Module):
    """
    Convert image into patches and embed them.

    Image (3, 224, 224) -> Patches (196, d_model) for 16x16 patches
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        d_model: int = 768,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        # Conv2d with kernel=patch_size, stride=patch_size creates non-overlapping patches
        self.projection = nn.Conv2d(
            in_channels,
            d_model,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, channels, height, width)

        Returns:
            patches: (batch_size, n_patches, d_model)
        """
        batch_size = x.shape[0]

        # Project patches: (batch, d_model, H/P, W/P)
        x = self.projection(x)

        # Flatten spatial dimensions: (batch, d_model, n_patches)
        x = x.flatten(2)

        # Transpose to (batch, n_patches, d_model)
        x = x.transpose(1, 2)

        return x


class QuantumViT(nn.Module):
    """
    Quantum-Enhanced Vision Transformer.

    Features:
    - Patch-based image processing
    - [CLS] token for image classification
    - Positional embeddings for patches
    - Bidirectional transformer encoder (like BERT)
    - Optional ternary weight compression
    - Compatible with Quantum Gradient Flow training

    Args:
        config: QuantumTrainingConfig with model parameters
    """

    def __init__(self, config: QuantumTrainingConfig):
        super().__init__()

        self.config = config
        use_ternary = config.train_compressed

        # Image parameters
        self.img_size = getattr(config, 'img_size', 224)
        self.patch_size = getattr(config, 'patch_size', 16)
        self.in_channels = getattr(config, 'in_channels', 3)

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            d_model=config.d_model
        )

        n_patches = self.patch_embed.n_patches

        # [CLS] token (learnable parameter)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_model))

        # Positional embeddings (for n_patches + 1 [CLS] token)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, config.d_model))

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

        # Classification head
        self.head = nn.Linear(config.d_model, getattr(config, 'num_classes', 1000))

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

        # Initialize weights
        self.apply(self._init_weights)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)

        # Count parameters
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f" QuantumViT initialized:")
        print(f"   Layers: {config.n_layers}, Heads: {config.n_heads}, d_model: {config.d_model}")
        print(f"   Image: {self.img_size}x{self.img_size}, Patches: {self.patch_size}x{self.patch_size}")
        print(f"   N_patches: {n_patches}")
        print(f"   Ternary: {use_ternary}")
        print(f"   Parameters: {n_params:,}")

    def _init_weights(self, module):
        """Initialize weights (ViT style)."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight, mode='fan_out')
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        images: torch.Tensor,
        return_features: bool = False,
    ):
        """
        Forward pass.

        Args:
            images: (batch_size, channels, height, width) Image tensor
            return_features: If True, return [CLS] features instead of logits

        Returns:
            If return_features=False:
                logits: (batch_size, num_classes) Classification logits
            If return_features=True:
                features: (batch_size, d_model) [CLS] token features
        """
        batch_size = images.shape[0]
        device = images.device

        # Convert image to patches: (batch, n_patches, d_model)
        x = self.patch_embed(images)

        # Expand [CLS] token: (1, 1, d_model) -> (batch, 1, d_model)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        # Prepend [CLS] token: (batch, n_patches+1, d_model)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional embeddings
        x = x + self.pos_embed
        x = self.dropout(x)

        # No attention mask needed - ViT is bidirectional!
        # (Unlike GPT which needs causal mask)

        # Transformer blocks
        for block in self.blocks:
            x, _ = block(x, mask=None)

        # Final layer norm
        x = self.ln_f(x)

        # Extract [CLS] token: (batch, d_model)
        cls_output = x[:, 0]

        # Return features or classification logits
        if return_features:
            return cls_output
        else:
            logits = self.head(cls_output)
            return logits

    def get_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Get [CLS] token features for image encoding.

        Args:
            images: (batch_size, channels, height, width)

        Returns:
            features: (batch_size, d_model) Image features
        """
        return self.forward(images, return_features=True)


# Export
__all__ = ['QuantumViT']
