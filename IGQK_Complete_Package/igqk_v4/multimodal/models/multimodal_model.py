"""
Multi-Modal Model for IGQK v4.0.

Integrates QuantumVisionEncoder, QuantumLanguageEncoder, and QuantumMultiModalFusion
into an end-to-end multi-modal architecture with CLIP-style contrastive learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from ...quantum_training.trainers.quantum_training_config import QuantumTrainingConfig
from ..vision.vision_encoder import QuantumVisionEncoder
from ..language.language_encoder import QuantumLanguageEncoder
from ..fusion.quantum_fusion import QuantumMultiModalFusion


class MultiModalModel(nn.Module):
    """
    Unified Multi-Modal Model combining Quantum Vision, Language, and Fusion.

    Features:
    - Vision encoder (ViT-based)
    - Language encoder (BERT-based)
    - Quantum entanglement fusion module
    - Classification head
    - CLIP-style symmetric contrastive loss
    """

    def __init__(self, config: QuantumTrainingConfig):
        super().__init__()
        self.config = config

        self.vision_encoder = QuantumVisionEncoder(config)
        self.language_encoder = QuantumLanguageEncoder(config)
        self.fusion = QuantumMultiModalFusion(config)

        # Classification / generation head
        num_classes = getattr(config, 'num_classes', config.vocab_size)
        self.classifier = nn.Linear(config.d_model, num_classes)

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f" MultiModalModel initialized:")
        print(f"   Vision: ViT, Language: BERT, Fusion: {config.multimodal_fusion}")
        print(f"   d_model: {config.d_model}, Parameters: {n_params:,}")

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_logits: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through vision, language, and fusion.

        Args:
            images: (batch_size, 3, 224, 224) image tensor
            input_ids: (batch_size, seq_len) text token IDs
            attention_mask: optional attention mask
            return_logits: if True, returns classification logits; else fused representation

        Returns:
            fused: (batch_size, d_model) or logits: (batch_size, num_classes)
        """
        v_feat = self.vision_encoder(images)
        l_feat = self.language_encoder(input_ids, attention_mask=attention_mask)
        fused = self.fusion(v_feat, l_feat)

        if return_logits:
            return self.classifier(fused)
        return fused

    def contrastive_loss(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        temperature: float = 0.07,
    ) -> torch.Tensor:
        """
        Computes symmetric CLIP-style contrastive loss between image and text features.

        Args:
            images: (batch_size, 3, 224, 224)
            input_ids: (batch_size, seq_len)
            attention_mask: optional attention mask
            temperature: softmax temperature scaling factor

        Returns:
            Scalar cross-entropy loss tensor
        """
        v_feat = self.vision_encoder(images)
        l_feat = self.language_encoder(input_ids, attention_mask=attention_mask)

        # Normalize features to unit sphere (L2 norm)
        v_norm = F.normalize(v_feat, dim=-1)
        l_norm = F.normalize(l_feat, dim=-1)

        # Compute cosine similarity matrix: (batch_size, batch_size)
        logits_per_image = torch.matmul(v_norm, l_norm.T) / temperature
        logits_per_text = logits_per_image.T

        batch_size = images.shape[0]
        labels = torch.arange(batch_size, device=images.device)

        loss_img = F.cross_entropy(logits_per_image, labels)
        loss_txt = F.cross_entropy(logits_per_text, labels)

        return (loss_img + loss_txt) / 2.0

    def compute_loss(self, batch, temperature: float = 0.07) -> torch.Tensor:
        """
        Computes training loss on a multi-modal batch.

        Args:
            batch: dict with 'images' and 'input_ids', or tuple (images, input_ids)
            temperature: softmax temperature scaling factor
        """
        if isinstance(batch, dict):
            images = batch['images']
            input_ids = batch['input_ids']
            attention_mask = batch.get('attention_mask', None)
        elif isinstance(batch, (list, tuple)):
            images = batch[0]
            input_ids = batch[1]
            attention_mask = batch[2] if len(batch) > 2 else None
        else:
            raise ValueError(f"Unsupported batch format for MultiModalModel: {type(batch)}")

        return self.contrastive_loss(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            temperature=temperature,
        )

