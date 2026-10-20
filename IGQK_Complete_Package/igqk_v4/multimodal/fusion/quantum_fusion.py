"""
Quantum Multi-Modal Fusion.

Fuses vision and language representations using quantum entanglement principles,
unitary quantum gates, or cross-modal attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from ...quantum_training.trainers.quantum_training_config import QuantumTrainingConfig


class QuantumMultiModalFusion(nn.Module):
    """
    Quantum Multi-Modal Fusion module.

    Fuses vision_features and language_features into a coherent joint representation.
    Supports strategies:
    - 'quantum_entanglement': Unitary quantum gate mixing + cross-modal entanglement
    - 'cross_attention': Bidirectional cross-modal attention
    - 'concat': Concatenation followed by MLP projection
    """

    def __init__(self, config: QuantumTrainingConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.strategy = getattr(config, 'multimodal_fusion', 'quantum_entanglement')

        if self.strategy == 'quantum_entanglement':
            # Parameterize unitary rotation angle / skew-symmetric generator
            self.theta = nn.Parameter(torch.tensor(0.785398))  # pi / 4 (equal superposition initially)
            self.gate_v = nn.Linear(self.d_model, self.d_model, bias=False)
            self.gate_l = nn.Linear(self.d_model, self.d_model, bias=False)
            self.entangle_proj = nn.Sequential(
                nn.Linear(self.d_model * 2, self.d_model),
                nn.GELU(),
                nn.LayerNorm(self.d_model),
                nn.Linear(self.d_model, self.d_model),
            )
            self.ln = nn.LayerNorm(self.d_model)

        elif self.strategy == 'cross_attention':
            self.cross_attn_v2l = nn.MultiheadAttention(self.d_model, config.n_heads, batch_first=True)
            self.cross_attn_l2v = nn.MultiheadAttention(self.d_model, config.n_heads, batch_first=True)
            self.out_proj = nn.Linear(self.d_model * 2, self.d_model)
            self.ln = nn.LayerNorm(self.d_model)

        else:  # 'concat' fallback
            self.mlp = nn.Sequential(
                nn.Linear(self.d_model * 2, self.d_model * 2),
                nn.GELU(),
                nn.Linear(self.d_model * 2, self.d_model),
                nn.LayerNorm(self.d_model),
            )

    def forward(
        self,
        vision_features: torch.Tensor,
        language_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse vision and language representations.

        Args:
            vision_features: (batch_size, d_model)
            language_features: (batch_size, d_model)

        Returns:
            fused: (batch_size, d_model) joint representation
        """
        assert vision_features.shape == language_features.shape, (
            f"Dimension mismatch: {vision_features.shape} vs {language_features.shape}"
        )

        if self.strategy == 'quantum_entanglement':
            # Unitary rotation superposition: |psi> = cos(theta)|v> + sin(theta)|l>
            cos_t = torch.cos(self.theta)
            sin_t = torch.sin(self.theta)

            # Quantum state phase evolution
            psi_1 = cos_t * self.gate_v(vision_features) + sin_t * self.gate_l(language_features)
            psi_2 = -sin_t * self.gate_v(vision_features) + cos_t * self.gate_l(language_features)

            # Entangled joint tensor projection
            joint = torch.cat([psi_1, psi_2], dim=-1)
            entangled = self.entangle_proj(joint)
            fused = self.ln(vision_features + language_features + entangled)
            return fused

        elif self.strategy == 'cross_attention':
            v_seq = vision_features.unsqueeze(1)
            l_seq = language_features.unsqueeze(1)

            v_attn, _ = self.cross_attn_v2l(v_seq, l_seq, l_seq)
            l_attn, _ = self.cross_attn_l2v(l_seq, v_seq, v_seq)

            combined = torch.cat([v_attn.squeeze(1), l_attn.squeeze(1)], dim=-1)
            fused = self.ln(self.out_proj(combined))
            return fused

        else:  # 'concat'
            combined = torch.cat([vision_features, language_features], dim=-1)
            return self.mlp(combined)
