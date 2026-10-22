"""
LLM Weight Compressor using IGQK (Informationsgeometrische Quantenkompression).

Implements projection Pi_N: M -> N onto the low-dimensional ternary submanifold:
    N = { W : W in {-alpha, 0, +alpha} }

Calculates optimal threshold Delta and scale factor alpha preserving
first-order gradient dynamics and information geometry on the statistical manifold.
"""

import math
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
from .ternary_packer import TernaryBitPacker


class LLMWeightCompressor:
    """Compresses Transformer / LLM layers into ternary representations with IGQK scaling."""

    @staticmethod
    def quantize_matrix(weight: torch.Tensor, threshold_ratio: float = 0.7) -> Tuple[torch.Tensor, float]:
        """
        Projects weight matrix W onto ternary manifold {-alpha, 0, +alpha}.

        Returns:
            ternary_w: Tensor with elements in {-1, 0, +1}
            alpha: Optimal scaling factor
        """
        with torch.no_grad():
            w = weight.detach()
            abs_w = torch.abs(w)

            # Threshold Delta based on mean absolute weight
            delta = threshold_ratio * torch.mean(abs_w)

            # Quantize to {-1, 0, +1}
            ternary = torch.zeros_like(w)
            pos_mask = w > delta
            neg_mask = w < -delta
            ternary[pos_mask] = 1.0
            ternary[neg_mask] = -1.0

            # Compute scale factor alpha = E[|W|] over non-zero elements
            non_zero_count = (pos_mask | neg_mask).sum().item()
            if non_zero_count > 0:
                alpha = abs_w[pos_mask | neg_mask].mean().item()
            else:
                alpha = 1.0

            return ternary, alpha

    @staticmethod
    def compress_layer(layer: nn.Linear, name: str = "linear") -> Dict[str, Any]:
        """
        Compresses a PyTorch Linear layer into packed ternary representation.
        """
        w = layer.weight.data
        ternary, alpha = LLMWeightCompressor.quantize_matrix(w)
        packed_bytes, shape, comp_ratio = TernaryBitPacker.pack(ternary)

        compressed_data = {
            "name": name,
            "shape": shape,
            "alpha": alpha,
            "packed_weights": packed_bytes,
            "bias": layer.bias.data.clone() if layer.bias is not None else None,
            "compression_ratio": comp_ratio,
            "in_features": layer.in_features,
            "out_features": layer.out_features,
        }
        return compressed_data

    @staticmethod
    def decompress_layer(compressed_data: Dict[str, Any]) -> nn.Linear:
        """
        Reconstructs a PyTorch Linear layer from compressed ternary representation.
        """
        shape = compressed_data["shape"]
        alpha = compressed_data["alpha"]
        packed_bytes = compressed_data["packed_weights"]

        unpacked_ternary = TernaryBitPacker.unpack(packed_bytes, shape)
        reconstructed_weight = unpacked_ternary * alpha

        layer = nn.Linear(
            compressed_data["in_features"],
            compressed_data["out_features"],
            bias=compressed_data["bias"] is not None
        )
        layer.weight.data.copy_(reconstructed_weight)
        if compressed_data["bias"] is not None:
            layer.bias.data.copy_(compressed_data["bias"])

        return layer

    @staticmethod
    def compress_model(model: nn.Module) -> Dict[str, Any]:
        """
        Compresses all Linear layers in a neural network model.
        """
        compressed_layers = {}
        total_orig_bytes = 0
        total_comp_bytes = 0

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                comp = LLMWeightCompressor.compress_layer(module, name=name)
                compressed_layers[name] = comp

                orig_bytes = module.weight.numel() * 4
                comp_bytes = len(comp["packed_weights"]) + 8 # + float64 alpha
                total_orig_bytes += orig_bytes
                total_comp_bytes += comp_bytes

        total_ratio = total_orig_bytes / max(1, total_comp_bytes)
        return {
            "layers": compressed_layers,
            "total_original_bytes": total_orig_bytes,
            "total_compressed_bytes": total_comp_bytes,
            "global_compression_ratio": total_ratio,
        }
