"""
Ternary Bit-Packer for Information-Geometric Quantum Compression (IGQK).

Packs ternary weights {-1, 0, +1} into 2-bit representations:
  -1 -> 0b00 (0)
   0 -> 0b01 (1)
  +1 -> 0b10 (2)

Packs 4 ternary weights per 8-bit byte, achieving up to 16x compression
over FP32 and 8x over FP16.
"""

import numpy as np
import torch
from typing import Tuple, Union


class TernaryBitPacker:
    """High-performance bit-packer and unpacker for ternary tensors."""

    @staticmethod
    def pack(ternary_tensor: Union[torch.Tensor, np.ndarray]) -> Tuple[bytes, Tuple[int, ...], float]:
        """
        Packs a ternary tensor containing values in {-1, 0, +1} into raw packed bytes.

        Returns:
            packed_bytes: Byte string containing packed 2-bit words
            original_shape: Shape of original tensor
            compression_ratio: Relative storage saving over FP32
        """
        if isinstance(ternary_tensor, torch.Tensor):
            arr = ternary_tensor.detach().cpu().numpy().flatten().astype(np.int8)
            orig_shape = tuple(ternary_tensor.shape)
        else:
            arr = np.asarray(ternary_tensor).flatten().astype(np.int8)
            orig_shape = ternary_tensor.shape

        # Map {-1, 0, +1} -> {0, 1, 2}
        mapped = np.clip(arr + 1, 0, 2).astype(np.uint8)

        total_elements = mapped.size
        # Pad to multiple of 4
        remainder = total_elements % 4
        if remainder != 0:
            pad_len = 4 - remainder
            mapped = np.pad(mapped, (0, pad_len), constant_values=1) # 1 maps to 0

        # Reshape into groups of 4 and pack
        grouped = mapped.reshape(-1, 4)
        # Byte = (w0 << 6) | (w1 << 4) | (w2 << 2) | w3
        packed = (
            (grouped[:, 0] << 6) |
            (grouped[:, 1] << 4) |
            (grouped[:, 2] << 2) |
            grouped[:, 3]
        ).astype(np.uint8)

        raw_bytes = packed.tobytes()
        original_bytes_fp32 = total_elements * 4
        packed_bytes_len = len(raw_bytes)
        ratio = original_bytes_fp32 / max(1, packed_bytes_len)

        return raw_bytes, orig_shape, ratio

    @staticmethod
    def unpack(packed_bytes: bytes, original_shape: Tuple[int, ...]) -> torch.Tensor:
        """
        Unpacks packed bytes back into a ternary float32 tensor of `original_shape`.
        """
        packed = np.frombuffer(packed_bytes, dtype=np.uint8)
        num_elements = int(np.prod(original_shape))

        # Extract 2-bit chunks
        w0 = (packed >> 6) & 0x03
        w1 = (packed >> 4) & 0x03
        w2 = (packed >> 2) & 0x03
        w3 = packed & 0x03

        unpacked = np.column_stack([w0, w1, w2, w3]).flatten()
        # Map {0, 1, 2} -> {-1.0, 0.0, +1.0}
        unpacked = (unpacked[:num_elements].astype(np.float32) - 1.0)

        tensor = torch.from_numpy(unpacked.reshape(original_shape))
        return tensor
