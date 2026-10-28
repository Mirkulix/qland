"""Bit-exact (lossless) model compression for IGQK.

Ternary 2-bit packing is lossy: it replaces each FP32 weight by -1/0/+1. When no
accuracy loss is acceptable, this module stores the exact state_dict instead:
`torch.save` bytes compressed with zstd. The round trip is bit-exact by
construction; the ratio is lower than 16x and depends on the weights.

The file is a zstd frame whose decompressed payload is a `torch.save` state_dict.
`load` requires the same architecture (the state_dict is loaded into it).
"""

from io import BytesIO
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
import zstandard as zstd

MAGIC = b"QALL1\n"
LEVEL = 19


class LosslessCompressor:
    """Exact state_dict compression: no quantization, no accuracy change."""

    @staticmethod
    def model_bytes(model: nn.Module) -> int:
        """Bytes of the FP32 parameters and buffers (the uncompressed reference)."""
        return sum(t.numel() * t.element_size() for t in model.state_dict().values())

    @staticmethod
    def dumps(model: nn.Module) -> bytes:
        """Return a zstd frame containing the exact state_dict of `model`."""
        buf = BytesIO()
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        torch.save(state, buf)
        packed = zstd.ZstdCompressor(level=LEVEL).compress(buf.getvalue())
        return MAGIC + packed

    @staticmethod
    def loads(model: nn.Module, blob: bytes) -> nn.Module:
        """Load an exact state_dict into `model`; raises on a foreign format."""
        if not blob.startswith(MAGIC):
            raise ValueError("not a QAL lossless blob")
        raw = zstd.ZstdDecompressor().decompress(blob[len(MAGIC):])
        state = torch.load(BytesIO(raw), map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        return model

    @staticmethod
    def save(model: nn.Module, path: Union[str, Path]) -> int:
        """Write the lossless blob; return its size in bytes."""
        blob = LosslessCompressor.dumps(model)
        Path(path).write_bytes(blob)
        return len(blob)

    @staticmethod
    def load(model: nn.Module, path: Union[str, Path]) -> nn.Module:
        return LosslessCompressor.loads(model, Path(path).read_bytes())

    @staticmethod
    def ratio(model: nn.Module, blob: bytes) -> float:
        """Uncompressed FP32 bytes divided by the blob size."""
        return LosslessCompressor.model_bytes(model) / len(blob)
