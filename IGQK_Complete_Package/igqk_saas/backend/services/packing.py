"""
Storage of ternary-compressed models.

A tensor whose values are alpha * {-1, 0, +1} (what TernaryProjector produces) is stored with 2 bits
per weight (igqk_v4's TernaryBitPacker) plus its alpha; every other tensor (biases, layers that were
not projected) is stored as it is. Sizes are measured on the written file, never derived.
"""

import os
import sys
from typing import Any, Dict, Optional

import torch

# igqk_v4 lives in IGQK_Complete_Package, three levels above this file.
_PACKAGE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PACKAGE not in sys.path:
    sys.path.insert(0, _PACKAGE)

from igqk_v4.compression.ternary_packer import TernaryBitPacker  # noqa: E402

FORMAT = "igqk-ternary-2bit-v1"


def ternary_scale(t: torch.Tensor) -> Optional[float]:
    """alpha if every value of `t` is in alpha * {-1, 0, +1}, else None."""
    if not t.is_floating_point() or t.numel() < 4:
        return None
    values = torch.unique(t.detach().float().cpu())
    if len(values) > 3:
        return None
    nonzero = values[values != 0].abs()
    if len(nonzero) == 0:
        return 0.0
    alpha = float(nonzero[0])
    if not torch.allclose(nonzero, torch.full_like(nonzero, alpha)):
        return None
    return alpha


def save_packed(state_dict: Dict[str, torch.Tensor], path: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Writes `state_dict` to `path`; returns the measured file size and what was packed."""
    packed, dense = {}, {}
    fp32_bytes = 0
    for name, t in state_dict.items():
        if t.is_floating_point():
            fp32_bytes += t.numel() * 4
        else:
            fp32_bytes += t.numel() * t.element_size()
        alpha = ternary_scale(t)
        if alpha is None:
            dense[name] = t.detach().cpu()
            continue
        signs = torch.zeros_like(t) if alpha == 0.0 else torch.round(t.detach().float() / alpha)
        data, shape, _ = TernaryBitPacker.pack(signs)
        # A uint8 tensor, not bytes: torch.save pickles bytes as text, which grows every byte >= 128 to two.
        packed[name] = {"packed": torch.frombuffer(bytearray(data), dtype=torch.uint8), "shape": tuple(shape), "alpha": alpha, "dtype": str(t.dtype)}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"format": FORMAT, "packed": packed, "dense": dense, "metadata": metadata or {}, "fp32_bytes": fp32_bytes}, path)
    return {
        "file_bytes": os.path.getsize(path),
        "fp32_bytes": fp32_bytes,
        "packed_tensors": len(packed),
        "dense_tensors": len(dense),
    }


def is_packed(checkpoint: Any) -> bool:
    return isinstance(checkpoint, dict) and checkpoint.get("format") == FORMAT


def load_packed(checkpoint: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """The state dict of a checkpoint written by `save_packed`."""
    state = {}
    for name, e in checkpoint["packed"].items():
        dtype = getattr(torch, e.get("dtype", "torch.float32").replace("torch.", ""), torch.float32)
        state[name] = (TernaryBitPacker.unpack(e["packed"].numpy().tobytes(), e["shape"]) * e["alpha"]).to(dtype)
    state.update(checkpoint["dense"])
    return state
