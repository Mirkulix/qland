"""
Tests for IGQK compression and 2-bit ternary packing.
"""

import torch
import torch.nn as nn
try:
    from igqk_v4.compression import TernaryBitPacker, LLMWeightCompressor
except ModuleNotFoundError:
    from IGQK_Complete_Package.igqk_v4.compression import TernaryBitPacker, LLMWeightCompressor


def test_ternary_bit_packer_roundtrip():
    """Verify that bit-packing and unpacking of ternary values {-1, 0, +1} is 100% exact."""
    tensor = torch.tensor([
        [-1.0,  0.0,  1.0, -1.0],
        [ 0.0,  1.0, -1.0,  0.0],
        [ 1.0, -1.0,  0.0,  1.0]
    ], dtype=torch.float32)

    packed_bytes, orig_shape, ratio = TernaryBitPacker.pack(tensor)

    # 12 elements * 2 bits = 24 bits = 3 bytes
    assert len(packed_bytes) == 3, f"Expected 3 packed bytes, got {len(packed_bytes)}"
    assert ratio >= 15.0, f"Expected compression ratio >= 15x over FP32, got {ratio}"

    unpacked = TernaryBitPacker.unpack(packed_bytes, orig_shape)
    assert torch.equal(tensor, unpacked), "Unpacked tensor does not match original ternary tensor!"


def test_llm_layer_compression_and_decompression():
    """Verify that an arbitrary PyTorch linear layer can be quantized, packed, and decompressed."""
    torch.manual_seed(42)
    in_features = 128
    out_features = 256
    layer = nn.Linear(in_features, out_features)

    compressed = LLMWeightCompressor.compress_layer(layer, name="attn_q")

    assert compressed["name"] == "attn_q"
    assert compressed["shape"] == (out_features, in_features)
    assert compressed["alpha"] > 0.0
    assert compressed["compression_ratio"] >= 15.0

    decompressed_layer = LLMWeightCompressor.decompress_layer(compressed)

    # Decompressed weights must be strictly in {-alpha, 0, +alpha}
    w = decompressed_layer.weight.data
    alpha = compressed["alpha"]
    valid_values = torch.isclose(w, torch.tensor(-alpha), atol=1e-5) | \
                   torch.isclose(w, torch.tensor(0.0), atol=1e-5) | \
                   torch.isclose(w, torch.tensor(alpha), atol=1e-5)
    assert valid_values.all(), "Reconstructed weights contain values outside {-alpha, 0, +alpha}!"


def test_full_model_compression_pipeline():
    """Test full multi-layer model compression."""
    model = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 64)
    )

    stats = LLMWeightCompressor.compress_model(model)
    assert "layers" in stats
    assert len(stats["layers"]) == 2
    assert stats["global_compression_ratio"] >= 15.0
    assert stats["total_compressed_bytes"] < stats["total_original_bytes"] / 10


def test_lossless_roundtrip_is_bit_exact():
    """A lossless blob restores every parameter exactly (no accuracy change)."""
    try:
        from igqk_v4.compression import LosslessCompressor
    except ModuleNotFoundError:
        from IGQK_Complete_Package.igqk_v4.compression import LosslessCompressor

    torch.manual_seed(7)
    model = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 4))
    before = {k: v.clone() for k, v in model.state_dict().items()}
    blob = LosslessCompressor.dumps(model)
    restored = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 4))
    LosslessCompressor.loads(restored, blob)
    for key, value in restored.state_dict().items():
        assert torch.equal(value, before[key]), f"{key} changed"
    assert LosslessCompressor.ratio(model, blob) >= 1.0
