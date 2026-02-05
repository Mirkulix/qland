"""
Test IGQK v4.0 Models.

Tests QuantumGPT, QuantumBERT, and QuantumViT in both standard and ternary modes.
"""

import torch
import sys

from igqk_v4.quantum_training.trainers.quantum_training_config import QuantumTrainingConfig
from igqk_v4.models import QuantumGPT, QuantumBERT, QuantumViT


def test_quantum_gpt_standard():
    """Test QuantumGPT in standard mode (no ternary compression)."""
    print("\n" + "="*60)
    print("TEST 1: QuantumGPT (Standard Mode)")
    print("="*60)

    config = QuantumTrainingConfig(
        model_type='GPT',
        n_layers=2,
        n_heads=4,
        d_model=128,
        d_ff=512,
        vocab_size=1000,
        max_seq_len=64,
        dropout=0.1,
        train_compressed=False,  # Standard mode
    )

    model = QuantumGPT(config)
    model.eval()

    # Test forward pass
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    print(f"\n>> Input shape: {input_ids.shape}")

    with torch.no_grad():
        logits = model(input_ids)

    print(f"<< Output shape: {logits.shape}")
    assert logits.shape == (batch_size, seq_len, config.vocab_size), "Wrong output shape!"

    # Test generation
    print(f"\n>> Testing generation (5 tokens)...")
    start_tokens = torch.randint(0, config.vocab_size, (1, 5))

    with torch.no_grad():
        generated = model.generate(start_tokens, max_new_tokens=5, temperature=1.0)

    print(f"<< Generated shape: {generated.shape}")
    assert generated.shape == (1, 10), "Wrong generation shape!"

    print(f"[PASS] QuantumGPT (Standard) - ALL TESTS PASSED!")
    return True


def test_quantum_gpt_ternary():
    """Test QuantumGPT with ternary compression."""
    print("\n" + "="*60)
    print("TEST 2: QuantumGPT (Ternary Mode)")
    print("="*60)

    config = QuantumTrainingConfig(
        model_type='GPT',
        n_layers=2,
        n_heads=4,
        d_model=128,
        d_ff=512,
        vocab_size=1000,
        max_seq_len=64,
        dropout=0.1,
        train_compressed=True,  # Ternary mode
    )

    model = QuantumGPT(config)
    model.eval()

    # Test forward pass
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    print(f"\n>> Input shape: {input_ids.shape}")

    with torch.no_grad():
        logits = model(input_ids)

    print(f"<< Output shape: {logits.shape}")
    assert logits.shape == (batch_size, seq_len, config.vocab_size), "Wrong output shape!"

    print(f"[PASS] QuantumGPT (Ternary) - ALL TESTS PASSED!")
    return True


def test_quantum_bert_standard():
    """Test QuantumBERT in standard mode."""
    print("\n" + "="*60)
    print("TEST 3: QuantumBERT (Standard Mode)")
    print("="*60)

    config = QuantumTrainingConfig(
        model_type='BERT',
        n_layers=2,
        n_heads=4,
        d_model=128,
        d_ff=512,
        vocab_size=1000,
        max_seq_len=64,
        dropout=0.1,
        train_compressed=False,
    )

    model = QuantumBERT(config)
    model.eval()

    # Test forward pass
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)

    print(f"\n>> Input shape: {input_ids.shape}")
    print(f">> Attention mask: {attention_mask.shape}")
    print(f">> Token types: {token_type_ids.shape}")

    with torch.no_grad():
        sequence_output, pooled_output = model(input_ids, attention_mask, token_type_ids)

    print(f"<< Sequence output: {sequence_output.shape}")
    print(f"<< Pooled output (CLS): {pooled_output.shape}")

    assert sequence_output.shape == (batch_size, seq_len, config.d_model), "Wrong sequence shape!"
    assert pooled_output.shape == (batch_size, config.d_model), "Wrong pooled shape!"

    # Test get_cls_representation
    with torch.no_grad():
        cls_rep = model.get_cls_representation(input_ids, attention_mask)

    print(f"<< CLS representation: {cls_rep.shape}")
    assert cls_rep.shape == (batch_size, config.d_model), "Wrong CLS shape!"

    print(f"[PASS] QuantumBERT (Standard) - ALL TESTS PASSED!")
    return True


def test_quantum_bert_ternary():
    """Test QuantumBERT with ternary compression."""
    print("\n" + "="*60)
    print("TEST 4: QuantumBERT (Ternary Mode)")
    print("="*60)

    config = QuantumTrainingConfig(
        model_type='BERT',
        n_layers=2,
        n_heads=4,
        d_model=128,
        d_ff=512,
        vocab_size=1000,
        max_seq_len=64,
        dropout=0.1,
        train_compressed=True,  # Ternary mode
    )

    model = QuantumBERT(config)
    model.eval()

    # Test forward pass
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    print(f"\n>> Input shape: {input_ids.shape}")

    with torch.no_grad():
        sequence_output, pooled_output = model(input_ids)

    print(f"<< Sequence output: {sequence_output.shape}")
    print(f"<< Pooled output (CLS): {pooled_output.shape}")

    assert sequence_output.shape == (batch_size, seq_len, config.d_model), "Wrong sequence shape!"
    assert pooled_output.shape == (batch_size, config.d_model), "Wrong pooled shape!"

    print(f"[PASS] QuantumBERT (Ternary) - ALL TESTS PASSED!")
    return True


def test_quantum_vit_standard():
    """Test QuantumViT in standard mode."""
    print("\n" + "="*60)
    print("TEST 5: QuantumViT (Standard Mode)")
    print("="*60)

    config = QuantumTrainingConfig(
        model_type='ViT',
        n_layers=2,
        n_heads=4,
        d_model=128,
        d_ff=512,
        dropout=0.1,
        train_compressed=False,
    )

    # Add ViT-specific parameters
    config.img_size = 224
    config.patch_size = 16
    config.in_channels = 3
    config.num_classes = 1000

    model = QuantumViT(config)
    model.eval()

    # Test forward pass with images
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224)

    print(f"\n>> Input shape: {images.shape}")

    with torch.no_grad():
        logits = model(images)

    print(f"<< Output shape: {logits.shape}")
    assert logits.shape == (batch_size, config.num_classes), "Wrong output shape!"

    # Test feature extraction
    with torch.no_grad():
        features = model.get_features(images)

    print(f"<< Features shape: {features.shape}")
    assert features.shape == (batch_size, config.d_model), "Wrong features shape!"

    print(f"[PASS] QuantumViT (Standard) - ALL TESTS PASSED!")
    return True


def test_quantum_vit_ternary():
    """Test QuantumViT with ternary compression."""
    print("\n" + "="*60)
    print("TEST 6: QuantumViT (Ternary Mode)")
    print("="*60)

    config = QuantumTrainingConfig(
        model_type='ViT',
        n_layers=2,
        n_heads=4,
        d_model=128,
        d_ff=512,
        dropout=0.1,
        train_compressed=True,  # Ternary mode
    )

    config.img_size = 224
    config.patch_size = 16
    config.in_channels = 3
    config.num_classes = 1000

    model = QuantumViT(config)
    model.eval()

    # Test forward pass
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224)

    print(f"\n>> Input shape: {images.shape}")

    with torch.no_grad():
        logits = model(images)

    print(f"<< Output shape: {logits.shape}")
    assert logits.shape == (batch_size, config.num_classes), "Wrong output shape!"

    print(f"[PASS] QuantumViT (Ternary) - ALL TESTS PASSED!")
    return True


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("IGQK v4.0 MODEL TESTS")
    print("="*60)

    results = []

    try:
        results.append(("GPT Standard", test_quantum_gpt_standard()))
    except Exception as e:
        print(f"[FAIL] GPT Standard FAILED: {e}")
        results.append(("GPT Standard", False))

    try:
        results.append(("GPT Ternary", test_quantum_gpt_ternary()))
    except Exception as e:
        print(f"[FAIL] GPT Ternary FAILED: {e}")
        results.append(("GPT Ternary", False))

    try:
        results.append(("BERT Standard", test_quantum_bert_standard()))
    except Exception as e:
        print(f"[FAIL] BERT Standard FAILED: {e}")
        results.append(("BERT Standard", False))

    try:
        results.append(("BERT Ternary", test_quantum_bert_ternary()))
    except Exception as e:
        print(f"[FAIL] BERT Ternary FAILED: {e}")
        results.append(("BERT Ternary", False))

    try:
        results.append(("ViT Standard", test_quantum_vit_standard()))
    except Exception as e:
        print(f"[FAIL] ViT Standard FAILED: {e}")
        results.append(("ViT Standard", False))

    try:
        results.append(("ViT Ternary", test_quantum_vit_ternary()))
    except Exception as e:
        print(f"[FAIL] ViT Ternary FAILED: {e}")
        results.append(("ViT Ternary", False))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    for name, passed in results:
        status = "[PASS] PASSED" if passed else "[FAIL] FAILED"
        print(f"{name:20s} {status}")

    total_tests = len(results)
    passed_tests = sum(1 for _, passed in results if passed)

    print("\n" + "="*60)
    print(f"TOTAL: {passed_tests}/{total_tests} tests passed")
    print("="*60)

    if passed_tests == total_tests:
        print("\n[SUCCESS] ALL TESTS PASSED! Models are ready!")
        print("\n[PASS] Phase 1 Complete:")
        print("   - QuantumGPT implemented and tested")
        print("   - QuantumBERT implemented and tested")
        print("   - QuantumViT implemented and tested")
        print("   - Standard and ternary modes working")
        print("\n[INFO] Next Steps:")
        print("   - Phase 2: Multi-Modal AI components")
        print("   - Phase 3: Complete QuantumLLMTrainer")
        return True
    else:
        print("\n[WARNING] Some tests failed. Please review errors above.")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
