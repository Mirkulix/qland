"""
Test Integration between Models and QuantumLLMTrainer.

Verifies that QuantumLLMTrainer can instantiate and work with the models.
"""

import torch
import sys

from igqk_v4.quantum_training.trainers.quantum_training_config import QuantumTrainingConfig
from igqk_v4.quantum_training.trainers.quantum_llm_trainer import QuantumLLMTrainer


def test_gpt_trainer_integration():
    """Test QuantumLLMTrainer with QuantumGPT."""
    print("\n" + "="*60)
    print("INTEGRATION TEST: QuantumLLMTrainer + QuantumGPT")
    print("="*60)

    config = QuantumTrainingConfig(
        model_type='GPT',
        n_layers=2,
        n_heads=2,
        d_model=64,
        d_ff=256,
        vocab_size=100,
        max_seq_len=32,
        learning_rate=0.001,
        train_compressed=False,
    )

    try:
        trainer = QuantumLLMTrainer(config)
        print("\n[PASS] QuantumLLMTrainer successfully initialized with QuantumGPT")
        print(f"       Model type: {type(trainer.model).__name__}")
        return True
    except Exception as e:
        print(f"\n[FAIL] Failed to initialize trainer: {e}")
        return False


def test_bert_trainer_integration():
    """Test QuantumLLMTrainer with QuantumBERT."""
    print("\n" + "="*60)
    print("INTEGRATION TEST: QuantumLLMTrainer + QuantumBERT")
    print("="*60)

    config = QuantumTrainingConfig(
        model_type='BERT',
        n_layers=2,
        n_heads=2,
        d_model=64,
        d_ff=256,
        vocab_size=100,
        max_seq_len=32,
        learning_rate=0.001,
        train_compressed=False,
    )

    try:
        trainer = QuantumLLMTrainer(config)
        print("\n[PASS] QuantumLLMTrainer successfully initialized with QuantumBERT")
        print(f"       Model type: {type(trainer.model).__name__}")
        return True
    except Exception as e:
        print(f"\n[FAIL] Failed to initialize trainer: {e}")
        return False


def test_vit_trainer_integration():
    """Test QuantumLLMTrainer with QuantumViT."""
    print("\n" + "="*60)
    print("INTEGRATION TEST: QuantumLLMTrainer + QuantumViT")
    print("="*60)

    config = QuantumTrainingConfig(
        model_type='ViT',
        n_layers=2,
        n_heads=2,
        d_model=64,
        d_ff=256,
        learning_rate=0.001,
        train_compressed=False,
    )

    # Add ViT-specific parameters
    config.img_size = 224
    config.patch_size = 16
    config.in_channels = 3
    config.num_classes = 10

    try:
        trainer = QuantumLLMTrainer(config)
        print("\n[PASS] QuantumLLMTrainer successfully initialized with QuantumViT")
        print(f"       Model type: {type(trainer.model).__name__}")
        return True
    except Exception as e:
        print(f"\n[FAIL] Failed to initialize trainer: {e}")
        return False


def main():
    """Run all integration tests."""
    print("\n" + "="*60)
    print("IGQK v4.0 INTEGRATION TESTS")
    print("="*60)

    results = []

    try:
        results.append(("GPT Integration", test_gpt_trainer_integration()))
    except Exception as e:
        print(f"[FAIL] GPT Integration failed: {e}")
        results.append(("GPT Integration", False))

    try:
        results.append(("BERT Integration", test_bert_trainer_integration()))
    except Exception as e:
        print(f"[FAIL] BERT Integration failed: {e}")
        results.append(("BERT Integration", False))

    try:
        results.append(("ViT Integration", test_vit_trainer_integration()))
    except Exception as e:
        print(f"[FAIL] ViT Integration failed: {e}")
        results.append(("ViT Integration", False))

    # Summary
    print("\n" + "="*60)
    print("INTEGRATION TEST SUMMARY")
    print("="*60)

    for name, passed in results:
        status = "[PASS] PASSED" if passed else "[FAIL] FAILED"
        print(f"{name:25s} {status}")

    total_tests = len(results)
    passed_tests = sum(1 for _, passed in results if passed)

    print("\n" + "="*60)
    print(f"TOTAL: {passed_tests}/{total_tests} tests passed")
    print("="*60)

    if passed_tests == total_tests:
        print("\n[SUCCESS] ALL INTEGRATION TESTS PASSED!")
        print("\n[INFO] Models successfully integrate with QuantumLLMTrainer")
        print("       Phase 1 is 100% complete!")
        return True
    else:
        print("\n[WARNING] Some integration tests failed.")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
