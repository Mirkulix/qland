"""
Test IGQK v4.0 Multi-Modal AI components.

Tests:
1. QuantumVisionEncoder
2. QuantumLanguageEncoder
3. QuantumMultiModalFusion (entanglement, cross_attention, concat)
4. MultiModalModel (forward and contrastive loss)
5. QuantumLLMTrainer integration with MultiModal
"""

import torch
import sys

from igqk_v4.quantum_training.trainers.quantum_training_config import QuantumTrainingConfig
from igqk_v4.multimodal import (
    QuantumVisionEncoder,
    QuantumLanguageEncoder,
    QuantumMultiModalFusion,
    MultiModalModel,
)


def test_vision_encoder():
    print("\n" + "=" * 60)
    print("TEST 1: QuantumVisionEncoder")
    print("=" * 60)

    config = QuantumTrainingConfig(
        model_type='ViT',
        n_layers=2,
        n_heads=4,
        d_model=128,
        d_ff=512,
        dropout=0.1,
        train_compressed=False,
    )

    encoder = QuantumVisionEncoder(config)
    encoder.eval()

    images = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        features = encoder(images)

    print(f">> Input images: {images.shape}")
    print(f"<< Output features: {features.shape}")
    assert features.shape == (2, 128), f"Expected (2, 128), got {features.shape}"
    print("[PASS] QuantumVisionEncoder - PASSED!")
    return True


def test_language_encoder():
    print("\n" + "=" * 60)
    print("TEST 2: QuantumLanguageEncoder")
    print("=" * 60)

    config = QuantumTrainingConfig(
        model_type='BERT',
        n_layers=2,
        n_heads=4,
        d_model=128,
        d_ff=512,
        vocab_size=1000,
        dropout=0.1,
        train_compressed=False,
    )

    encoder = QuantumLanguageEncoder(config)
    encoder.eval()

    input_ids = torch.randint(0, 1000, (2, 16))
    with torch.no_grad():
        features = encoder(input_ids)

    print(f">> Input tokens: {input_ids.shape}")
    print(f"<< Output features: {features.shape}")
    assert features.shape == (2, 128), f"Expected (2, 128), got {features.shape}"
    print("[PASS] QuantumLanguageEncoder - PASSED!")
    return True


def test_quantum_fusion():
    print("\n" + "=" * 60)
    print("TEST 3: QuantumMultiModalFusion")
    print("=" * 60)

    for strategy in ['quantum_entanglement', 'cross_attention', 'concat']:
        print(f"\n>> Testing strategy: {strategy}")
        config = QuantumTrainingConfig(
            d_model=128,
            n_heads=4,
            multimodal_fusion=strategy,
        )

        fusion = QuantumMultiModalFusion(config)
        fusion.eval()

        v_feat = torch.randn(2, 128)
        l_feat = torch.randn(2, 128)

        with torch.no_grad():
            fused = fusion(v_feat, l_feat)

        print(f"   Fused output shape: {fused.shape}")
        assert fused.shape == (2, 128), f"Expected (2, 128), got {fused.shape}"
        print(f"   [PASS] Strategy {strategy} - PASSED!")

    print("[PASS] QuantumMultiModalFusion - ALL STRATEGIES PASSED!")
    return True


def test_multimodal_model():
    print("\n" + "=" * 60)
    print("TEST 4: MultiModalModel (Forward & Contrastive Loss)")
    print("=" * 60)

    config = QuantumTrainingConfig(
        model_type='MultiModal',
        n_layers=2,
        n_heads=4,
        d_model=128,
        d_ff=512,
        vocab_size=1000,
        dropout=0.1,
        multimodal_fusion='quantum_entanglement',
        train_compressed=False,
    )

    model = MultiModalModel(config)
    model.eval()

    images = torch.randn(2, 3, 224, 224)
    input_ids = torch.randint(0, 1000, (2, 16))

    # Test forward representation
    with torch.no_grad():
        fused = model(images, input_ids)
        logits = model(images, input_ids, return_logits=True)

    print(f">> Forward fused shape: {fused.shape}")
    print(f">> Forward logits shape: {logits.shape}")
    assert fused.shape == (2, 128), f"Expected (2, 128), got {fused.shape}"
    assert logits.shape == (2, 1000), f"Expected (2, 1000), got {logits.shape}"

    # Test contrastive loss
    loss = model.contrastive_loss(images, input_ids, temperature=0.07)
    print(f">> Contrastive loss: {loss.item():.4f}")
    assert loss.item() > 0.0, "Loss must be positive"
    assert not torch.isnan(loss), "Loss cannot be NaN"

    print("[PASS] MultiModalModel - PASSED!")
    return True


def test_trainer_multimodal_integration():
    print("\n" + "=" * 60)
    print("TEST 5: QuantumLLMTrainer Integration")
    print("=" * 60)

    from igqk_v4.quantum_training.trainers.quantum_llm_trainer import QuantumLLMTrainer

    config = QuantumTrainingConfig(
        model_type='MultiModal',
        n_layers=2,
        n_heads=4,
        d_model=128,
        d_ff=512,
        vocab_size=1000,
        multimodal_fusion='quantum_entanglement',
        train_compressed=False,
        use_hlwt=False,
        use_tlgt=False,
        use_fchl=False,
    )

    trainer = QuantumLLMTrainer(config)
    print(f">> Successfully initialized trainer with model: {type(trainer.model).__name__}")
    assert isinstance(trainer.model, MultiModalModel), "Trainer model must be MultiModalModel"
    print("[PASS] QuantumLLMTrainer Integration - PASSED!")
    return True


def main():
    print("=" * 60)
    print("IGQK v4.0 MULTI-MODAL TESTS (PHASE 2)")
    print("=" * 60)

    test_vision_encoder()
    test_language_encoder()
    test_quantum_fusion()
    test_multimodal_model()
    test_trainer_multimodal_integration()

    print("\n" + "=" * 60)
    print("TOTAL: 5/5 Multi-Modal test suites passed!")
    print("=" * 60)
    print("[SUCCESS] Multi-Modal AI Phase is COMPLETE!")


if __name__ == '__main__':
    main()
