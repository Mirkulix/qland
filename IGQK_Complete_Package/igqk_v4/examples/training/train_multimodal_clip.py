"""
Train MultiModalModel CLIP-style (Vision + Language) with Quantum Entanglement Fusion.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset

from igqk_v4.quantum_training.trainers.quantum_training_config import QuantumTrainingConfig
from igqk_v4.quantum_training.trainers.quantum_llm_trainer import QuantumLLMTrainer


def main():
    print("=" * 80)
    print("TRAINING MULTI-MODAL MODEL (CLIP-STYLE) WITH QUANTUM FUSION")
    print("=" * 80)

    config = QuantumTrainingConfig(
        model_type='MultiModal',
        n_layers=4,
        n_heads=4,
        d_model=128,
        d_ff=512,
        vocab_size=1000,
        multimodal_modalities=['vision', 'language'],
        multimodal_fusion='quantum_entanglement',
        learning_rate=5e-4,
        n_epochs=5,
        use_quantum=True,
        train_compressed=True,
        compression_method='ternary',
        use_hlwt=True,
        use_tlgt=True,
        log_interval=5,
    )

    images = torch.randn(128, 3, 224, 224)
    tokens = torch.randint(0, 1000, (128, 20))
    dataloader = DataLoader(TensorDataset(images, tokens), batch_size=16)

    trainer = QuantumLLMTrainer(config)
    history, model = trainer.fit(dataloader, n_epochs=5)

    print("\n✅ Multi-Modal Training complete! Initial contrastive loss:", history['train_loss'][0], "Final contrastive loss:", history['train_loss'][-1])


if __name__ == '__main__':
    main()
