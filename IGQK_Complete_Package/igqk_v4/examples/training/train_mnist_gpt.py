"""
Train QuantumGPT from scratch on discrete sequences with ternary weight compression.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset

from igqk_v4.quantum_training.trainers.quantum_training_config import QuantumTrainingConfig
from igqk_v4.quantum_training.trainers.quantum_llm_trainer import QuantumLLMTrainer


def main():
    print("=" * 80)
    print("TRAINING QUANTUM GPT (LANGUAGE MODEL) FROM SCRATCH")
    print("=" * 80)

    config = QuantumTrainingConfig(
        model_type='GPT',
        n_layers=4,
        n_heads=4,
        d_model=128,
        d_ff=512,
        vocab_size=256,
        max_seq_len=64,
        learning_rate=5e-4,
        n_epochs=5,
        use_quantum=True,
        train_compressed=True,
        compression_method='ternary',
        use_hlwt=True,
        use_tlgt=True,
        log_interval=5,
    )

    data = torch.randint(0, 256, (128, 32))
    dataloader = DataLoader(TensorDataset(data), batch_size=16)

    trainer = QuantumLLMTrainer(config)
    history, model = trainer.fit(dataloader, n_epochs=5)

    print("\n✅ Training complete! Initial loss:", history['train_loss'][0], "Final loss:", history['train_loss'][-1])


if __name__ == '__main__':
    main()
