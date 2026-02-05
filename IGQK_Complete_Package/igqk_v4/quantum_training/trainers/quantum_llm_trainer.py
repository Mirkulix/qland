"""
Quantum LLM Trainer - The Core of IGQK v4.0

This trainer implements quantum-based training of neural networks from scratch,
combining v2.0 vision with v4.0 advanced features.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, Tuple
import time
import numpy as np
from pathlib import Path

from .quantum_training_config import QuantumTrainingConfig


class QuantumLLMTrainer:
    """
    Quantum LLM Trainer for IGQK v4.0.

    This trainer unifies:
    - Quantum Gradient Flow (QGF) from v1.0
    - Training from Scratch (v2.0 vision)
    - Advanced Math (HLWT, TLGT, FCHL) from Roadmap
    - Multi-Modal Support (v4.0)
    - Distributed Training (v4.0)
    """

    def __init__(
        self,
        config: QuantumTrainingConfig,
        model: Optional[nn.Module] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize Quantum LLM Trainer.

        Args:
            config: Training configuration
            model: Pre-initialized model (optional, will be created if None)
            device: Device to train on (auto-detected if None)
        """
        self.config = config
        self.device = device or self._detect_device()

        # Initialize or create model
        if model is None:
            self.model = self._create_model()
        else:
            self.model = model

        self.model = self.model.to(self.device)

        # Initialize quantum optimizer
        self.optimizer = self._create_optimizer()

        # Initialize advanced math modules
        self.hlwt = None
        self.tlgt = None
        self.fchl = None
        if config.use_hlwt:
            self.hlwt = self._init_hlwt()
        if config.use_tlgt:
            self.tlgt = self._init_tlgt()
        if config.use_fchl:
            self.fchl = self._init_fchl()

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'quantum_entropy': [],
            'quantum_purity': [],
        }

        # Distributed training setup
        if config.distributed:
            self._setup_distributed()

        print(f" QuantumLLMTrainer initialized on {self.device}")
        print(f"   Model: {self._count_parameters():,} parameters")
        if config.train_compressed:
            print(f"   Training Mode: DIRECT COMPRESSED ({config.compression_method})")
        else:
            print(f"   Training Mode: STANDARD")

    def _detect_device(self) -> str:
        """Auto-detect best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _create_model(self) -> nn.Module:
        """Create model based on configuration."""
        if self.config.model_type == 'GPT':
            from ...models.gpt import QuantumGPT  # Will implement
            return QuantumGPT(self.config)
        elif self.config.model_type == 'BERT':
            from ...models.bert import QuantumBERT
            return QuantumBERT(self.config)
        elif self.config.model_type == 'ViT':
            from ...models.vit import QuantumViT
            return QuantumViT(self.config)
        elif self.config.model_type == 'MultiModal':
            from ...multimodal.models.multimodal_model import MultiModalModel
            return MultiModalModel(self.config)
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")

    def _create_optimizer(self):
        """Create quantum-enhanced optimizer."""
        # Import from v1.0 (existing) or create enhanced version
        try:
            from ...igqk.optimizers.igqk_optimizer import IGQKOptimizer
            base_optimizer = IGQKOptimizer
        except ImportError:
            # Fallback: Use PyTorch Adam with quantum enhancements
            print("  IGQKOptimizer not found, using enhanced Adam")
            base_optimizer = torch.optim.AdamW

        if self.config.use_quantum and hasattr(base_optimizer, '__name__') and 'IGQK' in base_optimizer.__name__:
            # Use quantum optimizer
            optimizer = base_optimizer(
                self.model.parameters(),
                lr=self.config.learning_rate,
                hbar=self.config.hbar,
                gamma=self.config.gamma,
                use_quantum=True,
            )
        else:
            # Classical optimizer
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )

        return optimizer

    def _init_hlwt(self):
        """Initialize Hybrid Laplace-Wavelet Transform."""
        from ...theory.hlwt.hybrid_laplace_wavelet import HybridLaplaceWavelet
        return HybridLaplaceWavelet(
            grid_size=self.config.hlwt_wavelet_grid,
            wavelet_type=self.config.hlwt_wavelet_type,
        )

    def _init_tlgt(self):
        """Initialize Ternary Lie Group Theory."""
        from ...theory.tlgt.ternary_lie_group import TernaryLieGroup
        return TernaryLieGroup(
            manifold_dim=self.config.tlgt_manifold_dim,
            geodesic_steps=self.config.tlgt_geodesic_steps,
        )

    def _init_fchl(self):
        """Initialize Fractional Calculus Hebbian Learning."""
        from ...theory.fchl.fractional_hebbian import FractionalHebbian
        return FractionalHebbian(
            alpha=self.config.fchl_alpha,
            memory_length=self.config.fchl_memory_length,
        )

    def _setup_distributed(self):
        """Setup distributed training."""
        import torch.distributed as dist

        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')

        if self.config.strategy == 'ddp':
            from torch.nn.parallel import DistributedDataParallel as DDP
            self.model = DDP(self.model)
        elif self.config.strategy == 'fsdp':
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            self.model = FSDP(self.model)
        else:
            raise ValueError(f"Unknown distributed strategy: {self.config.strategy}")

    def _count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            dataloader: Training data loader

        Returns:
            Dictionary of metrics
        """
        self.model.train()
        total_loss = 0.0
        quantum_metrics = {'entropy': [], 'purity': []}

        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            if isinstance(batch, (tuple, list)):
                batch = [b.to(self.device) if isinstance(b, torch.Tensor) else b for b in batch]
            elif isinstance(batch, dict):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            else:
                batch = batch.to(self.device)

            # Forward pass
            loss = self._compute_loss(batch)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

            # Optimizer step (with quantum dynamics if enabled)
            if self.config.use_quantum:
                # Quantum Gradient Flow step
                self.optimizer.step()

                # Track quantum metrics
                if hasattr(self.optimizer, 'entropy'):
                    quantum_metrics['entropy'].append(self.optimizer.entropy())
                if hasattr(self.optimizer, 'purity'):
                    quantum_metrics['purity'].append(self.optimizer.purity())
            else:
                # Classical optimizer step
                self.optimizer.step()

            # Apply advanced math frameworks
            if self.config.use_hlwt and self.hlwt is not None:
                # HLWT: Adaptive learning rate based on local stability
                adaptive_lr = self.hlwt.compute_adaptive_lr(loss.item())
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = adaptive_lr

            if self.config.use_tlgt and self.tlgt is not None:
                # TLGT: Geodesic projection for ternary weights
                if self.config.train_compressed:
                    self.tlgt.project_to_manifold(self.model)

            if self.config.use_fchl and self.fchl is not None:
                # FCHL: Update fractional memory
                self.fchl.update_memory(self.model.parameters())

            total_loss += loss.item()
            self.global_step += 1

            # Logging
            if self.global_step % self.config.log_interval == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"Step {self.global_step}: Loss = {avg_loss:.4f}", end="")
                if quantum_metrics['entropy']:
                    print(f", Entropy = {np.mean(quantum_metrics['entropy']):.3f}", end="")
                print()

        # Epoch metrics
        metrics = {
            'loss': total_loss / len(dataloader),
            'entropy': np.mean(quantum_metrics['entropy']) if quantum_metrics['entropy'] else 0.0,
            'purity': np.mean(quantum_metrics['purity']) if quantum_metrics['purity'] else 0.0,
        }

        return metrics

    def _compute_loss(self, batch) -> torch.Tensor:
        """Compute loss for a batch."""
        # This depends on model type and task
        # For language models: next token prediction
        if self.config.model_type in ['GPT', 'BERT', 'T5']:
            if isinstance(batch, dict):
                input_ids = batch['input_ids']
                labels = batch.get('labels', input_ids)
            else:
                input_ids = batch[0]
                labels = batch[1] if len(batch) > 1 else input_ids

            outputs = self.model(input_ids)
            loss = nn.functional.cross_entropy(
                outputs.view(-1, outputs.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )

        # For vision models: classification
        elif self.config.model_type == 'ViT':
            images, labels = batch
            outputs = self.model(images)
            loss = nn.functional.cross_entropy(outputs, labels)

        # For multi-modal: contrastive loss
        elif self.config.model_type == 'MultiModal':
            loss = self.model.compute_loss(batch)

        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")

        return loss

    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate model on validation set."""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (tuple, list)):
                    batch = [b.to(self.device) if isinstance(b, torch.Tensor) else b for b in batch]
                elif isinstance(batch, dict):
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                else:
                    batch = batch.to(self.device)

                loss = self._compute_loss(batch)
                total_loss += loss.item()

        metrics = {'loss': total_loss / len(dataloader)}
        return metrics

    def fit(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        n_epochs: Optional[int] = None,
    ) -> Tuple[Dict, nn.Module]:
        """
        Complete training loop.

        Args:
            train_dataloader: Training data
            val_dataloader: Validation data (optional)
            n_epochs: Number of epochs (uses config if None)

        Returns:
            (history, trained_model)
        """
        n_epochs = n_epochs or self.config.n_epochs

        print("=" * 70)
        print("🚀 STARTING QUANTUM TRAINING")
        print("=" * 70)
        self.config.summary()
        print("=" * 70)

        start_time = time.time()

        for epoch in range(n_epochs):
            self.current_epoch = epoch
            print(f"\n📊 Epoch {epoch + 1}/{n_epochs}")

            # Train
            train_metrics = self.train_epoch(train_dataloader)
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['quantum_entropy'].append(train_metrics['entropy'])
            self.history['quantum_purity'].append(train_metrics['purity'])

            print(f"   Train Loss: {train_metrics['loss']:.4f}")

            # Validate
            if val_dataloader is not None:
                val_metrics = self.validate(val_dataloader)
                self.history['val_loss'].append(val_metrics['loss'])
                print(f"   Val Loss: {val_metrics['loss']:.4f}")

            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt")

        # Final compression (if not already trained compressed)
        if not self.config.train_compressed and self.config.compression_method != 'none':
            print("\n🗜️  Applying final compression...")
            self._compress_model()

        total_time = time.time() - start_time
        print("=" * 70)
        print(f" TRAINING COMPLETED in {total_time/60:.1f} minutes")
        print("=" * 70)

        return self.history, self.model

    def _compress_model(self):
        """Apply compression to trained model."""
        try:
            from ...igqk.compression.projectors import compress_model
            compress_model(
                self.model,
                method=self.config.compression_method,
                target_ratio=self.config.compression_target,
            )
            print(f" Model compressed with {self.config.compression_method} method")
        except ImportError:
            print("  Compression module not found, skipping")

    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.to_dict(),
            'history': self.history,
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
        }
        torch.save(checkpoint, path)
        print(f"💾 Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        self.global_step = checkpoint['global_step']
        self.current_epoch = checkpoint['current_epoch']
        print(f"📂 Checkpoint loaded: {path}")


# Quick training function for convenience
def train_quantum_model(
    config: QuantumTrainingConfig,
    train_data: DataLoader,
    val_data: Optional[DataLoader] = None,
) -> nn.Module:
    """
    Quick function to train a quantum model.

    Args:
        config: Training configuration
        train_data: Training dataloader
        val_data: Validation dataloader

    Returns:
        Trained and compressed model
    """
    trainer = QuantumLLMTrainer(config)
    history, model = trainer.fit(train_data, val_data)
    return model


if __name__ == "__main__":
    # Test the trainer with dummy data
    print("🧪 Testing QuantumLLMTrainer...\n")

    from .quantum_training_config import ConfigPresets

    config = ConfigPresets.small_gpt()

    # Create dummy data
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 100

        def __getitem__(self, idx):
            return {
                'input_ids': torch.randint(0, config.vocab_size, (config.max_seq_len,)),
                'labels': torch.randint(0, config.vocab_size, (config.max_seq_len,)),
            }

    train_loader = DataLoader(DummyDataset(), batch_size=4, shuffle=True)

    print("Creating trainer...")
    trainer = QuantumLLMTrainer(config)

    print("\nTrainer created successfully!")
    print(f"Model parameters: {trainer._count_parameters():,}")
